import gradio as gr
from gradio_leaderboard import Leaderboard
import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import threading
from dotenv import load_dotenv
import pandas as pd
import random
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='SWE Agent Commit Leaderboard')
parser.add_argument('--debug', '--DEBUG', action='store_true',
                    help='Enable debug mode (limits commit retrieval to 10 per query pattern)')
parser.add_argument('--no-debug', '--production', action='store_true',
                    help='Explicitly disable debug mode (force production mode)')
args = parser.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================

# DEBUG MODE: Set to True to limit commit retrieval for testing
# When enabled, only fetches up to 10 commits per query pattern per agent
# Priority: 1) Command-line args, 2) Environment variable, 3) Default (False)
if args.no_debug:
    DEBUG_MODE = False
elif args.debug:
    DEBUG_MODE = True
else:
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 'yes')

# In-memory cache for debug mode (data persists during session but NOT saved to HF)
DEBUG_COMMIT_METADATA_CACHE = defaultdict(list)

AGENTS_REPO = "SWE-Arena/swe_agents"  # HuggingFace dataset for agent metadata
COMMIT_METADATA_REPO = "SWE-Arena/commit_metadata"  # HuggingFace dataset for commit metadata
LEADERBOARD_TIME_FRAME_DAYS = 180  # Time frame for leaderboard (past 6 months)

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Website", "string"),
    ("Total Commits", "number"),
    ("Stable Commits", "number"),
    ("Retention Rate (%)", "number"),
]

# =============================================================================
# JSONL FILE OPERATIONS
# =============================================================================

def load_jsonl(filename):
    """Load JSONL file and return list of dictionaries."""
    if not os.path.exists(filename):
        return []
    
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def cache_to_dict(cache_list):
    """Convert list of cache entries to dictionary by identifier."""
    return {entry['github_identifier']: entry for entry in cache_list}


def dict_to_cache(cache_dict):
    """Convert dictionary back to list of values."""
    return list(cache_dict.values())


def normalize_date_format(date_string):
    """
    Convert date strings to standardized ISO 8601 format with Z suffix.
    Handles both old format (2025-10-15T23:23:47.983068) and new format (2025-10-15T23:23:47Z).
    """
    if not date_string or date_string == 'N/A':
        return 'N/A'
    
    try:
        # Parse the date string (handles both with and without microseconds)
        if '.' in date_string:
            # Old format with microseconds
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        else:
            # Already in correct format or GitHub format
            return date_string
        
        # Convert to standardized format
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


# =============================================================================
# GITHUB API OPERATIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.

    Returns the final requests.Response on success or non-retryable status, or None after exhausting retries.
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers or {},
                params=params,
                json=json_body,
                data=data,
                timeout=timeout
            )

            status = resp.status_code

            # Success
            if 200 <= status < 300:
                return resp

            # Rate limits or server errors -> retry with backoff
            if status in (403, 429) or 500 <= status < 600:
                wait = None

                # Prefer Retry-After when present
                retry_after = resp.headers.get('Retry-After') or resp.headers.get('retry-after')
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = None

                # Fallback to X-RateLimit-Reset when 403/429
                if wait is None and status in (403, 429):
                    reset_hdr = resp.headers.get('X-RateLimit-Reset') or resp.headers.get('x-ratelimit-reset')
                    if reset_hdr:
                        try:
                            reset_ts = int(float(reset_hdr))
                            wait = max(reset_ts - time.time() + 2, 1)
                        except Exception:
                            wait = None

                # Final fallback: exponential backoff with jitter
                if wait is None:
                    wait = delay + random.uniform(0, 0.5)

                # Cap individual wait to avoid extreme sleeps
                wait = max(1.0, min(wait, 120.0))
                print(f"GitHub API {status}. Backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            # Non-retryable error; return response for caller to handle
            return resp

        except requests.RequestException as e:
            # Network error -> retry with backoff
            wait = delay + random.uniform(0, 0.5)
            wait = max(1.0, min(wait, 60.0))
            print(f"Request error: {e}. Retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)

    print(f"Exceeded max retries for {url}")
    return None

def get_github_token():
    """Get GitHub token from environment variables."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found. API rate limits: 60/hour (authenticated: 5000/hour)")
    return token


def validate_github_username(identifier):
    """Verify that a GitHub identifier exists with backoff-aware requests."""
    try:
        token = get_github_token()
        headers = {'Authorization': f'token {token}'} if token else {}
        url = f'https://api.github.com/users/{identifier}'
        response = request_with_backoff('GET', url, headers=headers, max_retries=1)
        if response is None:
            return False, "Validation error: network/rate limit exhausted"
        if response.status_code == 200:
            return True, "Username is valid"
        elif response.status_code == 404:
            return False, "GitHub identifier not found"
        else:
            return False, f"Validation error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def fetch_commits_with_time_partition(base_query, start_date, end_date, headers, commits_by_sha, debug_limit=None, depth=0, check_reverts=True):
    """
    Fetch commits within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.
    Optionally checks revert status immediately for each commit.

    Args:
        debug_limit: If set, stops fetching after this many NEW commits total across all partitions (for testing)
        depth: Current recursion depth (for tracking)
        check_reverts: If True, checks revert status immediately for each commit (default: True)

    Returns the number of commits found in this time partition.
    """
    # Format dates for GitHub search (YYYY-MM-DD)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Add date range to query (use committer-date for commits)
    query = f'{base_query} committer-date:{start_str}..{end_str}'

    indent = "  " + "  " * depth
    print(f"{indent}Searching range {start_str} to {end_str}...")

    page = 1
    per_page = 100
    total_in_partition = 0

    while True:
        # Check debug limit GLOBALLY (total unique commits across all partitions)
        if debug_limit is not None and len(commits_by_sha) >= debug_limit:
            print(f"{indent}  🐛 DEBUG MODE: Reached global limit of {debug_limit} commits, stopping...")
            return total_in_partition
        url = 'https://api.github.com/search/commits'
        params = {
            'q': query,
            'per_page': per_page,
            'page': page,
            'sort': 'committer-date',
            'order': 'asc'
        }
        # Add required header for commit search API
        headers_with_accept = headers.copy() if headers else {}
        headers_with_accept['Accept'] = 'application/vnd.github.cloak-preview+json'

        try:
            response = request_with_backoff('GET', url, headers=headers_with_accept, params=params)
            if response is None:
                print(f"{indent}  Error: retries exhausted for range {start_str} to {end_str}")
                return total_in_partition

            if response.status_code != 200:
                print(f"{indent}  Error: HTTP {response.status_code} for range {start_str} to {end_str}")
                return total_in_partition

            data = response.json()
            total_count = data.get('total_count', 0)
            items = data.get('items', [])

            if not items:
                break

            # Add commits to global dict (keyed by SHA)
            # Check revert status immediately for each new commit
            for commit in items:
                commit_sha = commit.get('sha')
                if commit_sha and commit_sha not in commits_by_sha:
                    # Check revert status immediately if enabled
                    if check_reverts:
                        revert_status = check_single_commit_revert_status(commit_sha, headers)
                        # Store revert status in commit object for later extraction
                        commit['_revert_status'] = revert_status
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)

                    commits_by_sha[commit_sha] = commit
                    total_in_partition += 1

            # Check if we hit the 1000-result limit
            if total_count > 1000 and page == 10:
                print(f"{indent}  ⚠️ Hit 1000-result limit ({total_count} total). Splitting time range...")

                # Calculate time range in days
                time_diff = end_date - start_date
                days_diff = time_diff.days

                # Use aggressive splitting for large ranges or deep recursion
                # Split into 4 parts if range is > 30 days, otherwise split in half
                if days_diff > 30 or depth > 5:
                    # Split into 4 parts for more aggressive partitioning
                    quarter_diff = time_diff / 4
                    split_dates = [
                        start_date,
                        start_date + quarter_diff,
                        start_date + quarter_diff * 2,
                        start_date + quarter_diff * 3,
                        end_date
                    ]

                    total_from_splits = 0
                    for i in range(4):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges
                        if i > 0:
                            split_start = split_start + timedelta(days=1)

                        count = fetch_commits_with_time_partition(
                            base_query, split_start, split_end, headers, commits_by_sha, debug_limit, depth + 1, check_reverts
                        )
                        total_from_splits += count

                    return total_from_splits
                else:
                    # Binary split for smaller ranges
                    mid_date = start_date + time_diff / 2

                    # Recursively fetch both halves
                    count1 = fetch_commits_with_time_partition(
                        base_query, start_date, mid_date, headers, commits_by_sha, debug_limit, depth + 1, check_reverts
                    )
                    count2 = fetch_commits_with_time_partition(
                        base_query, mid_date + timedelta(days=1), end_date, headers, commits_by_sha, debug_limit, depth + 1, check_reverts
                    )

                    return count1 + count2

            # Normal pagination: check if there are more pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"{indent}  Error fetching range {start_str} to {end_str}: {str(e)}")
            return total_in_partition

    if total_in_partition > 0:
        print(f"{indent}  ✓ Found {total_in_partition} commits in range {start_str} to {end_str}")

    return total_in_partition


def fetch_prs_with_time_partition(base_query, start_date, end_date, headers, prs_by_id, debug_limit=None, depth=0):
    """
    Fetch PRs within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.
    Supports splitting by day, hour, minute, and second as needed.

    Args:
        debug_limit: If set, stops fetching after this many PRs (for testing)
        depth: Current recursion depth (for tracking)

    Returns the number of PRs found in this time partition.
    """
    # Calculate time difference
    time_diff = end_date - start_date
    total_seconds = time_diff.total_seconds()

    # Determine granularity and format dates accordingly
    if total_seconds >= 86400:  # >= 1 day
        # Use day granularity (YYYY-MM-DD)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    elif total_seconds >= 3600:  # >= 1 hour but < 1 day
        # Use hour granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:00:00Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:59:59Z')
    elif total_seconds >= 60:  # >= 1 minute but < 1 hour
        # Use minute granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:00Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:59Z')
    else:  # < 1 minute
        # Use second granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Add date range to query
    query = f'{base_query} created:{start_str}..{end_str}'

    indent = "  " + "  " * depth
    print(f"{indent}Searching range {start_str} to {end_str}...")

    page = 1
    per_page = 100
    total_in_partition = 0

    while True:
        # Check debug limit
        if debug_limit is not None and total_in_partition >= debug_limit:
            print(f"{indent}  🐛 DEBUG MODE: Reached limit of {debug_limit} PRs, stopping...")
            return total_in_partition
        url = 'https://api.github.com/search/issues'
        params = {
            'q': query,
            'per_page': per_page,
            'page': page,
            'sort': 'created',
            'order': 'asc'
        }

        try:
            response = request_with_backoff('GET', url, headers=headers, params=params)
            if response is None:
                print(f"{indent}  Error: retries exhausted for range {start_str} to {end_str}")
                return total_in_partition

            if response.status_code != 200:
                print(f"{indent}  Error: HTTP {response.status_code} for range {start_str} to {end_str}")
                return total_in_partition

            data = response.json()
            total_count = data.get('total_count', 0)
            items = data.get('items', [])

            if not items:
                break

            # Add PRs to global dict
            for pr in items:
                pr_id = pr.get('id')
                if pr_id and pr_id not in prs_by_id:
                    prs_by_id[pr_id] = pr
                    total_in_partition += 1

            # Check if we hit the 1000-result limit
            if total_count > 1000 and page == 10:
                print(f"{indent}  ⚠️ Hit 1000-result limit ({total_count} total). Splitting time range...")

                # Determine how to split based on time range duration
                if total_seconds < 2:  # Less than 2 seconds - can't split further
                    print(f"{indent}  ⚠️ Cannot split further (range < 2 seconds). Some results may be missing.")
                    break

                elif total_seconds < 120:  # Less than 2 minutes - split by seconds
                    # Split into 2-4 parts depending on range
                    num_splits = min(4, max(2, int(total_seconds / 30)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 second to start)
                        if i > 0:
                            split_start = split_start + timedelta(seconds=1)

                        count = fetch_prs_with_time_partition(
                            base_query, split_start, split_end, headers, prs_by_id, debug_limit, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                elif total_seconds < 7200:  # Less than 2 hours - split by minutes
                    # Split into 2-4 parts
                    num_splits = min(4, max(2, int(total_seconds / 1800)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 minute to start)
                        if i > 0:
                            split_start = split_start + timedelta(minutes=1)

                        count = fetch_prs_with_time_partition(
                            base_query, split_start, split_end, headers, prs_by_id, debug_limit, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                elif total_seconds < 172800:  # Less than 2 days - split by hours
                    # Split into 2-4 parts
                    num_splits = min(4, max(2, int(total_seconds / 43200)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 hour to start)
                        if i > 0:
                            split_start = split_start + timedelta(hours=1)

                        count = fetch_prs_with_time_partition(
                            base_query, split_start, split_end, headers, prs_by_id, debug_limit, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                else:  # 2+ days - split by days
                    days_diff = time_diff.days

                    # Use aggressive splitting for large ranges or deep recursion
                    # Split into 4 parts if range is > 30 days, otherwise split in half
                    if days_diff > 30 or depth > 5:
                        # Split into 4 parts for more aggressive partitioning
                        quarter_diff = time_diff / 4
                        split_dates = [
                            start_date,
                            start_date + quarter_diff,
                            start_date + quarter_diff * 2,
                            start_date + quarter_diff * 3,
                            end_date
                        ]

                        total_from_splits = 0
                        for i in range(4):
                            split_start = split_dates[i]
                            split_end = split_dates[i + 1]
                            # Avoid overlapping ranges
                            if i > 0:
                                split_start = split_start + timedelta(days=1)

                            count = fetch_prs_with_time_partition(
                                base_query, split_start, split_end, headers, prs_by_id, debug_limit, depth + 1
                            )
                            total_from_splits += count

                        return total_from_splits
                    else:
                        # Binary split for smaller ranges
                        mid_date = start_date + time_diff / 2

                        # Recursively fetch both halves
                        count1 = fetch_prs_with_time_partition(
                            base_query, start_date, mid_date, headers, prs_by_id, debug_limit, depth + 1
                        )
                        count2 = fetch_prs_with_time_partition(
                            base_query, mid_date + timedelta(days=1), end_date, headers, prs_by_id, debug_limit, depth + 1
                        )

                        return count1 + count2

            # Normal pagination: check if there are more pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"{indent}  Error fetching range {start_str} to {end_str}: {str(e)}")
            return total_in_partition

    if total_in_partition > 0:
        print(f"{indent}  ✓ Found {total_in_partition} PRs in range {start_str} to {end_str}")

    return total_in_partition


def check_single_commit_revert_status(sha, headers):
    """
    Check if a single commit has been reverted by searching GitHub for revert commits.

    Args:
        sha: Commit SHA to check
        headers: HTTP headers for GitHub API

    Returns:
        Dictionary with 'is_reverted' (bool) and 'revert_at' (date string or None)
    """
    if not sha:
        return {'is_reverted': False, 'revert_at': None}

    # Search for commits that mention this SHA in a revert context
    # Use abbreviated SHA (first 7 characters) which is commonly used in reverts
    sha_abbr = sha[:7]
    revert_query = f'"This reverts commit {sha_abbr}"'

    try:
        url = 'https://api.github.com/search/commits'
        params = {
            'q': revert_query,
            'per_page': 1  # We only need to know if ANY revert exists
        }

        headers_with_accept = headers.copy() if headers else {}
        headers_with_accept['Accept'] = 'application/vnd.github.cloak-preview+json'

        response = request_with_backoff('GET', url, headers=headers_with_accept, params=params, max_retries=3)

        if response and response.status_code == 200:
            data = response.json()
            total_count = data.get('total_count', 0)

            if total_count > 0:
                # This commit has been reverted
                items = data.get('items', [])
                if items:
                    # Get the date of the first revert commit
                    revert_commit = items[0]
                    revert_at = revert_commit.get('commit', {}).get('author', {}).get('date')
                    return {'is_reverted': True, 'revert_at': revert_at}

        return {'is_reverted': False, 'revert_at': None}

    except Exception as e:
        print(f"   Warning: Could not check revert status for {sha_abbr}: {e}")
        return {'is_reverted': False, 'revert_at': None}


def extract_commit_metadata(commit, revert_status=None):
    """
    Extract minimal commit metadata for efficient storage.
    Only keeps essential fields: html_url, commit_at, revert_at, is_reverted.
    Note: agent_name is not stored as it's inferred from the folder structure.

    Commit stability:
    - is_reverted: True if commit has been explicitly reverted, False otherwise
    - revert_at: Date when commit was reverted (if applicable)

    Stable Commit = commit that remains in repository history and has not been reverted

    Args:
        commit: Full commit object from GitHub API
        revert_status: Optional dict with 'is_reverted' and 'revert_at' (if already checked)
    """
    # Extract dates
    commit_at = commit.get('commit', {}).get('author', {}).get('date')

    # Use provided revert status or default to False
    if revert_status is None:
        revert_status = {'is_reverted': False, 'revert_at': None}

    return {
        'html_url': commit.get('html_url'),
        'commit_at': commit_at,
        'revert_at': revert_status.get('revert_at'),
        'is_reverted': revert_status.get('is_reverted', False),
        'sha': commit.get('sha')  # Store SHA for revert detection
    }


def detect_reverted_commits(metadata_list, headers, token):
    """
    DEPRECATED: This function is no longer used. Revert detection is now done inline
    during commit retrieval in fetch_commits_with_time_partition().

    Legacy function that detects which commits have been reverted by searching for revert commits.
    This approach checked all commits in batch after retrieval, which was less efficient.

    Searches GitHub for commits containing "This reverts commit {sha}" in the message.
    Updates metadata_list in-place with revert information.

    In DEBUG MODE: Skips revert detection to avoid API rate limits.

    Args:
        metadata_list: List of commit metadata dictionaries
        headers: HTTP headers for GitHub API
        token: GitHub API token

    Returns:
        Updated metadata_list with revert information
    """
    if not metadata_list:
        return metadata_list

    # In debug mode, skip revert detection to avoid excessive API calls
    if DEBUG_MODE:
        print(f"   🐛 DEBUG MODE: Skipping revert detection for {len(metadata_list)} commits")
        return metadata_list

    # Build a map of SHA to metadata for quick lookup
    sha_to_metadata = {meta['sha']: meta for meta in metadata_list if meta.get('sha')}

    reverted_count = 0

    # For each commit, search for potential revert commits
    # We'll search for commits with message containing the SHA (full or abbreviated)
    for metadata in metadata_list:
        sha = metadata.get('sha')
        if not sha:
            continue

        # Search for commits that mention this SHA in a revert context
        # Use abbreviated SHA (first 7 characters) which is commonly used in reverts
        sha_abbr = sha[:7]

        # Search for revert commits mentioning this SHA
        revert_query = f'"This reverts commit {sha_abbr}"'

        try:
            url = 'https://api.github.com/search/commits'
            params = {
                'q': revert_query,
                'per_page': 1  # We only need to know if ANY revert exists
            }

            headers_with_accept = headers.copy() if headers else {}
            headers_with_accept['Accept'] = 'application/vnd.github.cloak-preview+json'

            response = request_with_backoff('GET', url, headers=headers_with_accept, params=params, max_retries=3)

            if response and response.status_code == 200:
                data = response.json()
                total_count = data.get('total_count', 0)

                if total_count > 0:
                    # This commit has been reverted
                    items = data.get('items', [])
                    if items:
                        # Get the date of the first revert commit
                        revert_commit = items[0]
                        revert_at = revert_commit.get('commit', {}).get('author', {}).get('date')

                        metadata['is_reverted'] = True
                        metadata['revert_at'] = revert_at
                        reverted_count += 1

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"   Warning: Could not check revert status for {sha_abbr}: {e}")
            continue

    if reverted_count > 0:
        print(f"   ✓ Found {reverted_count} reverted commits")
    else:
        print(f"   ✓ No reverted commits found")

    return metadata_list


def fetch_all_commits_metadata(identifier, agent_name, token=None, start_from_date=None, exclude_dates=None):
    """
    Fetch commits associated with a GitHub user or bot for the past 6 months.
    Returns lightweight metadata instead of full commit objects.

    This function employs time-based partitioning to navigate GitHub's 1000-result limit per query.
    It searches using the query pattern:
    - is:commit author:{identifier} (commits authored by the bot)

    NEW: Revert status is checked IMMEDIATELY for each commit during retrieval by searching for:
    - "This reverts commit {sha}" in commit messages
    This is more efficient than batch checking after all commits are fetched.

    In DEBUG MODE: Revert checking is skipped to avoid excessive API calls.

    Args:
        identifier: GitHub username or bot identifier
        agent_name: Human-readable name of the agent for metadata purposes
        token: GitHub API token for authentication
        start_from_date: Only fetch commits created after this date (for incremental updates)
        exclude_dates: Set of date objects to exclude from mining (dates that have already been processed)

    Returns:
        List of dictionaries containing minimal commit metadata with revert status
    """
    headers = {'Authorization': f'token {token}'} if token else {}

    # Debug mode: limit commit retrieval for testing
    debug_limit_per_pattern = 10 if DEBUG_MODE else None

    if DEBUG_MODE:
        print(f"\n🐛 DEBUG MODE ENABLED: Limiting to {debug_limit_per_pattern} commits per query pattern")

    # Define query pattern for commits:
    # author pattern: commits authored by the identifier
    stripped_id = identifier.replace('[bot]', '')
    query_patterns = []

    # Add author pattern for commits
    query_patterns.append(f'is:commit author:{identifier}')
    if stripped_id != identifier:
        query_patterns.append(f'is:commit author:{stripped_id}')

    # Use a dict to deduplicate commits by SHA
    commits_by_sha = {}

    # Define time range: past 6 months only (or from start_from_date if specified)
    current_time = datetime.now(timezone.utc)
    six_months_ago = current_time - timedelta(days=180)  # ~6 months

    if start_from_date:
        # Use start_from_date but ensure it's not older than 6 months
        start_date = max(start_from_date, six_months_ago)
    else:
        start_date = six_months_ago

    # End date is current time
    end_date = current_time

    for query_pattern in query_patterns:
        print(f"\n🔍 Searching with query: {query_pattern}")
        print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        pattern_start_time = time.time()
        initial_count = len(commits_by_sha)

        # Fetch with time partitioning and inline revert checking
        # In DEBUG MODE, skip revert checking to avoid excessive API calls
        check_reverts = not DEBUG_MODE
        commits_found = fetch_commits_with_time_partition(
            query_pattern,
            start_date,
            end_date,
            headers,
            commits_by_sha,
            debug_limit_per_pattern,
            check_reverts=check_reverts
        )

        pattern_duration = time.time() - pattern_start_time
        new_commits = len(commits_by_sha) - initial_count

        print(f"   ✓ Pattern complete: {new_commits} new commits found ({commits_found} total fetched, {len(commits_by_sha) - initial_count - (commits_found - new_commits)} duplicates)")
        print(f"   ⏱️ Time taken: {pattern_duration:.1f} seconds")

        # Delay between different query patterns (shorter in debug mode)
        time.sleep(0.2 if DEBUG_MODE else 1.0)

    # Convert to lightweight metadata
    all_commits = list(commits_by_sha.values())

    # Filter out commits from excluded dates if specified
    if exclude_dates:
        filtered_commits = []
        excluded_count = 0
        for commit in all_commits:
            commit_at = commit.get('commit', {}).get('author', {}).get('date')
            if commit_at:
                try:
                    dt = datetime.fromisoformat(commit_at.replace('Z', '+00:00'))
                    commit_date = dt.date()
                    if commit_date not in exclude_dates:
                        filtered_commits.append(commit)
                    else:
                        excluded_count += 1
                except Exception:
                    filtered_commits.append(commit)  # Keep commits with unparseable dates
            else:
                filtered_commits.append(commit)  # Keep commits without commit_at

        if excluded_count > 0:
            print(f"   ⏭️ Skipped {excluded_count} commits from already-mined dates")
        all_commits = filtered_commits

    if DEBUG_MODE:
        print(f"\n✅ COMPLETE (DEBUG MODE): Found {len(all_commits)} unique commits for {identifier}")
        print(f"   Note: In production mode, this would fetch ALL commits and check revert status")
        print(f"   🐛 DEBUG MODE: Skipped inline revert detection")
    else:
        print(f"\n✅ COMPLETE: Found {len(all_commits)} unique commits for {identifier}")
    print(f"📦 Extracting minimal metadata...")

    # Extract metadata for each commit
    # Revert status was already checked inline during fetch (unless in DEBUG_MODE)
    metadata_list = []
    for commit in all_commits:
        # Get inline revert status if it was checked
        revert_status = commit.get('_revert_status')
        metadata_list.append(extract_commit_metadata(commit, revert_status))

    # Count reverted commits
    reverted_count = sum(1 for meta in metadata_list if meta.get('is_reverted', False))
    if reverted_count > 0:
        print(f"   ✓ Found {reverted_count} reverted commits")
    else:
        print(f"   ✓ No reverted commits found")

    # Calculate memory savings
    import sys
    original_size = sys.getsizeof(str(all_commits))
    metadata_size = sys.getsizeof(str(metadata_list))
    savings_pct = ((original_size - metadata_size) / original_size * 100) if original_size > 0 else 0

    print(f"💾 Memory efficiency: {original_size // 1024}KB → {metadata_size // 1024}KB (saved {savings_pct:.1f}%)")

    return metadata_list


def calculate_commit_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of commit metadata (lightweight objects).
    Works with minimal metadata: html_url, commit_at, revert_at, is_reverted, sha.

    Returns a dictionary with comprehensive commit metrics.

    Retention Rate is calculated as:
        stable commits / total commits * 100

    Stable Commits = commits that have NOT been reverted (is_reverted=False)
    Total Commits = all commits authored by the agent
    """
    total_commits = len(metadata_list)

    # Count stable commits - those that have NOT been reverted
    stable_commits = sum(1 for commit_meta in metadata_list
                        if not commit_meta.get('is_reverted', False))

    # Calculate retention rate
    retention_rate = (stable_commits / total_commits * 100) if total_commits > 0 else 0

    return {
        'total_commits': total_commits,
        'stable_commits': stable_commits,
        'retention_rate': round(retention_rate, 2),
    }


def calculate_monthly_metrics_by_agent():
    """
    Calculate monthly metrics for all agents for visualization.
    Loads data directly from SWE-Arena/commit_metadata dataset.

    Returns:
        dict: {
            'agents': list of agent names,
            'months': list of month labels (e.g., '2025-01'),
            'data': {
                agent_name: {
                    'retention_rates': list of retention rates by month,
                    'total_commits': list of commit counts by month,
                    'stable_commits': list of stable commit counts by month
                }
            }
        }
    """
    # Load ALL agents from HuggingFace agents repo
    agents = load_agents_from_hf()

    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {agent.get('github_identifier'): agent.get('agent_name') for agent in agents if agent.get('github_identifier')}

    # Load all commit metadata from commit_metadata dataset
    all_metadata = load_commit_metadata()

    if not all_metadata:
        return {'agents': [], 'months': [], 'data': {}}

    # Group by agent and month
    agent_month_data = defaultdict(lambda: defaultdict(list))

    for commit_meta in all_metadata:
        agent_identifier = commit_meta.get('agent_identifier')
        commit_at = commit_meta.get('commit_at')

        if not agent_identifier or not commit_at:
            continue

        # Get agent_name from identifier
        agent_name = identifier_to_name.get(agent_identifier, agent_identifier)

        try:
            dt = datetime.fromisoformat(commit_at.replace('Z', '+00:00'))
            month_key = f"{dt.year}-{dt.month:02d}"
            agent_month_data[agent_name][month_key].append(commit_meta)
        except Exception as e:
            print(f"Warning: Could not parse date '{commit_at}': {e}")
            continue

    # Get all unique months and sort them
    all_months = set()
    for agent_data in agent_month_data.values():
        all_months.update(agent_data.keys())
    months = sorted(list(all_months))

    # Calculate metrics for each agent and month
    result_data = {}
    for agent_name, month_dict in agent_month_data.items():
        retention_rates = []
        total_commits_list = []
        stable_commits_list = []

        for month in months:
            commits_in_month = month_dict.get(month, [])

            # Count stable commits (those that have NOT been reverted)
            stable_count = sum(1 for commit in commits_in_month if not commit.get('is_reverted', False))

            # Total commits created in this month
            total_count = len(commits_in_month)

            # Calculate retention rate
            retention_rate = (stable_count / total_count * 100) if total_count > 0 else None

            retention_rates.append(retention_rate)
            total_commits_list.append(total_count)
            stable_commits_list.append(stable_count)

        result_data[agent_name] = {
            'retention_rates': retention_rates,
            'total_commits': total_commits_list,
            'stable_commits': stable_commits_list
        }

    return {
        'agents': sorted(list(agent_month_data.keys())),
        'months': months,
        'data': result_data
    }


# =============================================================================
# COMMIT METADATA STORAGE & RETRIEVAL
# =============================================================================

def group_metadata_by_date(metadata_list):
    """
    Group commit metadata by exact date (year.month.day) for efficient daily storage.
    Returns dict: {(year, month, day): [metadata_list]}
    """
    grouped = defaultdict(list)

    for commit_meta in metadata_list:
        commit_at = commit_meta.get('commit_at')
        if not commit_at:
            continue

        try:
            dt = datetime.fromisoformat(commit_at.replace('Z', '+00:00'))
            key = (dt.year, dt.month, dt.day)
            grouped[key].append(commit_meta)
        except Exception as e:
            print(f"Warning: Could not parse date '{commit_at}': {e}")

    return dict(grouped)


def save_commit_metadata_to_hf(metadata_list, agent_identifier):
    """
    Save commit metadata to HuggingFace dataset, organized by [agent_identifier]/YYYY.MM.DD.jsonl.
    Each file is stored in the agent's folder and named YYYY.MM.DD.jsonl for that day's commits.
    In debug mode, saves to in-memory cache only.

    This function APPENDS new metadata and DEDUPLICATES by sha.

    Args:
        metadata_list: List of commit metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    # Skip saving to HF in debug mode - use in-memory cache instead
    if DEBUG_MODE:
        global DEBUG_COMMIT_METADATA_CACHE
        # Merge with existing cache, deduplicating by sha
        existing = {commit['sha']: commit for commit in DEBUG_COMMIT_METADATA_CACHE[agent_identifier] if commit.get('sha')}
        new = {commit['sha']: commit for commit in metadata_list if commit.get('sha')}
        existing.update(new)
        DEBUG_COMMIT_METADATA_CACHE[agent_identifier] = list(existing.values())
        print(f"🐛 DEBUG MODE: Saved to in-memory cache only ({len(metadata_list)} commits) - NOT saved to HuggingFace")
        return True

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi()

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        for (commit_year, month, day), day_metadata in grouped.items():
            # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
            filename = f"{agent_identifier}/{commit_year}.{month:02d}.{day:02d}.jsonl"
            local_filename = f"{commit_year}.{month:02d}.{day:02d}.jsonl"
            print(f"📤 Uploading {len(day_metadata)} commits to {filename}...")

            # Download existing file if it exists
            existing_metadata = []
            try:
                file_path = hf_hub_download(
                    repo_id=COMMIT_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                existing_metadata = load_jsonl(file_path)
                print(f"   Found {len(existing_metadata)} existing commits in {filename}")
            except Exception:
                print(f"   No existing file found for {filename}, creating new")

            # Merge and deduplicate by sha
            existing_by_sha = {meta['sha']: meta for meta in existing_metadata if meta.get('sha')}
            new_by_sha = {meta['sha']: meta for meta in day_metadata if meta.get('sha')}

            # Update with new data (new data overwrites old)
            existing_by_sha.update(new_by_sha)
            merged_metadata = list(existing_by_sha.values())

            # Save locally
            save_jsonl(local_filename, merged_metadata)

            try:
                # Upload to HuggingFace with folder path
                upload_with_retry(
                    api=api,
                    path_or_fileobj=local_filename,
                    path_in_repo=filename,
                    repo_id=COMMIT_METADATA_REPO,
                    repo_type="dataset",
                    token=token
                )
                print(f"   ✓ Saved {len(merged_metadata)} total commits to {filename}")
            finally:
                # Always clean up local file, even if upload fails
                if os.path.exists(local_filename):
                    os.remove(local_filename)

        return True

    except Exception as e:
        print(f"✗ Error saving commit metadata: {str(e)}")
        return False


def load_commit_metadata():
    """
    Load commit metadata from the last LEADERBOARD_TIME_FRAME_DAYS days.
    In debug mode, loads from in-memory cache if available.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Returns:
        List of dictionaries with 'agent_identifier' added to each commit metadata.
        Only includes commits within the last LEADERBOARD_TIME_FRAME_DAYS days.
    """
    # In debug mode, check in-memory cache first
    if DEBUG_MODE and DEBUG_COMMIT_METADATA_CACHE:
        all_metadata = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

        for agent_identifier, metadata_list in DEBUG_COMMIT_METADATA_CACHE.items():
            for commit_meta in metadata_list:
                # Filter by time frame
                commit_at = commit_meta.get('commit_at')
                if commit_at:
                    try:
                        dt = datetime.fromisoformat(commit_at.replace('Z', '+00:00'))
                        if dt < cutoff_date:
                            continue  # Skip commits older than time frame
                    except Exception:
                        pass  # Keep commits with unparseable dates

                commit_with_agent = commit_meta.copy()
                commit_with_agent['agent_identifier'] = agent_identifier
                all_metadata.append(commit_with_agent)

        if all_metadata:
            print(f"🐛 DEBUG MODE: Loading commit metadata from in-memory cache ({len(all_metadata)} commits within last {LEADERBOARD_TIME_FRAME_DAYS} days)")
            return all_metadata

    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate cutoff date (LEADERBOARD_TIME_FRAME_DAYS ago)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

        # List all files in the repository
        files = api.list_repo_files(repo_id=COMMIT_METADATA_REPO, repo_type="dataset")

        # Filter for files matching the pattern: [agent_identifier]/YYYY.MM.DD.jsonl
        # Only consider files within the time frame
        relevant_files = []
        for f in files:
            if f.endswith('.jsonl'):
                parts = f.split('/')
                if len(parts) == 2:  # [agent_identifier]/YYYY.MM.DD.jsonl
                    filename = parts[1]
                    try:
                        # Extract date from filename: YYYY.MM.DD.jsonl
                        date_part = filename.replace('.jsonl', '')
                        date_components = date_part.split('.')
                        if len(date_components) == 3:
                            file_year, file_month, file_day = map(int, date_components)
                            file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                            # Only include files within the time frame
                            if file_date >= cutoff_date:
                                relevant_files.append(f)
                    except Exception:
                        continue  # Skip files with invalid date format

        print(f"📥 Loading commit metadata from last {LEADERBOARD_TIME_FRAME_DAYS} days ({len(relevant_files)} daily files across all agents)...")

        all_metadata = []
        for filename in relevant_files:
            try:
                # Extract agent_identifier from path (first part)
                # Format: agent_identifier/YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    print(f"   Warning: Unexpected filename format: {filename}")
                    continue

                agent_identifier = parts[0]

                file_path = hf_hub_download(
                    repo_id=COMMIT_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                day_metadata = load_jsonl(file_path)

                # Add agent_identifier to each commit metadata and filter by commit_at
                for commit_meta in day_metadata:
                    # Double-check individual commit dates (in case file contains older data)
                    commit_at = commit_meta.get('commit_at')
                    if commit_at:
                        try:
                            dt = datetime.fromisoformat(commit_at.replace('Z', '+00:00'))
                            if dt < cutoff_date:
                                continue  # Skip commits older than time frame
                        except Exception:
                            pass  # Keep commits with unparseable dates

                    commit_meta['agent_identifier'] = agent_identifier
                    all_metadata.append(commit_meta)

                print(f"   ✓ Loaded {len(day_metadata)} commits from {filename}")
            except Exception as e:
                print(f"   Warning: Could not load {filename}: {str(e)}")

        print(f"✓ Loaded {len(all_metadata)} total commits within last {LEADERBOARD_TIME_FRAME_DAYS} days")
        return all_metadata

    except Exception as e:
        print(f"✗ Error loading commit metadata: {str(e)}")
        return []


def get_latest_commit_date_for_agent(agent_identifier):
    """
    Get the latest commit creation date for an agent from stored metadata.
    Used for incremental updates - only fetch commits newer than this date.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Args:
        agent_identifier: GitHub identifier of the agent

    Returns:
        datetime or None if no existing commits found.
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = api.list_repo_files(repo_id=COMMIT_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        if not agent_files:
            return None

        # Find latest created_at across all files
        latest_date = None
        for filename in agent_files:
            try:
                file_path = hf_hub_download(
                    repo_id=COMMIT_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                metadata = load_jsonl(file_path)

                for commit_meta in metadata:
                    commit_at = commit_meta.get("commit_at")
                    if commit_at:
                        try:
                            dt = datetime.fromisoformat(commit_at.replace("Z", "+00:00"))
                            if latest_date is None or dt > latest_date:
                                latest_date = dt
                        except Exception:
                            continue
            except Exception:
                continue

        return latest_date

    except Exception:
        return None


def get_daily_files_last_n_months(agent_identifier, n_months=6):
    """
    Get list of daily file paths for an agent from the last N months.

    Args:
        agent_identifier: GitHub identifier of the agent
        n_months: Number of months to look back (default: 6)

    Returns:
        List of file paths in format: [agent_identifier]/YYYY.MM.DD.jsonl
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate date range
        today = datetime.now(timezone.utc)
        n_months_ago = today - timedelta(days=30 * n_months)

        # List all files in the repository
        files = api.list_repo_files(repo_id=COMMIT_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        # Filter by date range (extract date from filename)
        recent_files = []
        for filename in agent_files:
            try:
                # Extract date from filename: YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    continue

                date_part = parts[1].replace('.jsonl', '')  # Get YYYY.MM.DD
                date_components = date_part.split('.')
                if len(date_components) != 3:
                    continue

                file_year, file_month, file_day = map(int, date_components)
                file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                # Include if within last n_months
                if n_months_ago <= file_date <= today:
                    recent_files.append(filename)
            except Exception:
                continue

        return recent_files

    except Exception as e:
        print(f"Error getting daily files: {str(e)}")
        return []


def get_already_mined_dates(agent_identifier, n_months=6):
    """
    Get set of dates that have already been mined for an agent.

    Args:
        agent_identifier: GitHub identifier of the agent
        n_months: Number of months to look back (default: 6)

    Returns:
        Set of date objects (datetime.date) that already have data files
    """
    try:
        api = HfApi()

        # Calculate date range
        today = datetime.now(timezone.utc)
        n_months_ago = today - timedelta(days=30 * n_months)

        # List all files in the repository
        files = api.list_repo_files(repo_id=COMMIT_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        mined_dates = set()
        for filename in agent_files:
            try:
                # Extract date from filename: [agent_identifier]/YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    continue

                date_part = parts[1].replace('.jsonl', '')  # Get YYYY.MM.DD
                date_components = date_part.split('.')
                if len(date_components) != 3:
                    continue

                file_year, file_month, file_day = map(int, date_components)
                file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc).date()

                # Only include dates within the last n_months
                if n_months_ago.date() <= file_date <= today.date():
                    mined_dates.add(file_date)
            except Exception as e:
                print(f"   Warning: Could not parse date from filename {filename}: {e}")
                continue

        return mined_dates

    except Exception as e:
        print(f"   Warning: Could not get already-mined dates for {agent_identifier}: {str(e)}")
        return set()


def fetch_commit_current_status(commit_url, token):
    """
    Fetch the current revert status of a single commit from GitHub API.

    Args:
        token: GitHub API token
        token: GitHub API token

    Returns:
        Dictionary with updated is_reverted and revert_at, or None if failed
    """
    try:
        # Convert HTML URL to API URL
        # https://github.com/owner/repo/commits/123 -> https://api.github.com/repos/owner/repo/commits/123
        parts = commit_url.replace('https://github.com/', '').split('/')
        if len(parts) < 4:
            return None

        owner, repo, commit_word, commit_number = parts[0], parts[1], parts[2], parts[3]
        api_url = f'https://api.github.com/repos/{owner}/{repo}/commits/{commit_number}'

        headers = {'Authorization': f'token {token}'} if token else {}
        response = request_with_backoff('GET', api_url, headers=headers, max_retries=3)

        if response is None or response.status_code != 200:
            return None

        commit_data = response.json()
        state = commit_data.get('state')
        state_reason = commit_data.get('state_reason')
        closed_at = commit_data.get('closed_at')

        return {
            'state': state,
            'state_reason': state_reason,
            'closed_at': closed_at
        }

    except Exception as e:
        print(f"   Error fetching commit status for {commit_url}: {str(e)}")
        return None


def refresh_commit_status_for_agent(agent_identifier, token):
    """
    Refresh status for all open commits from the last 6 months for an agent.
    Only updates commits that are still open (state="open" or no state_reason).

    This implements the smart update strategy:
    - Skip commits that are already closed/resolved
    - Fetch current status for open commits
    - Update and save back to daily files

    Args:
        agent_identifier: GitHub identifier of the agent
        token: GitHub API token

    Returns:
        Tuple: (total_checked, updated_count)
    """
    print(f"\n🔄 Refreshing open commits for {agent_identifier} (last 6 months)...")

    try:
        # Get daily files from last 6 months
        recent_files = get_daily_files_last_n_months(agent_identifier, n_months=6)

        if not recent_files:
            print(f"   No recent files found for {agent_identifier}")
            return (0, 0)

        print(f"   Found {len(recent_files)} daily files to check")

        total_checked = 0
        updated_count = 0

        # Process each file
        for filename in recent_files:
            try:
                # Download file
                file_path = hf_hub_download(
                    repo_id=COMMIT_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=get_hf_token()
                )
                commits = load_jsonl(file_path)

                if not commits:
                    continue

                updated_commits = []
                file_had_updates = False

                # Check each commit
                for commit in commits:
                    # Skip if already closed (has a state_reason)
                    if commit.get("is_reverted"):
                        updated_commits.append(commit)
                        continue

                    # Commit may have been reverted, check status
                    commit_url = commit.get("html_url")

                    if not commit_url:
                        updated_commits.append(commit)
                        continue

                    current_status = fetch_commit_current_status(commit_url, token)

                    if current_status:
                        # Check if status changed (now closed)
                        if current_status['state'] == 'closed':
                            print(f"   ✓ Commit status changed: {commit_url}")
                            commit['state'] = current_status['state']
                            commit['state_reason'] = current_status['state_reason']
                            commit['closed_at'] = current_status['closed_at']
                            updated_count += 1
                            file_had_updates = True

                    updated_commits.append(commit)
                    time.sleep(0.1)  # Rate limiting courtesy delay

                # Save file if there were updates
                if file_had_updates:
                    # Extract filename components for local save
                    parts = filename.split('/')
                    local_filename = parts[-1]  # Just YYYY.MM.DD.jsonl

                    # Save locally
                    save_jsonl(local_filename, updated_commits)

                    try:
                        # Upload back to HuggingFace
                        api = HfApi()
                        upload_with_retry(
                            api=api,
                            path_or_fileobj=local_filename,
                            path_in_repo=filename,
                            repo_id=COMMIT_METADATA_REPO,
                            repo_type="dataset",
                            token=get_hf_token()
                        )
                        print(f"   💾 Updated {filename}")
                    finally:
                        # Always clean up local file, even if upload fails
                        if os.path.exists(local_filename):
                            os.remove(local_filename)

            except Exception as e:
                print(f"   Warning: Could not process {filename}: {str(e)}")
                continue

        print(f"   ✅ Refresh complete: {total_checked} open commits checked, {updated_count} updated")
        return (total_checked, updated_count)

    except Exception as e:
        print(f"   ✗ Error refreshing commits for {agent_identifier}: {str(e)}")
        return (0, 0)


# =============================================================================
# HUGGINGFACE DATASET OPERATIONS
# =============================================================================

def load_agents_from_hf():
    """Load all agent metadata JSON files from HuggingFace dataset."""
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = api.list_repo_files(repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        print(f"Found {len(json_files)} agent files in {AGENTS_REPO}")

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)
                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return None




def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.

    Args:
        api: HfApi instance
        path_or_fileobj: Local file path to upload
        path_in_repo: Target path in the repository
        repo_id: Repository ID
        repo_type: Type of repository (e.g., "dataset")
        token: HuggingFace token
        max_retries: Maximum number of retry attempts

    Returns:
        True if upload succeeded, raises exception if all retries failed
    """
    delay = 2.0  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token
            )
            if attempt > 0:
                print(f"   ✓ Upload succeeded on attempt {attempt + 1}/{max_retries}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay + random.uniform(0, 1.0)
                print(f"   ⚠️ Upload failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"   ⏳ Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                delay = min(delay * 2, 60.0)  # Exponential backoff, max 60s
            else:
                print(f"   ✗ Upload failed after {max_retries} attempts: {str(e)}")
                raise


def save_agent_to_hf(data):
    """Save a new agent to HuggingFace dataset as {identifier}.json in root."""
    try:
        api = HfApi()
        token = get_hf_token()

        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        identifier = data['github_identifier']
        filename = f"{identifier}.json"

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        try:
            # Upload to HuggingFace (root directory)
            upload_with_retry(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=AGENTS_REPO,
                repo_type="dataset",
                token=token
            )
            print(f"✓ Saved agent to HuggingFace: {filename}")
            return True
        finally:
            # Always clean up local file, even if upload fails
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"✗ Error saving agent: {str(e)}")
        return False




# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def update_all_agents_incremental():
    """
    Memory-efficient incremental update of commit statistics for all agents.

    Strategy:
    1. For each agent, load existing data from SWE-Arena/commit_metadata
    2. Identify already-mined dates (based on filename: YYYY.MM.DD.jsonl)
    3. Only fetch commits from dates that haven't been mined yet (within last 6 months)
    4. If no data exists at all, mine everything from scratch
    5. Store minimal metadata (not full commit objects) to avoid storage limits
    6. Construct leaderboard from ALL stored metadata (last 6 months)

    Returns dictionary of all agent data with current stats.
    """
    token = get_github_token()

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return {}

    cache_dict = {}

    # Update each agent
    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        if not identifier:
            print(f"Warning: Skipping agent without identifier: {agent}")
            continue

        try:
            print(f"\n{'='*80}")
            print(f"Processing: {agent_name} ({identifier})")
            print(f"{'='*80}")

            # Get already-mined dates for this agent (last 6 months)
            already_mined_dates = get_already_mined_dates(identifier, n_months=6)

            if already_mined_dates:
                print(f"📅 Found {len(already_mined_dates)} already-mined dates")
                print(f"   Skipping these dates and fetching only new data...")
                # Fetch only commits from dates not yet mined
                new_metadata = fetch_all_commits_metadata(
                    identifier,
                    agent_name,
                    token,
                    start_from_date=None,  # Use full 6-month range
                    exclude_dates=already_mined_dates  # But exclude already-mined dates
                )
            else:
                print(f"📅 No existing data found. Mining everything from scratch...")
                # Mine everything from scratch (full 6-month range)
                new_metadata = fetch_all_commits_metadata(
                    identifier,
                    agent_name,
                    token,
                    start_from_date=None
                )

            if new_metadata:
                # Save new metadata to HuggingFace (organized by agent_identifier/YYYY.MM.DD.jsonl)
                print(f"💾 Saving {len(new_metadata)} new commit records...")
                save_commit_metadata_to_hf(new_metadata, identifier)
            else:
                print(f"   No new commits to save")

            # Load ALL metadata to calculate stats (aggregates entire last 6 months)
            print(f"📊 Calculating statistics from ALL stored metadata (last 6 months)...")
            all_metadata = load_commit_metadata()

            # Filter for this specific agent
            agent_metadata = [commit for commit in all_metadata if commit.get("agent_identifier") == identifier]

            # Calculate stats from metadata
            stats = calculate_commit_stats_from_metadata(agent_metadata)

            # Merge metadata with stats
            cache_dict[identifier] = {
                'agent_name': agent_name,
                'website': agent.get('website', 'N/A'),
                'github_identifier': identifier,
                **stats
            }

            print(f"✓ Updated {identifier}: {stats['total_commits']} commits, {stats['retention_rate']}% retention")

        except Exception as e:
            print(f"✗ Error updating {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return cache_dict


def construct_leaderboard_from_metadata():
    """
    Construct leaderboard from stored commit metadata instead of fetching all commits.
    Much more memory-efficient and faster.

    Returns dictionary of agent stats.
    """
    print("📊 Constructing leaderboard from commit metadata...")

    # Load agents
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found")
        return {}

    # Load all commit metadata
    all_metadata = load_commit_metadata()

    cache_dict = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        # Filter metadata for this agent
        agent_metadata = [commit for commit in all_metadata if commit.get("agent_identifier") == identifier]

        # Calculate stats
        stats = calculate_commit_stats_from_metadata(agent_metadata)

        cache_dict[identifier] = {
            'agent_name': agent_name,
            'website': agent.get('website', 'N/A'),
            'github_identifier': identifier,
            **stats
        }

    return cache_dict


def initialize_data():
    """
    Initialize data on application startup.
    Constructs leaderboard from commit metadata.

    In DEBUG MODE:
    - If no data available, automatically mine up to 10 commits per query per agent
    - Does NOT save to HuggingFace datasets
    """
    print("🚀 Initializing leaderboard data...")

    # Try constructing from commit metadata (fast, memory-efficient)
    print(f"📂 Checking {COMMIT_METADATA_REPO} for existing data...")
    try:
        cache_dict = construct_leaderboard_from_metadata()
        # Check if there's actually meaningful data (at least one agent with commits)
        has_data = any(entry.get('total_commits', 0) > 0 for entry in cache_dict.values())
        if cache_dict and has_data:
            print(f"✓ Found existing commit metadata. Leaderboard constructed from {COMMIT_METADATA_REPO}")
            return
        else:
            print(f"   No meaningful data found in {COMMIT_METADATA_REPO}")
    except Exception as e:
        print(f"   Could not construct from metadata: {e}")

    # If in debug mode and no data available, mine immediately
    if DEBUG_MODE:
        print("\n🐛 DEBUG MODE: No data available, mining immediately (up to 10 commits per query per agent)...")
        agents = load_agents_from_hf()
        if agents:
            print(f"✓ Loaded {len(agents)} agents from HuggingFace")
            print("⛏️ Mining GitHub data in debug mode (limited to 10 commits per query)...")
            cache_dict = update_all_agents_incremental()
            print("✓ Debug mining complete (data NOT saved to HuggingFace)")
            return
        else:
            print("⚠️ No agents found. Waiting for first submission...")
            return

    # Production mode: Fallback to full incremental mining from GitHub
    agents = load_agents_from_hf()
    if agents:
        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        print("⛏️ Mining GitHub data (this may take a while)...")
        cache_dict = update_all_agents_incremental()
        return

    # No data available
    print("⚠️ No data sources available. Waiting for first submission...")


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot():
    """
    Create a Plotly figure with dual y-axes showing:
    - Left y-axis: Retention Rate (%) as line curves
    - Right y-axis: Total Commits created as bar charts

    Each agent gets a unique color for both their line and bars.
    """
    metrics = calculate_monthly_metrics_by_agent()

    if not metrics['agents'] or not metrics['months']:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=None,
            xaxis_title=None,
            height=500
        )
        return fig

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Define colors for agents (using a color palette)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    # Add traces for each agent
    for idx, agent_name in enumerate(agents):
        color = colors[idx % len(colors)]
        agent_data = data[agent_name]

        # Add line trace for resolved rate (left y-axis)
        retention_rates = agent_data['retention_rates']
        # Filter out None values for plotting
        x_resolved = [month for month, rate in zip(months, retention_rates) if rate is not None]
        y_resolved = [rate for rate in retention_rates if rate is not None]

        if x_resolved and y_resolved:  # Only add trace if there's data
            fig.add_trace(
                go.Scatter(
                    x=x_resolved,
                    y=y_resolved,
                    name=agent_name,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    legendgroup=agent_name,
                    showlegend=True,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Retention Rate: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=False
            )

        # Add bar trace for total commits (right y-axis)
        # Only show bars for months where agent has commits
        x_bars = []
        y_bars = []
        for month, count in zip(months, agent_data['total_commits']):
            if count > 0:  # Only include months with commits
                x_bars.append(month)
                y_bars.append(count)

        if x_bars and y_bars:  # Only add trace if there's data
            fig.add_trace(
                go.Bar(
                    x=x_bars,
                    y=y_bars,
                    name=f"{agent_name} (Commits)",
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=agent_name,
                    showlegend=False,  # Don't show in legend (already shown for line)
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Total Commits: %{y}<br>' +
                                 '<extra></extra>',
                    offsetgroup=agent_name  # Group bars by agent for proper spacing
                ),
                secondary_y=True
            )

    # Update axes labels
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text="<b>Retention Rate (%)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Total Commits</b>", secondary_y=True)

    # Update layout
    fig.update_layout(
        title=None,
        hovermode='x unified',
        barmode='group',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=100, b=50)
    )

    return fig


def get_leaderboard_dataframe():
    """
    Construct leaderboard from commit metadata and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by retention rate.
    """
    # Construct leaderboard from metadata
    cache_dict = construct_leaderboard_from_metadata()

    if not cache_dict:
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    for data in cache_dict.values():
        # Filter out agents with zero total commits
        if data.get('total_commits', 0) == 0:
            continue
        # Only include display-relevant fields
        rows.append([
            data.get('agent_name', 'Unknown'),
            data.get('website', 'N/A'),
            data.get('total_commits', 0),
            data.get('stable_commits', 0),
            data.get('retention_rate', 0.0),
        ])

    # Create DataFrame
    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)

    # Ensure numeric types
    numeric_cols = ["Total Commits", "Stable Commits", "Retention Rate (%)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by Retention Rate (%) descending
    if "Retention Rate (%)" in df.columns and not df.empty:
        df = df.sort_values(by="Retention Rate (%)", ascending=False).reset_index(drop=True)

    return df


def submit_agent(identifier, agent_name, organization, description, website):
    """
    Submit a new agent to the leaderboard.
    Validates input, saves submission, and fetches PR metadata (memory-efficient).
    """
    # Validate required fields
    if not identifier or not identifier.strip():
        return "❌ GitHub identifier is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not agent_name or not agent_name.strip():
        return "❌ Agent name is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not organization or not organization.strip():
        return "❌ Organization name is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not website or not website.strip():
        return "❌ Website URL is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    organization = organization.strip()
    description = description.strip()
    website = website.strip()

    # Validate GitHub identifier
    is_valid, message = validate_github_username(identifier)
    if not is_valid:
        return f"❌ {message}", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Check for duplicates by loading agents from HuggingFace
    agents = load_agents_from_hf()
    if agents:
        existing_names = {agent['github_identifier'] for agent in agents}
        if identifier in existing_names:
            return f"⚠️ Agent with identifier '{identifier}' already exists", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Create submission
    submission = {
        'agent_name': agent_name,
        'organization': organization,
        'github_identifier': identifier,
        'description': description,
        'website': website,
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "❌ Failed to save submission", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Fetch commit metadata immediately (memory-efficient)
    token = get_github_token()
    try:
        print(f"Fetching commit metadata for {agent_name}...")

        # Fetch lightweight metadata
        metadata_list = fetch_all_commits_metadata(identifier, agent_name, token)

        if metadata_list:
            # Save metadata to HuggingFace
            save_commit_metadata_to_hf(metadata_list, identifier)

        # Calculate stats from metadata
        stats = calculate_commit_stats_from_metadata(metadata_list)

        return f"✅ Successfully submitted {agent_name}! Stats: {stats['total_commits']} commits, {stats['retention_rate']}% retention", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    except Exception as e:
        error_msg = f"⚠️ Submitted {agent_name}, but failed to fetch commit data: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, get_leaderboard_dataframe(), create_monthly_metrics_plot()


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

def daily_update_task():
    """
    Daily scheduled task (runs at 12:00 AM UTC) for smart commit updates.

    Strategy:
    1. For each agent, refresh open commits from last 6 months
    2. Skip commits that are already closed/resolved (no API calls)
    3. Only fetch status for open commits to check if they've been closed/resolved
    4. Update leaderboard with refreshed data

    This is much more efficient than fetching all commits every time.
    """
    print(f"\n{'='*80}")
    print(f"🕛 Daily update started at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*80}")

    try:
        token = get_github_token()

        # Load all agents
        agents = load_agents_from_hf()
        if not agents:
            print("No agents found")
            return

        print(f"📋 Processing {len(agents)} agents...")

        total_checked = 0
        total_updated = 0

        # Refresh open commits for each agent (last 6 months)
        for agent in agents:
            identifier = agent.get('github_identifier')
            agent_name = agent.get('agent_name', 'Unknown')

            if not identifier:
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {agent_name} ({identifier})")
            print(f"{'='*60}")

            # Refresh open commits from last 6 months
            checked, updated = refresh_commit_status_for_agent(identifier, token)
            total_checked += checked
            total_updated += updated

        print(f"\n{'='*80}")
        print(f"📊 Refresh Summary:")
        print(f"   Total open commits checked: {total_checked}")
        print(f"   Commits updated (newly reverted): {total_updated}")
        print(f"{'='*80}")

        print(f"\n✅ Daily update completed at {datetime.now(timezone.utc).isoformat()}")

    except Exception as e:
        print(f"✗ Daily update failed: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

# Initialize data before creating UI
if DEBUG_MODE:
    print("\n" + "="*80)
    print("🐛 DEBUG MODE ENABLED 🐛")
    print("="*80)
    print("Commit retrieval is limited to 10 commits per query pattern per agent")

    # Show how debug mode was enabled
    if args.debug:
        print("Enabled via: command-line flag '--debug'")
        print("To disable: run without '--debug' flag")
    else:
        print("Enabled via: DEBUG_MODE environment variable")
        print("To disable: run with '--no-debug' flag or unset DEBUG_MODE")

    print("="*80 + "\n")
else:
    print("\n🚀 Starting in PRODUCTION MODE - full commit retrieval enabled")
    if args.no_debug:
        print("   (Explicitly set via '--no-debug' flag)")
    print()

initialize_data()

# Start APScheduler for daily updates at 12:00 AM UTC
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    daily_update_task,
    trigger=CronTrigger(hour=0, minute=0),  # 12:00 AM UTC daily
    id='daily_commit_refresh',
    name='Daily Commit Status Refresh',
    replace_existing=True
)
scheduler.start()
print("✓ Scheduler started: Daily updates at 12:00 AM UTC")

# Create Gradio interface
with gr.Blocks(title="SWE Agent Commit Leaderboard", theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🏆 SWE Agent Commit Leaderboard")
    gr.Markdown("Track and compare GitHub commit retention statistics for SWE agents (last 6 months)")
    
    with gr.Tabs():
        
        # Leaderboard Tab
        with gr.Tab("📊 Leaderboard"):
            gr.Markdown("*All statistics are based on commits from the last 6 months*")
            leaderboard_table = Leaderboard(
                value=get_leaderboard_dataframe(),
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=["Retention Rate (%)"]
            )

            gr.Markdown("### Monthly Metrics")
            gr.Markdown("Track retention rates and commit activity over time")

            monthly_plot = gr.Plot(
                value=create_monthly_metrics_plot(),
                label="Monthly Commit Metrics"
            )

        # Submit Agent Tab
        with gr.Tab("➕ Submit Agent"):
            
            gr.Markdown("### Submit Your Agent")
            gr.Markdown("Fill in the details below to add your agent to the leaderboard. Make sure you're logged in to HuggingFace CLI on your machine.")
            
            with gr.Row():
                with gr.Column():
                    github_input = gr.Textbox(
                        label="GitHub Identifier*",
                        placeholder="Your agent username (e.g., my-agent-bot)"
                    )
                    name_input = gr.Textbox(
                        label="Agent Name*",
                        placeholder="Your agent's display name"
                    )
                
                with gr.Column():
                    organization_input = gr.Textbox(
                        label="Organization*",
                        placeholder="Your organization or team name"
                    )
                    description_input = gr.Textbox(
                        label="Description",
                        placeholder="Brief description of your agent",
                        lines=3
                    )
                    website_input = gr.Textbox(
                        label="Website",
                        placeholder="https://your-agent-website.com"
                    )
            
            submit_button = gr.Button(
                "Submit Agent",
                variant="primary"
            )
            submission_status = gr.Textbox(
                label="Submission Status",
                interactive=False
            )
            
            # Event handler
            submit_button.click(
                fn=submit_agent,
                inputs=[github_input, name_input, organization_input, description_input, website_input],
                outputs=[submission_status, leaderboard_table, monthly_plot]
            )


# Launch application
if __name__ == "__main__":
    app.launch()