"""
Minimalist Commit Metadata Mining Script
Mines commit metadata from GitHub and saves to HuggingFace dataset.
"""

import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/swe_agents"
COMMIT_METADATA_REPO = "SWE-Arena/commit_metadata"
LEADERBOARD_TIME_FRAME_DAYS = 180  # 6 months

# =============================================================================
# UTILITY FUNCTIONS
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
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def get_github_token():
    """Get GitHub token from environment variables."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found. API rate limits: 60/hour (authenticated: 5000/hour)")
    return token


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# GITHUB API FUNCTIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.
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


def batch_check_reverts_for_commits(commits_by_sha, headers, start_date=None, end_date=None):
    """
    Check revert status for commits using SCOPED batch approach (highly efficient).

    Strategy:
    - Groups commits into batches of ~50 SHAs
    - For each batch, searches for reverts mentioning those specific SHAs
    - This targets only relevant reverts instead of fetching ALL reverts in time range
    - Avoids the "8 million reverts" problem while staying efficient

    API calls: ~(num_commits / 50) * 2, typically 10-40 calls for 1000 commits
    This is 95%+ more efficient than per-commit checking, and scales with YOUR commits
    instead of scaling with ALL reverts in the ecosystem.

    Args:
        commits_by_sha: Dict of commits keyed by SHA
        headers: HTTP headers for GitHub API
        start_date: Optional start date to filter revert commits (datetime object)
        end_date: Optional end date to filter revert commits (datetime object)

    Returns:
        Dict mapping SHA to {'is_reverted': bool, 'revert_at': date_string}
    """
    if not commits_by_sha:
        return {}

    print(f"   🔍 Scoped batch checking revert status for {len(commits_by_sha)} commits...")

    # Group commits into batches for scoped searching
    BATCH_SIZE = 50  # Search for up to 50 SHAs at once
    commit_shas = list(commits_by_sha.keys())
    revert_map = {}  # {reverted_sha: revert_date}

    import re

    # Process commits in batches
    num_batches = (len(commit_shas) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(commit_shas))
        batch_shas = commit_shas[start_idx:end_idx]

        # Build query for this batch of SHAs
        # Search for: "This reverts commit <sha1>" OR "This reverts commit <sha2>" ...
        # GitHub uses abbreviated (7-char) SHA in revert commit messages by default
        sha_queries = []
        for sha in batch_shas:
            sha_abbr = sha[:7]  # Use 7-char abbreviated SHA
            sha_queries.append(f'"This reverts commit {sha_abbr}"')

        # Combine with OR (GitHub search supports this)
        revert_query = ' OR '.join(sha_queries)

        # Add date range if provided
        if start_date and end_date:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            revert_query += f' committer-date:{start_str}..{end_str}'

        # Fetch revert commits for this batch
        page = 1
        per_page = 100

        while True:
            url = 'https://api.github.com/search/commits'
            params = {
                'q': revert_query,
                'per_page': per_page,
                'page': page,
                'sort': 'committer-date',
                'order': 'asc'
            }

            headers_with_accept = headers.copy() if headers else {}
            headers_with_accept['Accept'] = 'application/vnd.github.cloak-preview+json'

            response = request_with_backoff('GET', url, headers=headers_with_accept, params=params, max_retries=3)

            if not response or response.status_code != 200:
                break

            data = response.json()
            items = data.get('items', [])

            if not items:
                break

            # Parse revert commits
            for revert_commit in items:
                message = revert_commit.get('commit', {}).get('message', '')
                revert_date = revert_commit.get('commit', {}).get('author', {}).get('date')

                # Extract SHA from message
                matches = re.findall(r'\b([0-9a-f]{7,40})\b', message.lower())

                for sha in matches:
                    # Store first (earliest) revert date for each SHA
                    if sha not in revert_map:
                        revert_map[sha] = revert_date

            # GitHub API limit: max 1000 results, max 10 pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.3)  # Courtesy delay

        # Delay between batches to avoid rate limiting
        if batch_idx < num_batches - 1:
            time.sleep(0.5)

    print(f"   📌 Identified {len(revert_map)} reverted commits")

    # Build results for our commits
    results = {}
    for sha in commits_by_sha:
        sha_lower = sha.lower()
        found_revert = False
        revert_date = None

        # Check if any extracted SHA matches this commit
        # Match by prefix to handle abbreviated SHAs safely
        for extracted_sha, date in revert_map.items():
            # Match if:
            # 1. Full SHA matches (sha_lower == extracted_sha), OR
            # 2. Abbreviated SHA matches AND it's a prefix of our full SHA
            if extracted_sha == sha_lower or (len(extracted_sha) >= 7 and sha_lower.startswith(extracted_sha)):
                found_revert = True
                revert_date = date
                break

        results[sha] = {
            'is_reverted': found_revert,
            'revert_at': revert_date
        }

    reverted_count = sum(1 for r in results.values() if r['is_reverted'])
    print(f"   ✅ Scoped batch check complete: {reverted_count}/{len(results)} commits were reverted")

    return results


def fetch_commits_with_time_partition(base_query, start_date, end_date, headers, commits_by_sha, depth=0, dates_to_skip=None, max_recursion_depth=8):
    """
    Fetch commits within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.
    Uses intelligent splitting: days at shallow depth, then hours/minutes/seconds at deeper depths.

    Args:
        base_query: GitHub search query pattern
        start_date: Start datetime for the range
        end_date: End datetime for the range
        headers: HTTP headers for GitHub API
        commits_by_sha: Dict to accumulate commits keyed by SHA
        depth: Current recursion depth
        dates_to_skip: Set of (year, month, day) tuples to skip during mining
        max_recursion_depth: Maximum recursion depth before giving up (prevents infinite recursion)

    Returns the number of commits found in this time partition.
    """
    if dates_to_skip is None:
        dates_to_skip = set()

    # Safety check: prevent infinite recursion
    if depth > max_recursion_depth:
        indent = "  " + "  " * depth
        print(f"{indent}⚠️ Reached maximum recursion depth ({max_recursion_depth}). Stopping recursive partitioning.")
        return 0

    # Check if this entire date range should be skipped (only for day-level comparisons)
    if start_date.date() == end_date.date():
        date_key = (start_date.year, start_date.month, start_date.day)
        if date_key in dates_to_skip:
            indent = "  " + "  " * depth
            print(f"{indent}⏭️  Skipping {start_date.strftime('%Y-%m-%d')} (already exists)")
            return 0

    # Format dates for GitHub search
    # Use full ISO format with time for hour/minute/second precision at deeper recursion levels
    if depth == 0:
        # Day-level precision for initial query
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    else:
        # Higher precision (ISO format) for deeper recursion levels
        start_str = start_date.isoformat().split('.')[0]  # Remove microseconds
        end_str = end_date.isoformat().split('.')[0]

    # Add date range to query (use committer-date for commits)
    query = f'{base_query} committer-date:{start_str}..{end_str}'

    indent = "  " + "  " * depth
    time_display = start_str if depth == 0 else f"{start_str} to {end_str}"
    print(f"{indent}Searching range {time_display}...")

    page = 1
    per_page = 100
    total_in_partition = 0

    while True:
        url = 'https://api.github.com/search/commits'
        params = {
            'q': query,
            'per_page': per_page,
            'page': page,
            'sort': 'committer-date',
            'order': 'asc'
        }

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
            for commit in items:
                commit_sha = commit.get('sha')
                if commit_sha and commit_sha not in commits_by_sha:
                    commits_by_sha[commit_sha] = commit
                    total_in_partition += 1

            # Check if we hit the 1000-result limit
            if total_count > 1000 and page == 10:
                print(f"{indent}  ⚠️ Hit 1000-result limit ({total_count} total). Splitting time range...")

                time_diff = end_date - start_date

                # Intelligent splitting strategy based on recursion depth and time granularity
                if depth < 2:
                    # At shallow depth: split by days (largest granularity)
                    split_strategy = "days"
                    num_splits = 4
                elif depth < 4:
                    # At medium depth: split by hours
                    split_strategy = "hours"
                    num_splits = 4
                else:
                    # At deep depth: split by minutes/seconds (smallest granularity)
                    split_strategy = "minutes"
                    num_splits = 4

                print(f"{indent}    Using {split_strategy}-based splitting (depth={depth})")

                # Generate split timestamps based on strategy
                split_dates = []
                if split_strategy == "days":
                    quarter_diff = time_diff / num_splits
                    for i in range(num_splits + 1):
                        split_dates.append(start_date + quarter_diff * i)
                elif split_strategy == "hours":
                    total_hours = time_diff.total_seconds() / 3600
                    hour_diff = total_hours / num_splits
                    for i in range(num_splits + 1):
                        split_dates.append(start_date + timedelta(hours=hour_diff * i))
                else:  # minutes
                    total_minutes = time_diff.total_seconds() / 60
                    minute_diff = total_minutes / num_splits
                    for i in range(num_splits + 1):
                        split_dates.append(start_date + timedelta(minutes=minute_diff * i))

                total_from_splits = 0
                for i in range(len(split_dates) - 1):
                    split_start = split_dates[i]
                    split_end = split_dates[i + 1]

                    # Add small offset to avoid overlap in range boundaries
                    if i > 0:
                        split_start = split_start + timedelta(seconds=1)

                    count = fetch_commits_with_time_partition(
                        base_query, split_start, split_end, headers, commits_by_sha, 
                        depth + 1, dates_to_skip, max_recursion_depth
                    )
                    total_from_splits += count

                return total_from_splits

            # Normal pagination: check if there are more pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"{indent}  Error fetching range {start_str} to {end_str}: {str(e)}")
            return total_in_partition

    if total_in_partition > 0:
        print(f"{indent}  ✓ Found {total_in_partition} commits in range {time_display}")

    return total_in_partition


def extract_commit_metadata(commit, revert_status=None):
    """
    Extract minimal commit metadata for efficient storage.
    Only keeps essential fields: html_url, commit_at, revert_at, is_reverted, sha.
    """
    commit_at = commit.get('commit', {}).get('author', {}).get('date')

    if revert_status is None:
        revert_status = {'is_reverted': False, 'revert_at': None}

    return {
        'html_url': commit.get('html_url'),
        'commit_at': commit_at,
        'revert_at': revert_status.get('revert_at'),
        'is_reverted': revert_status.get('is_reverted', False),
        'sha': commit.get('sha')
    }


def fetch_all_commits_metadata(identifier, agent_name, token=None, dates_to_skip=None):
    """
    Fetch commits associated with a GitHub user or bot for the past LEADERBOARD_TIME_FRAME_DAYS.
    Returns lightweight metadata instead of full commit objects.

    HYBRID ITERATIVE-RECURSIVE APPROACH:
    - OUTER LOOP (Iterative): Mines each day separately (180 iterations for 6 months)
    - INNER LOOP (Recursive): When hitting 1000-result limit, splits by hour/minute/second
    - MAX RECURSION DEPTH: Limited to 8 to prevent stack overflow

    This prevents recursion depth issues while efficiently mining all commits within the timeframe.

    Revert status is checked in BATCH after all commits are fetched using
    batch_check_reverts_for_commits(). This is 98% more efficient than checking
    each commit individually during retrieval.

    Args:
        identifier: GitHub username or bot identifier
        agent_name: Human-readable name of the agent for metadata purposes
        token: GitHub API token for authentication
        dates_to_skip: Set of (year, month, day) tuples to skip during mining

    Returns:
        List of dictionaries containing minimal commit metadata with revert status
    """
    if dates_to_skip is None:
        dates_to_skip = set()

    headers = {'Authorization': f'token {token}'} if token else {}

    # Define query pattern
    query_pattern = f'is:commit author:{identifier}'

    commits_by_sha = {}

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS
    current_time = datetime.now(timezone.utc)
    start_date = current_time - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)
    end_date = current_time

    print(f"\n🔍 Searching with query: {query_pattern}")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    if dates_to_skip:
        print(f"   Skipping {len(dates_to_skip)} existing date(s)")

    # OUTER ITERATIVE LOOP: Mine each day separately
    # This avoids hitting the recursion depth limit that plagued range-based queries
    print(f"\n📅 Mining strategy: Iterating through {LEADERBOARD_TIME_FRAME_DAYS} days (outer loop)")
    print(f"    Each day uses recursive partitioning if hitting 1000-result limit (inner loop)")

    total_days_mined = 0
    total_commits_found = 0
    pattern_start_time = time.time()

    current_day = start_date
    while current_day <= end_date:
        date_key = (current_day.year, current_day.month, current_day.day)

        # Check if this day should be skipped
        if date_key in dates_to_skip:
            print(f"   ⏭️  Skipping {current_day.strftime('%Y-%m-%d')} (already exists)")
            current_day += timedelta(days=1)
            continue

        day_start = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = current_day.replace(hour=23, minute=59, second=59, microsecond=999999)

        day_date_str = current_day.strftime('%Y-%m-%d')
        print(f"\n   📆 Mining day {day_date_str}...")
        day_start_time = time.time()
        initial_count = len(commits_by_sha)

        # INNER RECURSIVE LOOP: For this specific day, use time-based partitioning
        # This recursively splits by hour/minute/second if needed, but stays within max depth
        commits_found = fetch_commits_with_time_partition(
            query_pattern,
            day_start,
            day_end,
            headers,
            commits_by_sha,
            depth=0,
            dates_to_skip=dates_to_skip,
            max_recursion_depth=8  # Safely limit recursion
        )

        day_duration = time.time() - day_start_time
        new_commits_today = len(commits_by_sha) - initial_count
        total_commits_found += new_commits_today
        total_days_mined += 1

        if new_commits_today > 0:
            print(f"      Found {new_commits_today} new commits today ({day_duration:.1f}s)")

        current_day += timedelta(days=1)

        # Courtesy delay between days to avoid rate limiting
        time.sleep(0.3)

    pattern_duration = time.time() - pattern_start_time

    print(f"\n   ✓ Iterative mining complete:")
    print(f"     - Days processed: {total_days_mined}/{LEADERBOARD_TIME_FRAME_DAYS}")
    print(f"     - Total unique commits: {len(commits_by_sha)}")
    print(f"     ⏱️ Time taken: {pattern_duration:.1f} seconds")

    all_commits = list(commits_by_sha.values())

    # BATCH check revert status for all commits at once (much more efficient)
    if all_commits:
        print(f"\n🔄 Checking revert status for all commits in batch...")
        revert_results = batch_check_reverts_for_commits(
            commits_by_sha,
            headers,
            start_date=start_date,
            end_date=end_date
        )
        # Attach revert status to each commit
        for sha, commit in commits_by_sha.items():
            commit['_revert_status'] = revert_results.get(sha)

    print(f"\n✅ COMPLETE: Found {len(all_commits)} unique commits for {identifier}")
    print(f"📦 Extracting minimal metadata...")

    # Extract metadata for each commit
    # Revert status was checked in batch after all commits were fetched
    metadata_list = []
    for commit in all_commits:
        revert_status = commit.get('_revert_status')
        metadata_list.append(extract_commit_metadata(commit, revert_status))

    return metadata_list


# =============================================================================
# HUGGINGFACE STORAGE FUNCTIONS
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


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.
    """
    delay = 2.0

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
                delay = min(delay * 2, 60.0)
            else:
                print(f"   ✗ Upload failed after {max_retries} attempts: {str(e)}")
                raise


def update_revert_status_for_existing_dates(agent_identifier, existing_dates, token=None):
    """
    Update only the revert status for commits on existing dates.
    This is much more efficient than re-mining all commit data.

    Args:
        agent_identifier: GitHub identifier of the agent
        existing_dates: Set of (year, month, day) tuples that already have data
        token: GitHub API token for authentication

    Returns:
        Number of files updated
    """
    if not existing_dates:
        return 0

    print(f"\n🔄 Updating revert status for {len(existing_dates)} existing date(s)...")

    headers = {'Authorization': f'token {token}'} if token else {}
    hf_token = get_hf_token()
    if not hf_token:
        raise Exception("No HuggingFace token found")

    api = HfApi()
    updated_count = 0

    # Define time range for revert checking
    current_time = datetime.now(timezone.utc)
    start_date = current_time - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)
    end_date = current_time

    for (year, month, day) in sorted(existing_dates):
        filename = f"{agent_identifier}/{year}.{month:02d}.{day:02d}.jsonl"
        local_filename = f"{year}.{month:02d}.{day:02d}.jsonl"

        try:
            # Download existing file
            print(f"   📥 Downloading {filename}...")
            file_path = hf_hub_download(
                repo_id=COMMIT_METADATA_REPO,
                filename=filename,
                repo_type="dataset",
                token=hf_token
            )
            existing_metadata = load_jsonl(file_path)

            if not existing_metadata:
                print(f"   ⚠️  No commits found in {filename}, skipping")
                continue

            print(f"   🔍 Checking revert status for {len(existing_metadata)} commits...")

            # Build commits_by_sha dict for batch checking
            commits_by_sha = {}
            for meta in existing_metadata:
                sha = meta.get('sha')
                if sha:
                    # Create a minimal commit object for revert checking
                    commits_by_sha[sha] = {
                        'sha': sha,
                        'commit': {
                            'author': {
                                'date': meta.get('commit_at')
                            }
                        }
                    }

            # Batch check revert status
            revert_results = batch_check_reverts_for_commits(
                commits_by_sha,
                headers,
                start_date=start_date,
                end_date=end_date
            )

            # Update metadata with new revert status
            updated_metadata = []
            for meta in existing_metadata:
                sha = meta.get('sha')
                if sha and sha in revert_results:
                    revert_status = revert_results[sha]
                    meta['is_reverted'] = revert_status.get('is_reverted', False)
                    meta['revert_at'] = revert_status.get('revert_at')
                updated_metadata.append(meta)

            # Save locally
            save_jsonl(local_filename, updated_metadata)

            # Upload to HuggingFace
            try:
                upload_with_retry(
                    api=api,
                    path_or_fileobj=local_filename,
                    path_in_repo=filename,
                    repo_id=COMMIT_METADATA_REPO,
                    repo_type="dataset",
                    token=hf_token
                )
                print(f"   ✓ Updated revert status in {filename}")
                updated_count += 1
            finally:
                # Clean up local file
                if os.path.exists(local_filename):
                    os.remove(local_filename)

        except Exception as e:
            print(f"   ✗ Error updating {filename}: {str(e)}")
            continue

    print(f"✅ Updated revert status in {updated_count}/{len(existing_dates)} file(s)")
    return updated_count


def save_commit_metadata_to_hf(metadata_list, agent_identifier):
    """
    Save commit metadata to HuggingFace dataset, organized by [agent_identifier]/YYYY.MM.DD.jsonl.
    Each file is stored in the agent's folder and named YYYY.MM.DD.jsonl for that day's commits.

    This function APPENDS new metadata and DEDUPLICATES by sha.

    Args:
        metadata_list: List of commit metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi()

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        for (commit_year, month, day), day_metadata in grouped.items():
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


def get_existing_date_agent_combinations(agent_identifier):
    """
    Fetch existing (year, month, day) dates for a specific agent from HuggingFace.
    Returns a set of (year, month, day) tuples that already have data.
    """
    try:
        api = HfApi()

        # List all files in the repository
        files = api.list_repo_files(repo_id=COMMIT_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder with .jsonl extension
        # Expected format: agent_identifier/YYYY.MM.DD.jsonl
        existing_dates = set()
        prefix = f"{agent_identifier}/"

        for file in files:
            if file.startswith(prefix) and file.endswith('.jsonl'):
                # Extract date from filename: YYYY.MM.DD.jsonl
                filename = file[len(prefix):]  # Remove prefix
                try:
                    date_part = filename.replace('.jsonl', '')  # Remove extension
                    parts = date_part.split('.')
                    if len(parts) == 3:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                        existing_dates.add((year, month, day))
                except Exception as e:
                    print(f"   Warning: Could not parse date from filename {file}: {e}")
                    continue

        return existing_dates

    except Exception as e:
        print(f"   Warning: Could not fetch existing dates for {agent_identifier}: {e}")
        return set()


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
        return []


# =============================================================================
# MAIN MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine commit metadata for all agents within LEADERBOARD_TIME_FRAME_DAYS and save to HuggingFace.

    SMART MINING STRATEGY:
    1. Check which (date, agent) combinations already exist in COMMIT_METADATA_REPO
    2. For existing dates: Only update revert status (much faster)
    3. For missing dates: Mine commit metadata from scratch
    """
    token = get_github_token()

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    print(f"\n{'='*80}")
    print(f"Starting SMART commit metadata mining for {len(agents)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"{'='*80}\n")

    # Mine each agent
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

            # Step 1: Check which dates already exist for this agent
            print(f"\n📋 Checking existing data for {identifier}...")
            existing_dates = get_existing_date_agent_combinations(identifier)

            if existing_dates:
                print(f"   Found {len(existing_dates)} existing date(s)")

                # Step 2: Update revert status for existing dates
                print(f"\n🔄 Strategy: Update revert status for existing dates")
                update_revert_status_for_existing_dates(identifier, existing_dates, token)
            else:
                print(f"   No existing data found")

            # Step 3: Mine new commits (skipping existing dates)
            print(f"\n⛏️  Strategy: Mine new commits (skipping existing dates)")
            metadata = fetch_all_commits_metadata(
                identifier,
                agent_name,
                token,
                dates_to_skip=existing_dates
            )

            if metadata:
                print(f"💾 Saving {len(metadata)} new commit records...")
                save_commit_metadata_to_hf(metadata, identifier)
                print(f"✓ Successfully processed {agent_name}")
            else:
                if not existing_dates:
                    print(f"   No commits found for {agent_name}")
                else:
                    print(f"   No new commits to save (only updated existing data)")

        except Exception as e:
            print(f"✗ Error processing {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ SMART mining complete for all agents")
    print(f"{'='*80}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()
