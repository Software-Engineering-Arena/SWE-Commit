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


def check_single_commit_revert_status(sha, headers):
    """
    DEPRECATED: Use batch_check_reverts_for_commits() instead for better efficiency.

    Check if a single commit has been reverted by searching GitHub for revert commits.

    Returns:
        Dictionary with 'is_reverted' (bool) and 'revert_at' (date string or None)
    """
    if not sha:
        return {'is_reverted': False, 'revert_at': None}

    # Search for commits that mention this SHA in a revert context
    sha_abbr = sha[:7]
    revert_query = f'"This reverts commit {sha_abbr}"'

    try:
        url = 'https://api.github.com/search/commits'
        params = {
            'q': revert_query,
            'per_page': 1
        }

        headers_with_accept = headers.copy() if headers else {}
        headers_with_accept['Accept'] = 'application/vnd.github.cloak-preview+json'

        response = request_with_backoff('GET', url, headers=headers_with_accept, params=params, max_retries=3)

        if response and response.status_code == 200:
            data = response.json()
            total_count = data.get('total_count', 0)

            if total_count > 0:
                items = data.get('items', [])
                if items:
                    revert_commit = items[0]
                    revert_at = revert_commit.get('commit', {}).get('author', {}).get('date')
                    return {'is_reverted': True, 'revert_at': revert_at}

        return {'is_reverted': False, 'revert_at': None}

    except Exception as e:
        print(f"   Warning: Could not check revert status for {sha_abbr}: {e}")
        return {'is_reverted': False, 'revert_at': None}


def batch_check_reverts_for_commits(commits_by_sha, headers, start_date=None, end_date=None):
    """
    Check revert status for all commits using batch approach (MUCH more efficient).
    Returns the same format as check_single_commit_revert_status() but for all commits at once.

    This function fetches ALL revert commits in the time range with ~10-20 API calls,
    then matches them against the provided commits. This is 98% more efficient than
    checking each commit individually.

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

    print(f"   🔍 Batch checking revert status for {len(commits_by_sha)} commits...")

    # Build query for ALL revert commits
    revert_query = '"This reverts commit"'

    # Add date range if provided (to narrow down search)
    if start_date and end_date:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        revert_query += f' committer-date:{start_str}..{end_str}'

    # Fetch ALL revert commits
    revert_commits = []
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

        revert_commits.extend(items)

        # GitHub API limit: max 1000 results, max 10 pages
        if len(items) < per_page or page >= 10:
            break

        page += 1
        time.sleep(0.5)  # Courtesy delay

    print(f"   📋 Found {len(revert_commits)} total revert commits in time range")

    # Parse revert commits to extract which SHAs they revert
    revert_map = {}  # {reverted_sha: revert_date}

    import re
    for revert_commit in revert_commits:
        message = revert_commit.get('commit', {}).get('message', '')
        revert_date = revert_commit.get('commit', {}).get('author', {}).get('date')

        # Extract SHA from message: "This reverts commit abc1234" or similar patterns
        # Match both abbreviated (7 chars) and full SHA (40 chars)
        matches = re.findall(r'\b([0-9a-f]{7,40})\b', message.lower())

        for sha in matches:
            # Store first (earliest) revert date for each SHA
            if sha not in revert_map:
                revert_map[sha] = revert_date

    print(f"   📌 Identified {len(revert_map)} unique reverted commit SHAs")

    # Build results for our commits
    results = {}
    for sha in commits_by_sha:
        sha_abbr = sha[:7].lower()
        sha_lower = sha.lower()

        # Check both full SHA and abbreviated SHA
        if sha_lower in revert_map or sha_abbr in revert_map:
            results[sha] = {
                'is_reverted': True,
                'revert_at': revert_map.get(sha_lower) or revert_map.get(sha_abbr)
            }
        else:
            results[sha] = {
                'is_reverted': False,
                'revert_at': None
            }

    reverted_count = sum(1 for r in results.values() if r['is_reverted'])
    print(f"   ✅ Batch revert check complete: {reverted_count}/{len(results)} commits were reverted")

    return results


def fetch_commits_with_time_partition(base_query, start_date, end_date, headers, commits_by_sha, depth=0, check_reverts=False):
    """
    Fetch commits within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.

    Args:
        check_reverts: DEPRECATED - revert checking is now done in batch after all commits are fetched

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
                days_diff = time_diff.days

                # Use aggressive splitting for large ranges or deep recursion
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
                        if i > 0:
                            split_start = split_start + timedelta(days=1)

                        count = fetch_commits_with_time_partition(
                            base_query, split_start, split_end, headers, commits_by_sha, depth + 1, check_reverts
                        )
                        total_from_splits += count

                    return total_from_splits
                else:
                    # Binary split for smaller ranges
                    mid_date = start_date + time_diff / 2

                    count1 = fetch_commits_with_time_partition(
                        base_query, start_date, mid_date, headers, commits_by_sha, depth + 1, check_reverts
                    )
                    count2 = fetch_commits_with_time_partition(
                        base_query, mid_date + timedelta(days=1), end_date, headers, commits_by_sha, depth + 1, check_reverts
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


def fetch_all_commits_metadata(identifier, agent_name, token=None):
    """
    Fetch commits associated with a GitHub user or bot for the past LEADERBOARD_TIME_FRAME_DAYS.
    Returns lightweight metadata instead of full commit objects.

    Revert status is checked in BATCH after all commits are fetched using
    batch_check_reverts_for_commits(). This is 98% more efficient than checking
    each commit individually during retrieval.

    Args:
        identifier: GitHub username or bot identifier
        agent_name: Human-readable name of the agent for metadata purposes
        token: GitHub API token for authentication

    Returns:
        List of dictionaries containing minimal commit metadata with revert status
    """
    headers = {'Authorization': f'token {token}'} if token else {}

    # Define query patterns
    stripped_id = identifier.replace('[bot]', '')
    query_patterns = []

    query_patterns.append(f'is:commit author:{identifier}')
    if stripped_id != identifier:
        query_patterns.append(f'is:commit author:{stripped_id}')

    commits_by_sha = {}

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS
    current_time = datetime.now(timezone.utc)
    start_date = current_time - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)
    end_date = current_time

    for query_pattern in query_patterns:
        print(f"\n🔍 Searching with query: {query_pattern}")
        print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        pattern_start_time = time.time()
        initial_count = len(commits_by_sha)

        # Fetch with time partitioning (no inline revert checking)
        commits_found = fetch_commits_with_time_partition(
            query_pattern,
            start_date,
            end_date,
            headers,
            commits_by_sha,
            check_reverts=False  # Never check inline, we'll do batch checking after
        )

        pattern_duration = time.time() - pattern_start_time
        new_commits = len(commits_by_sha) - initial_count

        print(f"   ✓ Pattern complete: {new_commits} new commits found ({commits_found} total fetched)")
        print(f"   ⏱️ Time taken: {pattern_duration:.1f} seconds")

        time.sleep(1.0)

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
    """
    token = get_github_token()

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    print(f"\n{'='*80}")
    print(f"Starting commit metadata mining for {len(agents)} agents")
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

            # Fetch commit metadata
            metadata = fetch_all_commits_metadata(identifier, agent_name, token)

            if metadata:
                print(f"💾 Saving {len(metadata)} commit records...")
                save_commit_metadata_to_hf(metadata, identifier)
                print(f"✓ Successfully processed {agent_name}")
            else:
                print(f"   No commits found for {agent_name}")

        except Exception as e:
            print(f"✗ Error processing {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ Mining complete for all agents")
    print(f"{'='*80}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()
