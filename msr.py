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


def get_github_tokens():
    """
    Get all GitHub tokens from environment variables.
    Returns a list of tokens that have the GITHUB_TOKEN prefix.
    """
    tokens = []
    for key, value in os.environ.items():
        if key.startswith('GITHUB_TOKEN') and value:
            tokens.append(value)

    if not tokens:
        print("Warning: No GITHUB_TOKEN* found. API rate limits: 60/hour (authenticated: 5000/hour)")
    else:
        print(f"Loaded {len(tokens)} GitHub token(s) from environment")

    return tokens


class TokenPool:
    """
    Hybrid token pool with parallel execution and round-robin fallback.

    Strategy:
    - Splits tokens 50/50 into parallel pool and round-robin backup pool
    - Parallel pool: Used for concurrent API calls to maximize throughput
    - Round-robin pool: Fallback when parallel tokens hit rate limits
    - Rate limit tracking: Automatically marks and recovers tokens based on reset timestamps

    Example token distribution:
    - 1 token: parallel=[token], roundrobin=[token] (same token used)
    - 2 tokens: parallel=[token1], roundrobin=[token2]
    - 4 tokens: parallel=[token1, token2], roundrobin=[token3, token4]
    - 6 tokens: parallel=[token1, token2, token3], roundrobin=[token4, token5, token6]
    """
    def __init__(self, tokens):
        import threading

        self.all_tokens = tokens if tokens else [None]
        self.lock = threading.Lock()

        # Split tokens into parallel and round-robin pools (50/50)
        total_tokens = len(self.all_tokens)
        split_point = max(1, total_tokens // 2)

        self.parallel_tokens = self.all_tokens[:split_point]
        self.roundrobin_tokens = self.all_tokens[split_point:] if total_tokens > 1 else self.all_tokens

        # Round-robin state
        self.current_index = 0

        # Rate limit tracking: {token: reset_timestamp}
        self.rate_limited_parallel = {}
        self.rate_limited_roundrobin = {}

        # Statistics
        self.stats = {
            'parallel_calls': 0,
            'roundrobin_calls': 0,
            'fallback_triggers': 0
        }

        print(f"🔀 Hybrid Token Pool initialized:")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Parallel pool: {len(self.parallel_tokens)} tokens")
        print(f"   Round-robin pool: {len(self.roundrobin_tokens)} tokens")

    def _clean_expired_rate_limits(self):
        """Remove tokens from rate-limited sets if their rate limit has expired."""
        current_time = time.time()

        # Clean parallel pool
        expired_parallel = [token for token, reset_ts in self.rate_limited_parallel.items()
                          if current_time >= reset_ts]
        for token in expired_parallel:
            del self.rate_limited_parallel[token]

        # Clean round-robin pool
        expired_roundrobin = [token for token, reset_ts in self.rate_limited_roundrobin.items()
                             if current_time >= reset_ts]
        for token in expired_roundrobin:
            del self.rate_limited_roundrobin[token]

    def get_parallel_token(self):
        """
        Get an available token from the parallel pool.
        Returns None if all parallel tokens are rate-limited.
        """
        with self.lock:
            self._clean_expired_rate_limits()

            # Find first available parallel token (not rate-limited)
            for token in self.parallel_tokens:
                if token not in self.rate_limited_parallel:
                    self.stats['parallel_calls'] += 1
                    return token

            return None

    def get_roundrobin_token(self):
        """Get the next token from round-robin pool (fallback)."""
        with self.lock:
            self._clean_expired_rate_limits()

            if not self.roundrobin_tokens:
                return None

            # Try to find an available token (not rate-limited)
            attempts = 0
            while attempts < len(self.roundrobin_tokens):
                token = self.roundrobin_tokens[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.roundrobin_tokens)

                if token not in self.rate_limited_roundrobin:
                    self.stats['roundrobin_calls'] += 1
                    return token

                attempts += 1

            # All tokens are rate-limited, return None
            return None

    def get_next_token(self):
        """
        Get next available token with automatic fallback.
        Tries parallel pool first, falls back to round-robin if needed.
        """
        # Try parallel pool first
        token = self.get_parallel_token()
        if token:
            return token

        # Fallback to round-robin pool
        with self.lock:
            self.stats['fallback_triggers'] += 1

        return self.get_roundrobin_token()

    def get_headers(self):
        """Get headers with the next token in rotation."""
        token = self.get_next_token()
        return {'Authorization': f'token {token}'} if token else {}

    def mark_rate_limited(self, token, reset_timestamp=None):
        """
        Mark a token as rate-limited with optional reset timestamp.

        Args:
            token: The token to mark as rate-limited
            reset_timestamp: Unix timestamp when rate limit resets (optional)
        """
        if reset_timestamp is None:
            # Default: assume rate limit for 1 hour
            reset_timestamp = time.time() + 3600

        with self.lock:
            # Determine which pool this token belongs to
            if token in self.parallel_tokens:
                self.rate_limited_parallel[token] = reset_timestamp
            elif token in self.roundrobin_tokens:
                self.rate_limited_roundrobin[token] = reset_timestamp

    def get_available_parallel_tokens(self):
        """
        Get list of all available (non-rate-limited) parallel tokens.
        Used for parallel execution.
        """
        with self.lock:
            self._clean_expired_rate_limits()
            return [token for token in self.parallel_tokens
                   if token not in self.rate_limited_parallel]

    def get_stats(self):
        """Get token pool usage statistics."""
        with self.lock:
            self._clean_expired_rate_limits()
            return {
                'parallel_calls': self.stats['parallel_calls'],
                'roundrobin_calls': self.stats['roundrobin_calls'],
                'fallback_triggers': self.stats['fallback_triggers'],
                'parallel_rate_limited': len(self.rate_limited_parallel),
                'roundrobin_rate_limited': len(self.rate_limited_roundrobin)
            }

    def print_stats(self):
        """Print token pool usage statistics."""
        stats = self.get_stats()
        total_calls = stats['parallel_calls'] + stats['roundrobin_calls']

        print(f"\n📊 Token Pool Statistics:")
        print(f"   Total API calls: {total_calls}")
        if total_calls > 0:
            parallel_pct = (stats['parallel_calls'] / total_calls) * 100
            roundrobin_pct = (stats['roundrobin_calls'] / total_calls) * 100
            print(f"   Parallel calls: {stats['parallel_calls']} ({parallel_pct:.1f}%)")
            print(f"   Round-robin calls: {stats['roundrobin_calls']} ({roundrobin_pct:.1f}%)")
        print(f"   Fallback triggers: {stats['fallback_triggers']}")
        print(f"   Currently rate-limited: {stats['parallel_rate_limited']} parallel, {stats['roundrobin_rate_limited']} round-robin")


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# GITHUB API FUNCTIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30, token_pool=None, token=None):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.

    Args:
        token_pool: Optional TokenPool instance for rate limit tracking
        token: Optional token being used for this request (for rate limit marking)

    Returns the final requests.Response on success or non-retryable status, or None after exhausting retries.
    """
    from datetime import datetime, timezone

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
                reset_timestamp = None

                # Prefer Retry-After when present
                retry_after = resp.headers.get('Retry-After') or resp.headers.get('retry-after')
                if retry_after:
                    try:
                        wait = float(retry_after)
                        reset_timestamp = time.time() + wait
                    except Exception:
                        wait = None

                # Fallback to X-RateLimit-Reset when 403/429
                if wait is None and status in (403, 429):
                    reset_hdr = resp.headers.get('X-RateLimit-Reset') or resp.headers.get('x-ratelimit-reset')
                    if reset_hdr:
                        try:
                            reset_timestamp = int(float(reset_hdr))
                            wait = max(reset_timestamp - time.time() + 2, 1)
                        except Exception:
                            wait = None

                # Mark token as rate-limited if we have the necessary info
                if status in (403, 429) and token_pool and token and reset_timestamp:
                    token_pool.mark_rate_limited(token, reset_timestamp)
                    print(f"   ⚠️ Marked token as rate-limited until {datetime.fromtimestamp(reset_timestamp, timezone.utc).isoformat()}")

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


def batch_check_reverts_for_commits(commits_by_sha, token_pool, start_date=None, end_date=None):
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
        token_pool: TokenPool instance for rotating through GitHub tokens
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

            headers = token_pool.get_headers()
            headers['Accept'] = 'application/vnd.github.cloak-preview+json'

            response = request_with_backoff('GET', url, headers=headers, params=params, max_retries=3)

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


def fetch_commits_with_time_partition(base_query, start_date, end_date, token_pool, commits_by_sha, depth=0, dates_to_skip=None, max_recursion_depth=8):
    """
    Fetch commits within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.
    Uses intelligent splitting: days at shallow depth, then hours/minutes/seconds at deeper depths.

    Args:
        base_query: GitHub search query pattern
        start_date: Start datetime for the range
        end_date: End datetime for the range
        token_pool: TokenPool instance for rotating through GitHub tokens
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

        headers = token_pool.get_headers()
        headers['Accept'] = 'application/vnd.github.cloak-preview+json'

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
                        base_query, split_start, split_end, token_pool, commits_by_sha,
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


def fetch_commits_parallel(query_patterns, start_date, end_date, token_pool, commits_by_sha, max_workers=None):
    """
    Fetch commits using parallel execution with multiple query patterns.
    Uses the parallel token pool to make concurrent API calls for maximum throughput.

    Args:
        query_patterns: List of query patterns to search with
        start_date: Start datetime for the range
        end_date: End datetime for the range
        token_pool: TokenPool instance with parallel tokens
        commits_by_sha: Dict to accumulate commits keyed by SHA
        max_workers: Maximum number of concurrent workers (defaults to number of available parallel tokens)

    Returns:
        Total number of commits found across all query patterns
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Get available parallel tokens
    available_tokens = token_pool.get_available_parallel_tokens()

    if not available_tokens or len(query_patterns) <= 1:
        # Fallback to sequential execution
        if not available_tokens:
            print("   ⚠️ No parallel tokens available, using sequential fallback")
        total = 0
        for query_pattern in query_patterns:
            count = fetch_commits_with_time_partition(
                query_pattern, start_date, end_date, token_pool, commits_by_sha, depth=0
            )
            total += count
        return total

    # Use parallel execution
    if max_workers is None:
        max_workers = len(available_tokens)

    print(f"   🚀 Using parallel execution with {max_workers} workers for {len(query_patterns)} query patterns")

    def fetch_pattern(query_pattern, token):
        """Wrapper function for parallel execution."""
        pattern_commits = {}
        headers = {'Authorization': f'token {token}'} if token else {}

        count = fetch_commits_with_time_partition(
            query_pattern, start_date, end_date, token_pool, pattern_commits, depth=0
        )
        return (query_pattern, pattern_commits, count)

    total_found = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all patterns for parallel execution
        futures = {}
        for idx, query_pattern in enumerate(query_patterns):
            # Get a token for this pattern (round-robin through available parallel tokens)
            token = available_tokens[idx % len(available_tokens)]
            future = executor.submit(fetch_pattern, query_pattern, token)
            futures[future] = query_pattern

        # Collect results as they complete
        for future in as_completed(futures):
            query_pattern = futures[future]
            try:
                pattern, pattern_commits, count = future.result()

                # Merge pattern commits into global dict
                for sha, commit in pattern_commits.items():
                    if sha not in commits_by_sha:
                        commits_by_sha[sha] = commit
                        total_found += 1

                print(f"   ✓ Pattern '{pattern}' completed: {count} commits")
            except Exception as e:
                print(f"   ✗ Error in pattern '{query_pattern}': {str(e)}")

    print(f"   ✅ Parallel execution complete: {total_found} unique commits from {len(query_patterns)} patterns")
    return total_found


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


def fetch_all_commits_metadata(identifier, agent_name, token_pool, dates_to_skip=None):
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
        token_pool: TokenPool instance for rotating through GitHub tokens
        dates_to_skip: Set of (year, month, day) tuples to skip during mining

    Returns:
        List of dictionaries containing minimal commit metadata with revert status
    """
    if dates_to_skip is None:
        dates_to_skip = set()

    # Define query pattern
    query_pattern = f'is:commit author:{identifier}'

    commits_by_sha = {}

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS, excluding today (12am onwards)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)  # 12am today
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    print(f"\n🔍 Searching with query: {query_pattern}")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (excluding today)")
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
            token_pool,
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
            token_pool,
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


def update_revert_status_for_existing_dates(agent_identifier, existing_dates, token_pool):
    """
    Update only the revert status for commits on existing dates.
    This is much more efficient than re-mining all commit data.
    Uses BATCH UPLOAD to avoid HuggingFace rate limits.

    Args:
        agent_identifier: GitHub identifier of the agent
        existing_dates: Set of (year, month, day) tuples that already have data
        token_pool: TokenPool instance for rotating through GitHub tokens

    Returns:
        Number of files updated
    """
    import tempfile
    import shutil

    if not existing_dates:
        return 0

    print(f"\n🔄 Updating revert status for {len(existing_dates)} existing date(s)...")
    hf_token = get_hf_token()
    if not hf_token:
        raise Exception("No HuggingFace token found")

    api = HfApi()

    # Define time range for revert checking
    current_time = datetime.now(timezone.utc)
    start_date = current_time - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)
    end_date = current_time

    # Create temporary directory for batch upload
    temp_dir = tempfile.mkdtemp(prefix=f"hf_revert_update_{agent_identifier}_")
    agent_folder = os.path.join(temp_dir, agent_identifier)
    os.makedirs(agent_folder, exist_ok=True)

    try:
        updated_count = 0

        for (year, month, day) in sorted(existing_dates):
            filename = f"{agent_identifier}/{year}.{month:02d}.{day:02d}.jsonl"
            local_filename = os.path.join(agent_folder, f"{year}.{month:02d}.{day:02d}.jsonl")

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
                    token_pool,
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

                # Save to temp directory
                save_jsonl(local_filename, updated_metadata)
                print(f"   ✓ Prepared updated {filename}")
                updated_count += 1

            except Exception as e:
                print(f"   ✗ Error processing {filename}: {str(e)}")
                continue

        # Batch upload all updated files in a single commit
        if updated_count > 0:
            print(f"📤 Batch uploading {updated_count} updated file(s)...")
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=COMMIT_METADATA_REPO,
                repo_type="dataset",
                token=hf_token,
                commit_message=f"Batch revert status update: {agent_identifier} ({updated_count} files)"
            )
            print(f"✅ Batch upload complete!")
        else:
            print("   No files were updated")

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    print(f"✅ Updated revert status in {updated_count}/{len(existing_dates)} file(s)")
    return updated_count


def save_commit_metadata_to_hf(metadata_list, agent_identifier):
    """
    Save commit metadata to HuggingFace dataset, organized by [agent_identifier]/YYYY.MM.DD.jsonl.
    Each file is stored in the agent's folder and named YYYY.MM.DD.jsonl for that day's commits.

    This function APPENDS new metadata and DEDUPLICATES by sha.
    Uses BATCH UPLOAD to avoid HuggingFace rate limits (256 commits/hour).

    Args:
        metadata_list: List of commit metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    import tempfile
    import shutil

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi()

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        if not grouped:
            print("   No metadata to save")
            return True

        # Create temporary directory for batch upload
        temp_dir = tempfile.mkdtemp(prefix=f"hf_upload_{agent_identifier}_")
        agent_folder = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_folder, exist_ok=True)

        try:
            print(f"📦 Preparing batch upload for {len(grouped)} daily file(s)...")

            # Prepare all daily files in temp directory
            for (commit_year, month, day), day_metadata in grouped.items():
                filename = f"{agent_identifier}/{commit_year}.{month:02d}.{day:02d}.jsonl"
                local_filename = os.path.join(agent_folder, f"{commit_year}.{month:02d}.{day:02d}.jsonl")

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
                    print(f"   Creating new file: {filename}")

                # Merge and deduplicate by sha
                existing_by_sha = {meta['sha']: meta for meta in existing_metadata if meta.get('sha')}
                new_by_sha = {meta['sha']: meta for meta in day_metadata if meta.get('sha')}

                # Update with new data (new data overwrites old)
                existing_by_sha.update(new_by_sha)
                merged_metadata = list(existing_by_sha.values())

                # Save to temp directory
                save_jsonl(local_filename, merged_metadata)
                print(f"   ✓ Prepared {len(merged_metadata)} commits for {filename}")

            # Batch upload entire folder (single commit instead of N commits)
            print(f"📤 Uploading {len(grouped)} file(s) in single batch commit...")
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=COMMIT_METADATA_REPO,
                repo_type="dataset",
                token=token,
                commit_message=f"Batch update: {agent_identifier} ({len(grouped)} daily files)"
            )
            print(f"   ✅ Batch upload complete!")

        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

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

    Uses token pool to rotate through multiple GitHub tokens to minimize rate limiting.
    """
    # Initialize token pool from all GITHUB_TOKEN* environment variables
    tokens = get_github_tokens()
    token_pool = TokenPool(tokens)

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
                update_revert_status_for_existing_dates(identifier, existing_dates, token_pool)
            else:
                print(f"   No existing data found")

            # Step 3: Mine new commits (skipping existing dates)
            print(f"\n⛏️  Strategy: Mine new commits (skipping existing dates)")
            metadata = fetch_all_commits_metadata(
                identifier,
                agent_name,
                token_pool,
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
