"""
GitHub API data collection and rate limiting functionality
"""

import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .config import (
    GITHUB_API_BASE_URL, DEFAULT_USER_AGENT, DEFAULT_RATE_LIMIT,
    RATE_LIMIT_BUFFER, RETRY_WAIT_TIME, DEFAULT_PER_PAGE, MAX_PER_PAGE
)


class GitHubAPIClient:
    """
    Handles GitHub API interactions, rate limiting, and data collection
    """

    def __init__(self, token: Optional[str] = None, base_url: str = GITHUB_API_BASE_URL):
        """
        Initialize GitHub API client

        Args:
            token: GitHub personal access token
            base_url: GitHub API base URL
        """
        self.token = token
        self.base_url = base_url
        self.headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': DEFAULT_USER_AGENT
        }
        if token:
            self.headers['Authorization'] = f'token {token}'

        self.rate_limit_remaining = DEFAULT_RATE_LIMIT
        self.rate_limit_reset = None

    def check_rate_limit(self) -> bool:
        """Check current GitHub API rate limit status"""
        url = f"{self.base_url}/rate_limit"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                self.rate_limit_remaining = data['rate']['remaining']
                self.rate_limit_reset = datetime.fromtimestamp(data['rate']['reset'])
                print(f"Rate limit remaining: {self.rate_limit_remaining}")
                return True
            return False
        except requests.RequestException as e:
            print(f"Error checking rate limit: {e}")
            return False

    def wait_for_rate_limit(self) -> None:
        """Wait if the rate limit is approaching exhaustion"""
        if self.rate_limit_remaining <= RATE_LIMIT_BUFFER:
            if self.rate_limit_reset:
                wait_time = (self.rate_limit_reset - datetime.now()).total_seconds()
                if wait_time > 0:
                    print(f"Rate limit protection: waiting {wait_time:.0f} seconds...")
                    time.sleep(wait_time + 1)
                    self.check_rate_limit()

    def get_date_range(self, time_window: str) -> str:
        """
        Generate date range string for GitHub API queries

        Args:
            time_window: Time window specification

        Returns:
            Date range string in format 'YYYY-MM-DD.YYYY-MM-DD'
        """
        today = datetime.now().date()

        if time_window == "last 12 months":
            start_date = today - timedelta(days=365)
        elif time_window == "last 6 months":
            start_date = today - timedelta(days=180)
        elif time_window == "last month":
            start_date = today.replace(day=1) - timedelta(days=1)
            start_date = start_date.replace(day=1)
        elif time_window == "this month":
            start_date = today.replace(day=1)
        elif time_window == "last two weeks":
            start_date = today - timedelta(days=14)
        elif time_window == "last week":
            start_date = today - timedelta(days=7)
        elif time_window == "today":
            start_date = today
        else:
            start_date = today - timedelta(days=365)  # default to last year

        return f"{start_date.strftime('%Y-%m-%d')}..{today.strftime('%Y-%m-%d')}"

    def search_repositories(self, query: str, sort: str = 'stars', order: str = 'desc',
                            per_page: int = DEFAULT_PER_PAGE, max_repos: int = 1000,
                            time_window: Optional[str] = None) -> List[Dict]:
        """
        Search GitHub repositories using the Search API

        Args:
            query: Search query string
            sort: Sort field ('stars', 'forks', 'updated')
            order: Sort order ('asc', 'desc')
            per_page: Results per page (max 100)
            max_repos: Maximum repositories to collect
            time_window: Time window for filtering results

        Returns:
            List of repository data dictionaries
        """
        repositories = []
        page = 1
        total_collected = 0

        # Add time filter to the query if specified
        if time_window:
            date_range = self.get_date_range(time_window)
            query = f"{query} updated:{date_range}"

        while total_collected < max_repos:
            self.wait_for_rate_limit()

            url = f"{self.base_url}/search/repositories"
            params = {
                'q': query,
                'sort': sort,
                'order': order,
                'per_page': min(per_page, max_repos - total_collected, MAX_PER_PAGE),
                'page': page
            }

            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    repos = data.get('items', [])

                    if not repos:  # No more results
                        break

                    for repo in repos:
                        repo_data = self.extract_repository_features(repo)
                        repositories.append(repo_data)
                        total_collected += 1

                    print(f"Collected {total_collected} repositories (Page {page})")
                    page += 1

                    # Update rate limit info from response headers
                    self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))

                elif response.status_code == 403:
                    print("Rate limit exceeded. Waiting...")
                    time.sleep(RETRY_WAIT_TIME)
                elif response.status_code == 422:
                    print(f"Invalid query: {query}")
                    break
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    break

            except requests.RequestException as e:
                print(f"Request error: {e}")
                time.sleep(5)  # Brief wait before retry

        return repositories

    def extract_repository_features(self, repo: Dict) -> Dict:
        """
        Extract relevant features from GitHub repository data

        Args:
            repo: Raw repository data from GitHub API

        Returns:
            Dictionary with processed repository features
        """
        return {
            'id': repo.get('id'),
            'name': repo.get('name'),
            'full_name': repo.get('full_name'),
            'owner': repo.get('owner', {}).get('login'),
            'description': repo.get('description', ''),
            'language': repo.get('language'),
            'stars': repo.get('stargazers_count', 0),
            'forks': repo.get('forks_count', 0),
            'watchers': repo.get('watchers_count', 0),
            'issues': repo.get('open_issues_count', 0),
            'size': repo.get('size', 0),
            'created_at': repo.get('created_at'),
            'updated_at': repo.get('updated_at'),
            'pushed_at': repo.get('pushed_at'),
            'topics': repo.get('topics', []),
            'license': repo.get('license', {}).get('name') if repo.get('license') else None,
            'has_wiki': repo.get('has_wiki', False),
            'has_pages': repo.get('has_pages', False),
            'has_downloads': repo.get('has_downloads', False),
            'archived': repo.get('archived', False),
            'disabled': repo.get('disabled', False),
            'default_branch': repo.get('default_branch'),
            'score': repo.get('score', 0)
        }

    def get_repository_details(self, full_name: str) -> Optional[Dict]:
        """
        Get detailed information for a specific repository

        Args:
            full_name: Repository full name (owner/repo)

        Returns:
            Detailed repository information or None if there is an error
        """
        self.wait_for_rate_limit()

        url = f"{self.base_url}/repos/{full_name}"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching {full_name}: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Request error for {full_name}: {e}")
            return None