# github_repository_analyzer/data_collector.py
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from config import (
    GITHUB_API_BASE_URL, DEFAULT_USER_AGENT, DEFAULT_RATE_LIMIT,
    RATE_LIMIT_BUFFER, RETRY_WAIT_TIME, DEFAULT_PER_PAGE, MAX_PER_PAGE
)

class GitHubAPIClient:
    def __init__(self, token: Optional[str] = None, base_url: str = GITHUB_API_BASE_URL):
        if not token:
            raise ValueError("GitHub token required for API access")
        self.token = token
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": DEFAULT_USER_AGENT
        }
        self.rate_limit_remaining = DEFAULT_RATE_LIMIT
        self.rate_limit_reset = None

    def check_rate_limit(self) -> bool:
        url = f"{self.base_url}/rate_limit"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                core = data['resources']['core']
                self.rate_limit_remaining = core['remaining']
                self.rate_limit_reset = datetime.fromtimestamp(core['reset'])
                return True
            return False
        except requests.RequestException as e:
            print(f"Error checking rate limit: {e}")
            return False

    def wait_for_rate_limit(self) -> None:
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= 0:
            now = datetime.now()
            wait_time = (self.rate_limit_reset - now).total_seconds() if self.rate_limit_reset else 60
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {int(wait_time)} seconds until reset...")
                time.sleep(wait_time + 1)
            self.check_rate_limit()

    def get_date_range(self, time_window: str) -> str:
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
            start_date = today - timedelta(days=365)
        return f"{start_date.strftime('%Y-%m-%d')}..{today.strftime('%Y-%m-%d')}"

    def search_repositories(self, query: str, sort: str = 'stars', order: str = 'desc',
                            per_page: int = DEFAULT_PER_PAGE, max_repos: int = 1000,
                            time_window: Optional[str] = None) -> List[Dict]:
        repositories = []
        page = 1
        total_collected = 0

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
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    repos = data.get('items', [])
                    if not repos:
                        break
                    for repo in repos:
                        repo_data = self.extract_repository_features(repo)
                        repositories.append(repo_data)
                        total_collected += 1
                    page += 1
                    self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 1))
                    self.rate_limit_reset = datetime.fromtimestamp(
                        int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                    )
                elif response.status_code == 403 and 'rate limit' in response.text.lower():
                    self.check_rate_limit()
                    self.wait_for_rate_limit()
                elif response.status_code == 422:
                    print(f"Invalid query: {query}")
                    break
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    break
            except requests.RequestException as e:
                print(f"Request error: {e}")
                time.sleep(2)
        return repositories

    def extract_repository_features(self, repo: Dict) -> Dict:
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