"""
Utility functions for data validation, file operations, and helpers
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import pandas as pd

from .config import DEFAULT_FILENAME_PREFIX, TIMESTAMP_FORMAT


class DataValidator:
    """Data validation utilities"""

    @staticmethod
    def validate_repository_data(repo_data: Dict) -> bool:
        """
        Validate repository data structure

        Args:
            repo_data: Repository data dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['id', 'name', 'full_name', 'stars', 'forks']
        return all(field in repo_data for field in required_fields)

    @staticmethod
    def validate_api_token(token: str) -> bool:
        """
        Basic validation for GitHub API token format

        Args:
            token: GitHub API token

        Returns:
            True if format is valid
        """
        if not token or not isinstance(token, str):
            return False

        # GitHub personal access tokens are typically 40 characters
        # GitHub Apps tokens start with 'ghs_' and are longer
        return (len(token) == 40 and token.startswith(('ghp_', 'gho_', 'ghu_', 'ghs_'))) or \
            (len(token) > 40 and token.startswith('ghs_'))

    @staticmethod
    def clean_repository_data(repositories: List[Dict]) -> List[Dict]:
        """
        Clean and validate repository data

        Args:
            repositories: List of repository dictionaries

        Returns:
            List of cleaned repository data
        """
        cleaned_repos = []

        for repo in repositories:
            if DataValidator.validate_repository_data(repo):
                # Ensure numeric fields are properly typed
                numeric_fields = ['stars', 'forks', 'watchers', 'issues', 'size']
                for field in numeric_fields:
                    if field in repo:
                        try:
                            repo[field] = int(repo[field]) if repo[field] is not None else 0
                        except (ValueError, TypeError):
                            repo[field] = 0

                # Clean text fields
                text_fields = ['name', 'description', 'language']
                for field in text_fields:
                    if field in repo and repo[field] is not None:
                        repo[field] = str(repo[field]).strip()

                # Ensure topics is a list
                if 'topics' in repo:
                    if not isinstance(repo['topics'], list):
                        repo['topics'] = []

                cleaned_repos.append(repo)

        return cleaned_repos


class FileManager:
    """File operations and data persistence"""

    @staticmethod
    def save_to_json(data: Any, filename: str, indent: int = 2) -> bool:
        """
        Save data to JSON file

        Args:
            data: Data to save
            filename: Output filename
            indent: JSON indentation

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving JSON to {filename}: {e}")
            return False

    @staticmethod
    def load_from_json(filename: str) -> Optional[Any]:
        """
        Load data from JSON file

        Args:
            filename: Input filename

        Returns:
            Loaded data or None if error
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {filename}: {e}")
            return None

    @staticmethod
    def save_to_csv(data: List[Dict], filename: str) -> bool:
        """
        Save repository data to a CSV file

        Args:
            data: List of dictionaries to save
            filename: Output filename

        Returns:
            True if successful, False otherwise
        """
        if not data:
            print("No data to save to CSV")
            return False

        try:
            df = pd.DataFrame(data)
            # Flatten lists in the topic column
            if 'topics' in df.columns:
                df['topics'] = df['topics'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

            df.to_csv(filename, index=False, encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error saving CSV to {filename}: {e}")
            return False

    @staticmethod
    def create_timestamped_filename(prefix: str = DEFAULT_FILENAME_PREFIX,
                                    suffix: str = "", extension: str = "json") -> str:
        """
        Create filename with a timestamp

        Args:
            prefix: Filename prefix
            suffix: Filename suffix
            extension: File extension

        Returns:
            Timestamped filename
        """
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        if suffix:
            return f"{prefix}_{suffix}_{timestamp}.{extension}"
        return f"{prefix}_{timestamp}.{extension}"

    @staticmethod
    def ensure_directory_exists(directory: str) -> bool:
        """
        Ensure a directory exists, create if not

        Args:
            directory: Directory path

        Returns:
            True if a directory exists or was created
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False


class DateTimeHelper:
    """Date and time utility functions"""

    @staticmethod
    def parse_github_date(date_string: str) -> Optional[datetime]:
        """
        Parse GitHub API date string to datetime object

        Args:
            date_string: Date string from GitHub API

        Returns:
            Datetime object or None if parsing fails
        """
        if not date_string:
            return None

        try:
            # GitHub API returns ISO 8601 format: 2023-01-01T12:00:00Z
            return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def calculate_age_in_days(created_date: str) -> Optional[int]:
        """
        Calculate repository age in days

        Args:
            created_date: Repository creation date string

        Returns:
            Age in days or None if calculation fails
        """
        created = DateTimeHelper.parse_github_date(created_date)
        if created:
            return (datetime.now(created.tzinfo) - created).days
        return None

    @staticmethod
    def get_time_window_dates(window: str) -> tuple:
        """
        Get start and end dates for a time window

        Args:
            window: Time window specification

        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        now = datetime.now()

        if window == "last 12 months":
            start = now.replace(year=now.year - 1)
        elif window == "last 6 months":
            month = now.month - 6
            year = now.year
            if month <= 0:
                month += 12
                year -= 1
            start = now.replace(year=year, month=month)
        elif window == "last month":
            if now.month == 1:
                start = now.replace(year=now.year - 1, month=12, day=1)
            else:
                start = now.replace(month=now.month - 1, day=1)
        elif window == "this month":
            start = now.replace(day=1)
        elif window == "last two weeks":
            start = now - pd.Timedelta(days=14)
        elif window == "last week":
            start = now - pd.Timedelta(days=7)
        elif window == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start = now.replace(year=now.year - 1)  # default to last year

        return start, now


class StatisticsHelper:
    """Statistical utility functions"""

    @staticmethod
    def calculate_percentiles(data: List[Union[int, float]],
                              percentiles: List[int] = [25, 50, 75, 90, 95]) -> Dict[int, float]:
        """
        Calculate percentiles for numerical data

        Args:
            data: List of numerical values
            percentiles: List of percentile values to calculate

        Returns:
            Dictionary mapping percentile to value
        """
        if not data:
            return {}

        import numpy as np
        return {p: np.percentile(data, p) for p in percentiles}

    @staticmethod
    def calculate_growth_rate(old_value: float, new_value: float) -> float:
        """
        Calculate the growth rate between two values

        Args:
            old_value: Original value
            new_value: New value

        Returns:
            Growth rate as percentage
        """
        if old_value == 0:
            return float('inf') if new_value > 0 else 0
        return ((new_value - old_value) / old_value) * 100

    @staticmethod
    def normalize_scores(scores: List[float], min_val: float = 0, max_val: float = 100) -> List[float]:
        """
        Normalize scores to a specified range

        Args:
            scores: List of scores to normalize
            min_val: Minimum value in output range
            max_val: Maximum value in output range

        Returns:
            List of normalized scores
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return [min_val] * len(scores)

        normalized = []
        for score in scores:
            norm_score = (score - min_score) / (max_score - min_score)
            scaled_score = min_val + norm_score * (max_val - min_val)
            normalized.append(scaled_score)

        return normalized


class TextProcessor:
    """Text processing utilities"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text for analysis

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        import re

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.lower()

    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from a text

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return

        Returns:
            List of keywords
        """
        if not text:
            return []

        from collections import Counter

        # Clean text and split into words
        cleaned_text = TextProcessor.clean_text(text)
        words = cleaned_text.split()

        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their'
        }

        # Filter words
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]

        # Count word frequency
        word_counts = Counter(filtered_words)

        # Return top keywords
        return [word for word, count in word_counts.most_common(max_keywords)]


def merge_analysis_results(*results: Dict) -> Dict:
    """
    Merge multiple analysis result dictionaries

    Args:
        *results: Variable number of result dictionaries

    Returns:
        Merged results dictionary
    """
    merged = {}
    for result in results:
        if isinstance(result, dict):
            merged.update(result)
    return merged


def format_large_number(number: Union[int, float]) -> str:
    """
    Format large numbers with appropriate suffixes

    Args:
        number: Number to format

    Returns:
        Formatted number string
    """
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number / 1_000:.1f}K"
    else:
        return str(int(number))


def create_summary_table(data: Dict, title: str = "Summary") -> str:
    """
    Create formatted summary table from dictionary

    Args:
        data: Dictionary of key-value pairs
        title: Table title

    Returns:
        Formatted table string
    """
    if not data:
        return f"{title}: No data available"

    lines = [f"\n{title}", "=" * len(title)]

    for key, value in data.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                value = round(value, 2)
            value = format_large_number(value)

        lines.append(f"{key.replace('_', ' ').title()}: {value}")

    return "\n".join(lines)