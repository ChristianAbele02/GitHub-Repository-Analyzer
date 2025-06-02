# GitHub Repository Analyzer Configuration Template
#
# This file contains all configurable settings for the GitHub Repository Analyzer.
# Copy this file to config.py in your github_analyzer/ directory and modify as needed.

"""
Configuration settings for GitHub Repository Analyzer

This module contains all configurable parameters for the analyzer including
API settings, analysis parameters, visualization options, and export settings.
"""

import os
from typing import Dict, List, Tuple

# =====================================
# API Configuration
# =====================================

# GitHub API settings
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_SEARCH_ENDPOINT = "/search/repositories"
GITHUB_REPO_ENDPOINT = "/repos"
DEFAULT_USER_AGENT = "GitHub-Repository-Analyzer"

# Rate Limiting
DEFAULT_RATE_LIMIT = 5000
RATE_LIMIT_BUFFER = 10
RETRY_WAIT_TIME = 60
DEFAULT_RESULTS_PER_PAGE = 100  # Max: 100 for GitHub API
MAX_REPOSITORIES = 1000         # Maximum repositories to analyze per query
RATE_LIMIT_BUFFER = 1          # Seconds to wait between requests
MAX_RETRIES = 3                # Maximum retry attempts for failed requests
RETRY_BACKOFF_FACTOR = 2       # Exponential backoff multiplier
REQUEST_TIMEOUT = 30           # Request timeout in seconds

# Authentication
GITHUB_TOKEN_ENV_VAR = "GITHUB_TOKEN"  # Environment variable name for token
REQUIRE_AUTHENTICATION = False         # Set to True to require token

# =====================================
# Analysis Configuration
# =====================================

# Default analysis parameters
DEFAULT_TIME_WINDOW = "last 6 months"
DEFAULT_MAX_REPOS = 500
DEFAULT_MIN_STARS = 1
DEFAULT_LANGUAGE_FILTER = None
DEFAULT_PER_PAGE = 100
MAX_PER_PAGE = 100

# Time window mappings (in days)
TIME_WINDOWS: Dict[str, int] = {
    "today": 1,
    "last week": 7,
    "last two weeks": 14,
    "last month": 30,
    "this month": 30,      # Approximation
    "last 6 months": 180,
    "last 12 months": 365
}

# Clustering and ML parameters
DEFAULT_N_CLUSTERS = 10
MIN_REPOS_PER_CLUSTER = 5
MAX_CLUSTERS = 20
CLUSTERING_ALGORITHM = "kmeans"  # Options: kmeans, hierarchical
TF_IDF_MAX_FEATURES = 1000
TF_IDF_MIN_DF = 2
TF_IDF_MAX_DF = 0.95

# Statistical analysis parameters
CONFIDENCE_LEVEL = 0.95
FORECAST_PERIODS = 12  # months
MIN_DATA_POINTS_FOR_FORECAST = 10

# =====================================
# Data Processing Configuration
# =====================================

# Language analysis settings
EXCLUDE_LANGUAGES: List[str] = [
    "HTML", "CSS", "Dockerfile", "Makefile",
    "Shell", "Batchfile", "PowerShell"
]
LANGUAGE_POPULARITY_WEIGHTS = {
    "stars": 0.4,
    "forks": 0.3,
    "count": 0.2,
    "recent_activity": 0.1
}

# Topic analysis settings
MIN_TOPIC_FREQUENCY = 2
MAX_TOPICS_DISPLAY = 50
EXCLUDE_TOPICS: List[str] = [
    "github", "repository", "project", "code",
    "programming", "software", "development"
]

# Repository filtering
MIN_REPO_SIZE_KB = 1        # Minimum repository size
MAX_REPO_AGE_DAYS = 365 * 5  # Maximum age in days
EXCLUDE_FORKS = False       # Whether to exclude forked repositories
EXCLUDE_ARCHIVED = True     # Whether to exclude archived repositories

# =====================================
# Visualization Configuration
# =====================================

# General plotting settings
FIGURE_SIZE: Tuple[int, int] = (12, 8)
DPI = 300
FONT_SIZE = 12
TITLE_FONT_SIZE = 16
SAVE_FORMAT = "png"
SAVE_QUALITY = 95

WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_MAX_WORDS = 100
WORDCLOUD_COLORMAP = "viridis"

# Color schemes
COLOR_PALETTE = "Set2"      # Seaborn color palette
CUSTOM_COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#34495e", "#e67e22"
]

# Chart-specific settings
BAR_CHART_SETTINGS = {
    "orientation": "horizontal",
    "show_values": True,
    "max_bars": 15
}

SCATTER_PLOT_SETTINGS = {
    "alpha": 0.7,
    "size_factor": 50,
    "show_trend_line": True
}

WORD_CLOUD_SETTINGS = {
    "width": 800,
    "height": 400,
    "max_words": 100,
    "background_color": "white",
    "colormap": "viridis"
}

TIME_SERIES_SETTINGS = {
    "show_confidence_interval": True,
    "smooth_data": True,
    "show_markers": True
}

# =====================================
# Export Configuration
# =====================================
# File Output
DEFAULT_FILENAME_PREFIX = "github_analysis"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# File export settings
DEFAULT_EXPORT_FORMAT = "csv"
SUPPORTED_EXPORT_FORMATS = ["csv", "json", "xlsx", "parquet"]
MAX_EXPORT_ROWS = 50000
DEFAULT_EXPORT_DIR = "exports"

# CSV export settings
CSV_SETTINGS = {
    "encoding": "utf-8",
    "index": False,
    "separator": ","
}

# JSON export settings
JSON_SETTINGS = {
    "indent": 2,
    "ensure_ascii": False,
    "sort_keys": True
}

# Excel export settings
EXCEL_SETTINGS = {
    "index": False,
    "engine": "openpyxl"
}

# =====================================
# Logging and Debug Configuration
# =====================================

# Logging settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_TO_FILE = False
LOG_FILE_PATH = "github_analyzer.log"
MAX_LOG_FILE_SIZE_MB = 10

# Debug settings
DEBUG_MODE = False
VERBOSE_OUTPUT = True
SHOW_PROGRESS_BARS = True
CACHE_API_RESPONSES = False
CACHE_DIRECTORY = ".cache"

# =====================================
# Performance Configuration
# =====================================

# Memory management
MAX_MEMORY_USAGE_MB = 1024  # Maximum memory usage
BATCH_SIZE = 100            # Batch size for processing large datasets
ENABLE_MULTIPROCESSING = True
MAX_WORKERS = 4             # Number of worker processes

# Caching settings
ENABLE_CACHING = True
CACHE_EXPIRY_HOURS = 24
CACHE_SIZE_LIMIT_MB = 500

# =====================================
# Advanced Features Configuration
# =====================================

# Machine learning model settings
MODEL_SETTINGS = {
    "language_trend_model": "linear_regression",
    "popularity_prediction_model": "random_forest",
    "topic_clustering_model": "kmeans",
    "sentiment_analysis_model": "vader"
}

# API enhancement settings
ENABLE_REPOSITORY_DETAILS = True  # Fetch detailed repo information
ENABLE_CONTRIBUTOR_ANALYSIS = False  # Analyze contributor patterns
ENABLE_COMMIT_ANALYSIS = False    # Analyze commit patterns
ENABLE_ISSUE_ANALYSIS = False     # Analyze issue patterns

# Experimental features
EXPERIMENTAL_FEATURES = {
    "real_time_monitoring": False,
    "trend_prediction": True,
    "social_network_analysis": False,
    "code_quality_metrics": False
}

# =====================================
# Environment-Specific Overrides
# =====================================

# Override settings based on environment
ENVIRONMENT = os.getenv("GITHUB_ANALYZER_ENV", "development")

if ENVIRONMENT == "production":
    # Production settings
    DEBUG_MODE = False
    LOG_LEVEL = "WARNING"
    CACHE_API_RESPONSES = True
    MAX_REPOSITORIES = 5000

elif ENVIRONMENT == "testing":
    # Testing settings
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"
    MAX_REPOSITORIES = 50
    RATE_LIMIT_BUFFER = 0.1

elif ENVIRONMENT == "development":
    # Development settings
    DEBUG_MODE = True
    LOG_LEVEL = "DEBUG"
    VERBOSE_OUTPUT = True
    MAX_REPOSITORIES = 200

# =====================================
# Validation Functions
# =====================================

def validate_config():
    """Validate configuration settings and return any issues."""
    issues = []

    # Validate time windows
    for window, days in TIME_WINDOWS.items():
        if days <= 0:
            issues.append(f"Invalid time window '{window}': {days} days")

    # Validate numeric settings
    if DEFAULT_RESULTS_PER_PAGE > 100:
        issues.append("DEFAULT_RESULTS_PER_PAGE cannot exceed 100 (GitHub API limit)")

    if MAX_REPOSITORIES <= 0:
        issues.append("MAX_REPOSITORIES must be positive")

    # Validate figure size
    if len(FIGURE_SIZE) != 2 or any(x <= 0 for x in FIGURE_SIZE):
        issues.append("FIGURE_SIZE must be (width, height) with positive values")

    return issues

def get_github_token():
    """Get GitHub token from environment variable or config."""
    token = os.getenv(GITHUB_TOKEN_ENV_VAR)

    if not token and REQUIRE_AUTHENTICATION:
        raise ValueError(f"GitHub token required. Set {GITHUB_TOKEN_ENV_VAR} environment variable.")

    return token

# =====================================
# Configuration Summary
# =====================================

def print_config_summary():
    """Print a summary of current configuration."""
    print("GitHub Repository Analyzer Configuration Summary")
    print("=" * 50)
    print(f"Environment: {ENVIRONMENT}")
    print(f"Max repositories: {MAX_REPOSITORIES}")
    print(f"Default time window: {DEFAULT_TIME_WINDOW}")
    print(f"Rate limit buffer: {RATE_LIMIT_BUFFER}s")
    print(f"Debug mode: {DEBUG_MODE}")
    print(f"Caching enabled: {ENABLE_CACHING}")
    print(f"Multiprocessing: {ENABLE_MULTIPROCESSING}")
    print("=" * 50)

# Validate configuration on import
if __name__ == "__main__":
    issues = validate_config()
    if issues:
        print("Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Configuration validation passed")

    print_config_summary()