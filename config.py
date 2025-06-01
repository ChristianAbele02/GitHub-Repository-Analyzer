"""
Configuration settings for GitHub Repository Analyzer
"""

# GitHub API Configuration
GITHUB_API_BASE_URL = "https://api.github.com"
DEFAULT_USER_AGENT = "GitHub-Repository-Analyzer"

# Rate Limiting
DEFAULT_RATE_LIMIT = 5000
RATE_LIMIT_BUFFER = 10
RETRY_WAIT_TIME = 60

# Search Parameters
DEFAULT_PER_PAGE = 100
MAX_PER_PAGE = 100
DEFAULT_SORT = "stars"
DEFAULT_ORDER = "desc"

# Analysis Parameters
DEFAULT_CLUSTERS = 10
DEFAULT_FORECAST_PERIODS = 12
MIN_CLUSTER_SIZE = 5

# Visualization Parameters
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_MAX_WORDS = 100
WORDCLOUD_COLORMAP = "viridis"

# File Output
DEFAULT_FILENAME_PREFIX = "github_analysis"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# TF-IDF Parameters
TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8
TFIDF_NGRAM_RANGE = (1, 2)