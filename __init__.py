"""
GitHub Repository Analyzer Package

A comprehensive Python package for analyzing GitHub repository trends,
programming language popularity, and technology adoption patterns.

Main Components:
- GitHubRepositoryAnalyzer: Main orchestrating class
- GitHubAPIClient: API interaction and data collection
- AnalysisEngine: Data analysis and trend prediction
- VisualizationEngine: Chart generation and dashboards

Example Usage:
    from github_analyzer import GitHubRepositoryAnalyzer

    Analyzer = GitHubRepositoryAnalyzer(token="your_github_token")
    analyzer.search_repositories("language:python topic:machine-learning")
    results = analyzer.analyze_all()
    visualizations = analyzer.create_visualizations()
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

from .analyzer import AnalysisEngine
from .data_collector import GitHubAPIClient
# Main exports
from .main import GitHubRepositoryAnalyzer, analyze_github_trends
from .utils import DataValidator, FileManager, DateTimeHelper, TextProcessor
from .visualizer import VisualizationEngine

# Configuration

__all__ = [
    # Main classes
    'GitHubRepositoryAnalyzer',
    'GitHubAPIClient',
    'AnalysisEngine',
    'VisualizationEngine',

    # Utility classes
    'DataValidator',
    'FileManager',
    'DateTimeHelper',
    'TextProcessor',

    # Convenience functions
    'analyze_github_trends',

    # Package info
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]

# Package-level configuration
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Validate dependencies on import
def _check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        'pandas', 'numpy', 'requests', 'matplotlib',
        'seaborn', 'plotly', 'scikit-learn', 'wordcloud'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Warning: Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))


# Check dependencies when a package is imported
_check_dependencies()