# GitHub Repository Analyzer

A comprehensive Python tool for analyzing GitHub repositories, identifying trends in programming languages, topics, and predicting future development patterns. This analyzer provides insights into the open-source ecosystem through data collection, statistical analysis, and interactive visualizations.

## Features

- **Comprehensive Repository Search**: Search GitHub repositories with flexible criteria including language, topics, star count, and time windows
- **Advanced Analytics**: Statistical analysis of programming language trends, topic clustering, and predictive modeling
- **Interactive Visualizations**: Generate professional charts, word clouds, and interactive dashboards
- **Time-Based Analysis**: Analyze trends across different time periods (last 12 months, 6 months, month, week, today)
- **Trend Prediction**: Forecast future technology trends based on current repository patterns
- **Professional Data Export**: Export results in multiple formats (CSV, JSON, visualizations)

## Installation

### Prerequisites
- Python 3.7 or higher
- GitHub Personal Access Token (for API access)

### Quick Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/github-repository-analyzer.git
cd github-repository-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. **Set up GitHub token:**
Create a `.env` file in the project root:
```
GITHUB_TOKEN=your_personal_access_token_here
```

## Configuration

### GitHub API Setup
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `public_repo` scope
3. Add the token to your environment variables or `.env` file

### Basic Configuration
The analyzer uses default settings that work for most use cases. You can modify configurations in `github_analyzer/config.py`:

- `DEFAULT_RESULTS_PER_PAGE`: Number of repositories per API request (default: 100)
- `MAX_REPOSITORIES`: Maximum repositories to analyze (default: 1000)
- `RATE_LIMIT_BUFFER`: Buffer time for rate limiting (default: 1 second)

## Usage

### Basic Usage

```python
import main
from main import GitHubRepositoryAnalyzer
from visualizer import VisualizationEngine

# Initialize the analyzer
analyzer = GitHubRepositoryAnalyzer(token="your_github_token")

# Search and analyze repositories
results = main.analyze_github_trends(
    query="language:python topic:machine-learning",
    time_window="last 6 months",
    max_repos=500
)

# Generate visualizations
VisualizationEngine.create_trend_prediction_chart()
VisualizationEngine.generate_wordcloud()
VisualizationEngine.create_comprehensive_dashboard()
```

### Advanced Analysis

```python
from analyzer import AnalysisEngine
# Analyze specific programming languages
language_trends = AnalysisEngine.analyze_language_trends(
    languages=["python", "javascript", "go", "rust"],
    time_window="last 12 months"
)

# Perform topic clustering
clusters = AnalysisEngine.perform_topic_clustering(
    n_clusters=10,
    min_repos_per_cluster=20
)

# Generate trend predictions
predictions = AnalysisEngine.predict_future_trends(
    forecast_months=12,
    confidence_level=0.95
)
```

### Time Window Analysis

```python
from analyzer import AnalysisEngine
# Available time windows
time_windows = [
    "last 12 months", "last 6 months", "last month", 
    "this month", "last two weeks", "last week", "today"
]

# Compare trends across different time periods
for window in time_windows:
    trends = AnalysisEngine.predict_trends(time_window=window)
    print(f"Top languages in {window}: {trends['top_languages']}")
```

## Analysis Features

### Language Trend Analysis
- Repository count by programming language
- Weighted popularity scores
- Growth rate calculations
- Quality metrics (stars per repository)

### Topic Analysis
- Topic frequency analysis
- Co-occurrence patterns
- Emerging topic identification
- Semantic clustering

### Predictive Analytics
- Time series forecasting
- Trend extrapolation
- Popularity predictions
- Technology adoption modeling

## Visualization Options

The analyzer generates various visualization types:

1. **Language Popularity Charts**: Bar charts and line graphs showing language trends
2. **Topic Word Clouds**: Visual representation of popular topics and keywords
3. **Scatter Plots**: Correlation analysis between different metrics
4. **Time Series Plots**: Trend analysis over time periods
5. **Interactive Dashboards**: Comprehensive analysis dashboards

## API Rate Limiting

The analyzer automatically handles GitHub API rate limiting:
- Intelligent request pacing
- Automatic retry with exponential backoff
- Rate limit monitoring and alerts
- Optimal request batching

**Authenticated requests**: 5,000 requests per hour
**Unauthenticated requests**: 60 requests per hour

## Example Outputs

### Language Analysis Results
```python
{
    "python": {
        "repository_count": 1247,
        "total_stars": 89432,
        "average_stars": 71.7,
        "growth_rate": 0.15
    },
    "javascript": {
        "repository_count": 892,
        "total_stars": 45231,
        "average_stars": 50.7,
        "growth_rate": 0.08
    }
}
```

### Topic Clustering Results
```python
{
    "cluster_0": {
        "primary_topics": ["machine-learning", "artificial-intelligence", "neural-networks"],
        "repository_count": 234,
        "dominant_language": "python"
    },
    "cluster_1": {
        "primary_topics": ["web-development", "frontend", "react"],
        "repository_count": 189,
        "dominant_language": "javascript"
    }
}
```

## Data Export

Export analysis results in multiple formats:

```python
from analyzer import AnalysisEngine
# Export to CSV
AnalysisEngine.export_to_csv("analysis_results.csv")

# Export to JSON
AnalysisEngine.export_to_json("analysis_results.json")

# Export visualizations
AnalysisEngine.save_all_charts("charts/")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Future Enhancements

- Real-time streaming analysis
- Integration with GitLab and other platforms
- Advanced machine learning models
- Web dashboard interface
- API endpoint for remote access
