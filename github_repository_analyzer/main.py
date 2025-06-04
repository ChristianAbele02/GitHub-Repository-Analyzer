"""
Main GitHub Repository Analyzer class that orchestrates all components
"""

from typing import List, Dict, Optional, Any

from analyzer import AnalysisEngine
from data_collector import GitHubAPIClient
from utils import DataValidator, FileManager, create_summary_table
from visualizer import VisualizationEngine
from github_repository_analyzer.config import DEFAULT_FILENAME_PREFIX


class GitHubRepositoryAnalyzer:
    """
    Main class that orchestrates GitHub repository analysis workflow
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub Repository Analyzer
        
        Args:
            token: GitHub personal access token for higher rate limits
        """
        self.api_client = GitHubAPIClient(token)
        self.repositories_data = []
        self.analysis_engine = None
        self.visualization_engine = None
        
        # Analysis results cache
        self._language_stats = None
        self._topic_stats = None
        self._clustering_results = None
        self._trend_predictions = None

    def search_repositories(self, query: str, sort: str = 'stars', order: str = 'desc',
                          max_repos: int = 1000, time_window: Optional[str] = None) -> List[Dict]:
        """
        Search for GitHub repositories
        
        Args:
            query: Search query string
            sort: Sort field ('stars', 'forks', 'updated')
            order: Sort order ('asc', 'desc')
            max_repos: Maximum repositories to collect
            time_window: Time window for filtering (e.g., "last 6 months")
            
        Returns:
            List of repository data
        """
        print(f"Searching for repositories with query: {query}")
        if time_window:
            print(f"Time window: {time_window}")
        
        repositories = self.api_client.search_repositories(
            query=query,
            sort=sort,
            order=order,
            max_repos=max_repos,
            time_window=time_window
        )
        
        # Clean and validate data
        repositories = DataValidator.clean_repository_data(repositories)
        
        # Add to existing data
        self.repositories_data.extend(repositories)
        
        # Reset cached analysis results
        self._reset_cache()
        
        print(f"Total repositories collected: {len(self.repositories_data)}")
        return repositories

    def analyze_all(self, perform_clustering: bool = True, 
                   predict_trends: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on collected repository data
        
        Args:
            perform_clustering: Whether to perform topic clustering
            predict_trends: Whether to generate trend predictions
            
        Returns:
            Dictionary with all analysis results
        """
        if not self.repositories_data:
            raise ValueError("No repository data available. Run search_repositories() first.")
        
        print("Starting comprehensive analysis...")
        
        # Initialize analysis engine
        self.analysis_engine = AnalysisEngine(self.repositories_data)
        
        # Perform core analyses
        print("Analyzing programming languages...")
        self._language_stats = self.analysis_engine.analyze_programming_languages()
        
        print("Analyzing repository topics...")
        self._topic_stats = self.analysis_engine.get_repository_topics()
        
        # Optional advanced analysis
        if perform_clustering and len(self.repositories_data) >= 10:
            print("Performing topic clustering...")
            self._clustering_results = self.analysis_engine.perform_topic_clustering()
        
        if predict_trends and not self._language_stats.empty:
            print("Generating trend predictions...")
            self._trend_predictions = self.analysis_engine.predict_trends(self._language_stats)
        
        # Generate summary statistics
        summary_stats = self.analysis_engine.generate_summary_statistics()
        
        print("Analysis complete!")
        
        return {
            'summary': summary_stats,
            'languages': self._language_stats.to_dict('records') if not self._language_stats.empty else [],
            'topics': self._topic_stats.to_dict('records') if not self._topic_stats.empty else [],
            'clustering': self._clustering_results,
            'predictions': self._trend_predictions or {},
            'correlation_matrix': self.analysis_engine.calculate_correlation_matrix().to_dict() if hasattr(self.analysis_engine, 'calculate_correlation_matrix') else {}
        }

    def create_visualizations(self, save_charts: bool = True, 
                            output_dir: str = "charts") -> Dict[str, Any]:
        """
        Create comprehensive visualizations
        
        Args:
            save_charts: Whether to save charts to files
            output_dir: Directory to save charts
            
        Returns:
            Dictionary of visualization objects
        """
        if not self.repositories_data:
            raise ValueError("No repository data available. Run search_repositories() first.")
        
        print("Creating visualizations...")
        
        # Initialize visualization engine
        self.visualization_engine = VisualizationEngine(self.repositories_data)
        
        # Ensure we have analysis results
        if self._language_stats is None or self._topic_stats is None:
            print("Running analysis first...")
            self.analyze_all()
        
        # Create comprehensive dashboard
        dashboard = self.visualization_engine.create_comprehensive_dashboard(
            language_stats=self._language_stats,
            topic_stats=self._topic_stats,
            predictions=self._trend_predictions or {},
            clustering_results=self._clustering_results
        )
        
        # Generate word cloud
        wordcloud = self.visualization_engine.generate_wordcloud()
        
        # Save charts if requested
        if save_charts:
            if FileManager.ensure_directory_exists(output_dir):
                self.visualization_engine.save_charts(dashboard, output_dir)
                print(f"Charts saved to {output_dir}/")
        
        print("Visualizations complete!")
        
        return {
            'dashboard': dashboard,
            'wordcloud': wordcloud
        }

    def generate_insights_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive insights report
        
        Returns:
            Dictionary with insights and recommendations
        """
        if not self.repositories_data:
            return {"error": "No repository data available"}
        
        # Ensure analysis is complete
        if self._language_stats is None:
            self.analyze_all()
        
        # Basic statistics
        total_repos = len(self.repositories_data)
        total_stars = sum(repo['stars'] for repo in self.repositories_data)
        total_forks = sum(repo['forks'] for repo in self.repositories_data)
        
        # Generate insights
        insights = {
            'executive_summary': {
                'total_repositories_analyzed': total_repos,
                'total_stars': total_stars,
                'total_forks': total_forks,
                'average_stars_per_repo': round(total_stars / total_repos, 1) if total_repos > 0 else 0,
                'unique_languages': len(self._language_stats) if not self._language_stats.empty else 0,
                'unique_topics': len(self._topic_stats) if not self._topic_stats.empty else 0
            },
            'top_findings': self._generate_top_findings(),
            'language_insights': self._generate_language_insights(),
            'topic_insights': self._generate_topic_insights(),
            'trend_insights': self._generate_trend_insights(),
            'recommendations': self._generate_recommendations()
        }
        
        return insights

    def _generate_top_findings(self) -> List[str]:
        """Generate top findings from analysis"""
        findings = []
        
        if not self._language_stats.empty:
            top_lang = self._language_stats.iloc[0]
            findings.append(f"{top_lang['language']} is the most popular language with {top_lang['count_stars']} repositories")
        
        if not self._topic_stats.empty:
            top_topic = self._topic_stats.iloc[0]
            findings.append(f"'{top_topic['topic']}' is the most common topic appearing in {top_topic['percentage']:.1f}% of repositories")
        
        if self._clustering_results:
            cluster_count = len(self._clustering_results['cluster_analysis'])
            findings.append(f"Repository clustering revealed {cluster_count} distinct technology clusters")
        
        if self._trend_predictions:
            growing_langs = [lang for lang, data in self._trend_predictions.items() 
                           if data['trend_direction'] == 'up']
            if growing_langs:
                findings.append(f"Languages showing upward trends: {', '.join(growing_langs[:3])}")
        
        return findings

    def _generate_language_insights(self) -> List[str]:
        """Generate language-specific insights"""
        if self._language_stats.empty:
            return ["No language data available for analysis"]
        
        insights = []
        
        # Top languages by different metrics
        top_by_count = self._language_stats.iloc[0]
        insights.append(f"Most prolific language: {top_by_count['language']} ({top_by_count['count_stars']} repositories)")
        
        if 'mean_stars' in self._language_stats.columns:
            top_by_quality = self._language_stats.loc[self._language_stats['mean_stars'].idxmax()]
            insights.append(f"Highest average quality: {top_by_quality['language']} ({top_by_quality['mean_stars']:.1f} stars/repo)")
        
        # Language diversity
        total_repos = sum(self._language_stats['count_stars'])
        if len(self._language_stats) > 1:
            top_3_share = sum(self._language_stats.head(3)['count_stars']) / total_repos * 100
            insights.append(f"Top 3 languages represent {top_3_share:.1f}% of all repositories")
        
        return insights

    def _generate_topic_insights(self) -> List[str]:
        """Generate topic-specific insights"""
        if self._topic_stats.empty:
            return ["No topic data available for analysis"]
        
        insights = []
        
        # Topic coverage
        total_repos_with_topics = sum(self._topic_stats['count'])
        coverage = total_repos_with_topics / len(self.repositories_data) * 100
        insights.append(f"Topic coverage: {coverage:.1f}% of repositories have topic tags")
        
        # Popular categories
        api_topics = self._topic_stats[self._topic_stats['topic'].str.contains('api|rest|graphql', case=False, na=False)]
        if not api_topics.empty:
            api_count = sum(api_topics['count'])
            insights.append(f"API-related topics appear in {api_count} repositories")
        
        web_topics = self._topic_stats[self._topic_stats['topic'].str.contains('web|frontend|backend|react|vue|angular', case=False, na=False)]
        if not web_topics.empty:
            web_count = sum(web_topics['count'])
            insights.append(f"Web development topics appear in {web_count} repositories")
        
        return insights

    def _generate_trend_insights(self) -> List[str]:
        """Generate trend prediction insights"""
        if not self._trend_predictions:
            return ["No trend predictions available"]
        
        insights = []
        
        # Growth trends
        growing = [(lang, data['growth_rate']) for lang, data in self._trend_predictions.items() 
                  if data['trend_direction'] == 'up']
        declining = [(lang, data['growth_rate']) for lang, data in self._trend_predictions.items() 
                    if data['trend_direction'] == 'down']
        
        if growing:
            growing.sort(key=lambda x: x[1], reverse=True)
            top_growing = growing[:3]
            insights.append(f"Fastest growing languages: {', '.join([f'{lang} (+{rate:.1f}%)' for lang, rate in top_growing])}")
        
        if declining:
            declining.sort(key=lambda x: x[1])
            most_declining = declining[:3]
            insights.append(f"Declining languages: {', '.join([f'{lang} ({rate:.1f}%)' for lang, rate in most_declining])}")
        
        return insights

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not self._language_stats.empty:
            top_lang = self._language_stats.iloc[0]['language']
            recommendations.append(f"Consider learning {top_lang} - it's the most active language in the ecosystem")
        
        if self._trend_predictions:
            growing_langs = [lang for lang, data in self._trend_predictions.items() 
                           if data['trend_direction'] == 'up' and data['growth_rate'] > 2]
            if growing_langs:
                recommendations.append(f"Emerging opportunities in: {', '.join(growing_langs[:3])}")
        
        if not self._topic_stats.empty:
            hot_topics = self._topic_stats.head(5)['topic'].tolist()
            recommendations.append(f"Focus on trending topics: {', '.join(hot_topics)}")
        
        if self._clustering_results:
            cluster_analysis = self._clustering_results['cluster_analysis']
            largest_cluster = max(cluster_analysis.values(), key=lambda x: x['size'])
            top_terms = largest_cluster['top_terms'][:3]
            recommendations.append(f"Major technology cluster centers around: {', '.join(top_terms)}")
        
        return recommendations

    def save_all_data(self, filename_prefix: str = DEFAULT_FILENAME_PREFIX) -> Dict[str, str]:
        """
        Save all collected data and analysis results
        
        Args:
            filename_prefix: Prefix for output files
            
        Returns:
            Dictionary of saved filenames
        """
        saved_files = {}
        
        # Save raw repository data
        if self.repositories_data:
            json_file = FileManager.create_timestamped_filename(filename_prefix, "repositories", "json")
            if FileManager.save_to_json(self.repositories_data, json_file):
                saved_files['repositories_json'] = json_file
            
            csv_file = FileManager.create_timestamped_filename(filename_prefix, "repositories", "csv")
            if FileManager.save_to_csv(self.repositories_data, csv_file):
                saved_files['repositories_csv'] = csv_file
        
        # Save analysis results
        insights = self.generate_insights_report()
        insights_file = FileManager.create_timestamped_filename(filename_prefix, "insights", "json")
        if FileManager.save_to_json(insights, insights_file):
            saved_files['insights'] = insights_file
        
        # Save individual analysis components
        if not self._language_stats.empty:
            lang_file = FileManager.create_timestamped_filename(filename_prefix, "languages", "csv")
            if self._language_stats.to_csv(lang_file, index=False):
                saved_files['languages'] = lang_file
        
        if not self._topic_stats.empty:
            topic_file = FileManager.create_timestamped_filename(filename_prefix, "topics", "csv")
            if self._topic_stats.to_csv(topic_file, index=False):
                saved_files['topics'] = topic_file
        
        print(f"Data saved to {len(saved_files)} files")
        return saved_files

    def _reset_cache(self):
        """Reset cached analysis results"""
        self._language_stats = None
        self._topic_stats = None
        self._clustering_results = None
        self._trend_predictions = None

    def print_summary(self):
        """Print analysis summary to the console"""
        if not self.repositories_data:
            print("No data to summarize. Run search_repositories() first.")
            return
        
        insights = self.generate_insights_report()
        
        # Print executive summary
        print(create_summary_table(insights['executive_summary'], "Executive Summary"))
        
        # Print top findings
        print(f"\nTop Findings:")
        for i, finding in enumerate(insights['top_findings'], 1):
            print(f"{i}. {finding}")
        
        # Print recommendations
        print(f"\nRecommendations:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"{i}. {rec}")


# Example usage function
def analyze_github_trends(queries: List[str], token: Optional[str] = None, 
                         time_window: Optional[str] = None, max_repos_per_query: int = 500,
                          fast_mode: bool = False) -> GitHubRepositoryAnalyzer:
    """
    Convenience function to analyze GitHub trends for multiple queries
    
    Args:
        queries: List of search queries
        token: GitHub API token
        time_window: Time window for analysis
        max_repos_per_query: Maximum repositories per query
        
    Returns:
        Configured GitHubRepositoryAnalyzer instance
    """
    analyzer = GitHubRepositoryAnalyzer(token)
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        analyzer.search_repositories(
            query=query,
            max_repos=max_repos_per_query,
            time_window=time_window
        )
    
    # Perform complete analysis
    analyzer.analyze_all()
    analyzer.create_visualizations()
    
    # Print summary
    analyzer.print_summary()
    
    # Save all data
    analyzer.save_all_data()
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    analyzer = analyze_github_trends(
        queries=["language:python topic:machine-learning", "topic:web-development"],
        time_window="last 6 months",
        max_repos_per_query=300
    )