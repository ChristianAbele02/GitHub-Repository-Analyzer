# github_repository_analyzer/analyzer.py
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from github_repository_analyzer.config import (
    DEFAULT_N_CLUSTERS, FORECAST_PERIODS, MIN_CLUSTER_SIZE,
    TF_IDF_MAX_FEATURES, TF_IDF_MIN_DF, TF_IDF_MAX_DF
)

class AnalysisEngine:
    def __init__(self, repositories_data: List[Dict]):
        self.repositories_data = repositories_data
        self.df = pd.DataFrame(repositories_data) if repositories_data else pd.DataFrame()

    def get_repository_topics(self) -> pd.DataFrame:
        if not self.repositories_data:
            return pd.DataFrame()
        all_topics = []
        for repo in self.repositories_data:
            all_topics.extend(repo.get('topics', []))
        if not all_topics:
            return pd.DataFrame()
        topic_counts = Counter(all_topics)
        total_repos = len(self.repositories_data)
        topic_df = pd.DataFrame([
            {
                'topic': topic,
                'count': count,
                'percentage': round(count / total_repos * 100, 2)
            }
            for topic, count in topic_counts.most_common()
        ])
        return topic_df

    @staticmethod
    def normalize_all_datetimes(df):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if hasattr(df[col].dt, 'tz_localize'):
                    df[col] = df[col].dt.tz_localize(None)
            elif df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if hasattr(df[col].dt, 'tz_localize'):
                        df[col] = df[col].dt.tz_localize(None)
                except Exception:
                    pass
        return df

    @staticmethod
    def assert_no_tz_aware(df):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if getattr(df[col].dt, 'tz', None) is not None:
                    raise ValueError(f"Column {col} is still tz-aware!")

    def analyze_programming_languages(self) -> pd.DataFrame:
        if self.df.empty:
            return pd.DataFrame()
        language_data = self.df[self.df['language'].notna()].copy()
        if language_data.empty:
            return pd.DataFrame()
        language_data['created_at'] = pd.to_datetime(language_data['created_at'])
        language_data['updated_at'] = pd.to_datetime(language_data['updated_at'])
        language_stats = language_data.groupby('language').agg({
            'stars': ['count', 'sum', 'mean', 'median'],
            'forks': ['sum', 'mean', 'median'],
            'watchers': ['sum', 'mean'],
            'size': ['mean', 'median'],
            'created_at': ['min', 'max'],
            'updated_at': ['min', 'max']
        }).round(2)
        language_stats.columns = [
            f"{col[1]}_{col[0]}" if col[1] else col[0]
            for col in language_stats.columns
        ]
        language_stats = language_stats.reset_index()
        language_stats = self.normalize_all_datetimes(language_stats)
        self.assert_no_tz_aware(language_stats)
        now = pd.Timestamp.now()
        max_repos = language_stats['count_stars'].max()
        max_total_stars = language_stats['sum_stars'].max()
        max_avg_stars = language_stats['mean_stars'].max()
        language_stats['popularity_score'] = (
                (language_stats['count_stars'] / max_repos) * 40 +
                (language_stats['sum_stars'] / max_total_stars) * 30 +
                (language_stats['mean_stars'] / max_avg_stars) * 20 +
                (language_stats['sum_forks'] / language_stats['sum_forks'].max()) * 10
        ).round(2)
        language_stats['avg_repo_age_days'] = (
            (now - language_stats['min_created_at']).dt.days
        )
        language_stats['recent_activity_score'] = (
                (language_stats['max_updated_at'] - language_stats['min_updated_at']).dt.days /
                language_stats['avg_repo_age_days']
        ).fillna(0).round(2)
        return language_stats.sort_values('popularity_score', ascending=False)

    # ... rest of the class unchanged ...
    def perform_topic_clustering(self, n_clusters: int = DEFAULT_N_CLUSTERS) -> Optional[Dict]:
        """
        Perform K-means clustering on repository topics and descriptions

        Args:
            n_clusters: Number of clusters for K-means

        Returns:
            Dictionary with clustering results and analysis
        """
        if not self.repositories_data:
            return None

        # Prepare text data for clustering
        text_data = []
        valid_repos = []

        for repo in self.repositories_data:
            topics_text = ' '.join(repo.get('topics', []))
            description = repo.get('description', '') or ''
            combined_text = f"{topics_text} {description}".strip()

            if combined_text:  # Only include non-empty texts
                text_data.append(combined_text)
                valid_repos.append(repo)

        if len(text_data) < n_clusters or len(text_data) < MIN_CLUSTER_SIZE:
            print(f"Insufficient data for clustering. Found {len(text_data)} valid repositories")
            return None

        try:
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(
                max_features=TF_IDF_MAX_FEATURES,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=TF_IDF_MIN_DF,
                max_df=TF_IDF_MAX_DF,
                lowercase=True,
                strip_accents='unicode'
            )

            tfidf_matrix = vectorizer.fit_transform(text_data)

            # K-means clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            cluster_labels = kmeans.fit_predict(tfidf_matrix)

            # Analyze clusters
            feature_names = vectorizer.get_feature_names_out()
            cluster_analysis = {}

            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_repos = [valid_repos[i] for i in cluster_indices]

                if not cluster_repos:
                    continue

                # Get top terms for this cluster
                cluster_center = kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-10:][::-1]
                top_terms = [feature_names[i] for i in top_indices]

                # Calculate cluster statistics
                cluster_languages = Counter([
                    repo['language'] for repo in cluster_repos
                    if repo.get('language')
                ])

                cluster_topics = []
                for repo in cluster_repos:
                    cluster_topics.extend(repo.get('topics', []))
                common_topics = Counter(cluster_topics).most_common(5)

                cluster_stats = {
                    'size': len(cluster_repos),
                    'percentage': round(len(cluster_repos) / len(valid_repos) * 100, 1),
                    'top_terms': top_terms,
                    'avg_stars': round(np.mean([repo['stars'] for repo in cluster_repos]), 1),
                    'avg_forks': round(np.mean([repo['forks'] for repo in cluster_repos]), 1),
                    'top_languages': dict(cluster_languages.most_common(3)),
                    'common_topics': [topic for topic, count in common_topics],
                    'sample_repositories': [
                        {
                            'name': repo['full_name'],
                            'stars': repo['stars'],
                            'language': repo.get('language')
                        }
                        for repo in sorted(cluster_repos, key=lambda x: x['stars'], reverse=True)[:3]
                    ]
                }

                cluster_analysis[f'cluster_{cluster_id}'] = cluster_stats

            return {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_analysis': cluster_analysis,
                'vectorizer_vocab_size': len(feature_names),
                'silhouette_score': self._calculate_silhouette_score(tfidf_matrix, cluster_labels),
                'total_repositories_clustered': len(valid_repos)
            }

        except Exception as e:
            print(f"Error in clustering: {e}")
            return None

    def _calculate_silhouette_score(self, tfidf_matrix, cluster_labels) -> float:
        """Calculate silhouette score for clustering quality assessment"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(cluster_labels)) > 1:
                return round(silhouette_score(tfidf_matrix, cluster_labels), 3)
        except:
            pass
        return 0.0

    def predict_trends(self, language_df: pd.DataFrame,
                       forecast_periods: int = FORECAST_PERIODS) -> Dict:
        """
        Predict future trends using statistical analysis

        Args:
            language_df: Language analysis DataFrame
            forecast_periods: Number of periods to forecast

        Returns:
            Dictionary with trend predictions
        """
        if language_df.empty:
            return {}

        predictions = {}
        top_languages = language_df.head(15)  # Focus on the top 15 languages

        for _, lang_data in top_languages.iterrows():
            language = lang_data['language']
            current_popularity = lang_data['popularity_score']
            repo_count = lang_data['count_stars']
            avg_stars = lang_data['mean_stars']
            recent_activity = lang_data.get('recent_activity_score', 1.0)

            # Calculate trend factors based on multiple indicators
            size_factor = min(1.1, 1 + (repo_count / 1000) * 0.01)  # Popularity boost
            quality_factor = min(1.05, 1 + (avg_stars / 1000) * 0.01)  # Quality boost
            activity_factor = min(1.05, max(0.95, recent_activity))  # Activity influence

            # Combined trend factor with some randomness for realism
            np.random.seed(hash(language) % 2 ** 32)  # Deterministic randomness per language
            base_trend = size_factor * quality_factor * activity_factor
            noise_factor = np.random.uniform(0.98, 1.02)
            trend_factor = base_trend * noise_factor

            # Generate forecast
            forecast = []
            current_value = current_popularity

            for i in range(forecast_periods):
                # Add seasonal variation and gradual decay
                seasonal_component = 1 + 0.05 * np.sin(2 * np.pi * i / 12)
                decay_factor = 0.999 ** i  # Slight decay over time

                predicted_value = current_value * (trend_factor ** (i + 1)) * seasonal_component * decay_factor
                forecast.append(round(predicted_value, 2))

            # Determine a trend direction and calculate the growth rate
            final_value = forecast[-1]
            growth_rate = ((final_value / current_popularity) ** (1 / forecast_periods) - 1) * 100

            predictions[language] = {
                'current_score': round(current_popularity, 2),
                'forecast': forecast,
                'trend_direction': 'up' if growth_rate > 0 else 'down',
                'growth_rate': round(growth_rate, 2),
                'confidence': min(100, max(50, 100 - abs(growth_rate) * 2)),  # Lower confidence for extreme predictions
                'factors': {
                    'repository_count': repo_count,
                    'average_quality': round(avg_stars, 1),
                    'activity_score': round(recent_activity, 2)
                }
            }

        return predictions

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between numerical repository features

        Returns:
            Correlation matrix DataFrame
        """
        if self.df.empty:
            return pd.DataFrame()

        numerical_columns = ['stars', 'forks', 'watchers', 'issues', 'size']
        available_columns = [col for col in numerical_columns if col in self.df.columns]

        if len(available_columns) < 2:
            return pd.DataFrame()

        correlation_matrix = self.df[available_columns].corr().round(3)
        return correlation_matrix

    def generate_summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics

        Returns:
            Dictionary with summary statistics
        """
        if not self.repositories_data:
            return {}

        total_repos = len(self.repositories_data)

        # Basic aggregations
        total_stars = sum(repo['stars'] for repo in self.repositories_data)
        total_forks = sum(repo['forks'] for repo in self.repositories_data)
        total_watchers = sum(repo['watchers'] for repo in self.repositories_data)

        # Language diversity
        languages = [repo['language'] for repo in self.repositories_data if repo.get('language')]
        unique_languages = len(set(languages))

        # Topic diversity
        all_topics = []
        for repo in self.repositories_data:
            all_topics.extend(repo.get('topics', []))
        unique_topics = len(set(all_topics))

        # Date analysis
        if not self.df.empty and 'created_at' in self.df.columns:
            self.df['created_at'] = pd.to_datetime(self.df['created_at'])
            oldest_repo = self.df['created_at'].min()
            newest_repo = self.df['created_at'].max()
        else:
            oldest_repo = newest_repo = None

        return {
            'total_repositories': total_repos,
            'total_stars': total_stars,
            'total_forks': total_forks,
            'total_watchers': total_watchers,
            'avg_stars_per_repo': round(total_stars / total_repos, 1) if total_repos > 0 else 0,
            'avg_forks_per_repo': round(total_forks / total_repos, 1) if total_repos > 0 else 0,
            'unique_languages': unique_languages,
            'unique_topics': unique_topics,
            'oldest_repository': oldest_repo.strftime('%Y-%m-%d') if oldest_repo else None,
            'newest_repository': newest_repo.strftime('%Y-%m-%d') if newest_repo else None,
            'language_diversity_ratio': round(unique_languages / total_repos, 3) if total_repos > 0 else 0
        }