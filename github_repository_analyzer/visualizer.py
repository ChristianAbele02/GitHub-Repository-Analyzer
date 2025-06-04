"""
Visualization engine for charts, dashboards, and visual analytics
"""

from typing import List, Dict, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud

from config import (
    WORDCLOUD_WIDTH, WORDCLOUD_HEIGHT, WORDCLOUD_MAX_WORDS, WORDCLOUD_COLORMAP
)


class VisualizationEngine:
    """
    Handles chart generation, dashboards, and visual analytics
    """

    def __init__(self, repositories_data: List[Dict]):
        """
        Initialize visualization engine

        Args:
            repositories_data: List of repository dictionaries
        """
        self.repositories_data = repositories_data
        self.df = pd.DataFrame(repositories_data) if repositories_data else pd.DataFrame()

    def create_language_popularity_chart(self, language_stats: pd.DataFrame,
                                         top_n: int = 15) -> go.Figure:
        """
        Improved: Horizontal bar chart, value labels, color by mean_stars, sorted by popularity.
        """
        if language_stats.empty:
            return self._create_empty_chart("No language data available")
        df = language_stats.copy()
        df = df.sort_values('popularity_score', ascending=False).head(top_n)
        fig = px.bar(
            df,
            x='popularity_score',
            y='language',
            orientation='h',
            color='mean_stars',
            color_continuous_scale='viridis',
            text='count_stars',
            labels={
                'popularity_score': 'Popularity Score',
                'language': 'Programming Language',
                'mean_stars': 'Avg. Stars',
                'count_stars': 'Repo Count'
            },
            title=f'Top {top_n} Programming Languages by Popularity'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_colorbar=dict(title='Avg. Stars'),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),
        )
        return fig

    def create_topic_distribution_chart(self, topic_stats: pd.DataFrame,
                                        chart_type: str = 'bar', top_n: int = 20) -> go.Figure:
        """
        Improved: Horizontal bar chart, value and percentage labels, sorted by count.
        """
        if topic_stats.empty:
            return self._create_empty_chart("No topic data available")

        df = topic_stats.copy()
        df = df.sort_values('count', ascending=True).tail(top_n)
        fig = px.bar(
            df,
            x='count',
            y='topic',
            orientation='h',
            text='percentage',
            color='count',
            color_continuous_scale='plasma',
            labels={
                'count': 'Number of Repositories',
                'topic': 'Topic',
                'percentage': 'Percentage (%)'
            },
            title=f'Top {top_n} Repository Topics'
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgrey'),
            yaxis=dict(showgrid=False)
        )
        return fig

    def create_stars_vs_forks_scatter(self, color_by: str = 'language',
                                      log_scale: bool = True,
                                      max_points: int = 1000) -> go.Figure:
        """
        Create a scatter plot of stars vs. forks with improved performance

        Args:
            color_by: Column to color points by
            log_scale: Whether to use logarithmic scale
            max_points: Maximum number of points to plot

        Returns:
            Plotly figure object
        """
        if self.df.empty:
            return self._create_empty_chart("No repository data available")

        # Filter without creating a full copy
        if log_scale:
            plot_df = self.df.loc[(self.df['stars'] > 0) & (self.df['forks'] > 0)]
        else:
            plot_df = self.df

        if plot_df.empty:
            return self._create_empty_chart("No valid data for scatter plot")

        # Sample data if too large
        original_count = len(plot_df)
        if len(plot_df) > max_points:
            plot_df = plot_df.sample(max_points, random_state=42)

        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='stars',
            y='forks',
            color=color_by,
            hover_data=['name', 'language', 'stars', 'forks'],
            title=f'Repository Stars vs Forks (showing {len(plot_df)} of {original_count} repos)',
            labels={'stars': 'Stars', 'forks': 'Forks'},
            log_x=log_scale,
            log_y=log_scale
        )

        # Add proper trend line using numpy
        if len(plot_df) > 1:
            import numpy as np

            # For log scale, use log values for calculation
            x_vals = np.log10(plot_df['stars']) if log_scale else plot_df['stars'].values
            y_vals = np.log10(plot_df['forks']) if log_scale else plot_df['forks'].values

            # Only use finite values
            valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            if sum(valid_mask) > 1:  # Need at least 2 points for regression
                x_valid = x_vals[valid_mask]
                y_valid = y_vals[valid_mask]

                # Linear regression
                z = np.polyfit(x_valid, y_valid, 1)
                slope, intercept = z

                # Generate x points for the line
                x_range = np.linspace(min(x_valid), max(x_valid), 100)
                y_range = slope * x_range + intercept

                # Convert back from log if needed
                if log_scale:
                    x_range = 10 ** x_range
                    y_range = 10 ** y_range

                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', dash='dash'),
                        showlegend=True
                    )
                )

        return fig

    def create_trend_prediction_chart(self, predictions: Dict,
                                      languages: Optional[List[str]] = None) -> go.Figure:
        """
        Create a trend prediction chart

        Args:
            predictions: Trend predictions dictionary
            languages: Specific languages to display (None for all)

        Returns:
            Plotly figure object
        """
        if not predictions:
            return self._create_empty_chart("No prediction data available")

        fig = go.Figure()

        display_langs = languages if languages else list(predictions.keys())[:10]

        for lang in display_langs:
            if lang not in predictions:
                continue

            pred_data = predictions[lang]
            current_score = pred_data['current_score']
            forecast = pred_data['forecast']

            # Create x-axis (months)
            x_months = list(range(-1, len(forecast)))  # -1 for current, then forecast months
            y_values = [current_score] + forecast

            fig.add_trace(go.Scatter(
                x=x_months,
                y=y_values,
                mode='lines+markers',
                name=f"{lang} ({pred_data['trend_direction']})",
                line=dict(width=2),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title='Programming Language Trend Predictions',
                xaxis_title='Months from Now',
                yaxis_title='Popularity Score',
                hovermode='x unified'
            )

            # Add vertical line at x=0 to separate current from predictions
            fig.add_vline(x=0, line_dash="dash", line_color="red",
                          annotation_text="Current")

        return fig

    def create_cluster_visualization(self, clustering_results: Dict) -> go.Figure:
        """
        Create cluster analysis visualization

        Args:
            clustering_results: Results from topic clustering

        Returns:
            Plotly figure object
        """
        if not clustering_results or 'cluster_analysis' not in clustering_results:
            return self._create_empty_chart("No clustering data available")

        cluster_analysis = clustering_results['cluster_analysis']

        # Prepare data for visualization
        cluster_data = []
        for cluster_id, data in cluster_analysis.items():
            cluster_data.append({
                'cluster': cluster_id.replace('cluster_', 'Cluster '),
                'size': data['size'],
                'percentage': data['percentage'],
                'avg_stars': data['avg_stars'],
                'top_terms': ', '.join(data['top_terms'][:3])
            })

        cluster_df = pd.DataFrame(cluster_data)

        # Create a subplot with multiple charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Sizes', 'Average Stars per Cluster',
                            'Cluster Size Distribution', 'Cluster Percentage'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )

        # Cluster sizes bar chart
        fig.add_trace(
            go.Bar(x=cluster_df['cluster'], y=cluster_df['size'], name='Size'),
            row=1, col=1
        )

        # Average stars bar chart
        fig.add_trace(
            go.Bar(x=cluster_df['cluster'], y=cluster_df['avg_stars'], name='Avg Stars'),
            row=1, col=2
        )

        # Pie chart of cluster sizes
        fig.add_trace(
            go.Pie(labels=cluster_df['cluster'], values=cluster_df['size'], name='Distribution'),
            row=2, col=1
        )

        # Scatter plot of size vs avg stars
        fig.add_trace(
            go.Scatter(
                x=cluster_df['size'],
                y=cluster_df['avg_stars'],
                mode='markers+text',
                text=cluster_df['cluster'],
                textposition='top center',
                name='Size vs Quality'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text='Repository Clustering Analysis',
            showlegend=False,
            height=800
        )

        return fig

    def generate_wordcloud(self, topic_data: Optional[List[str]] = None) -> WordCloud:
        """
        Generate word cloud from repository topics or descriptions

        Args:
            topic_data: Custom topic data (None to use repository data)

        Returns:
            WordCloud object
        """
        if topic_data is None:
            # Extract topics and descriptions from repositories
            text_data = []
            for repo in self.repositories_data:
                text_data.extend(repo.get('topics', []))
                desc = repo.get('description', '')
                if desc:
                    text_data.append(desc)
        else:
            text_data = topic_data

        if not text_data:
            # Create a minimal wordcloud with placeholder text
            text_data = ['no', 'data', 'available']

        text = ' '.join(text_data)

        wordcloud = WordCloud(
            width=WORDCLOUD_WIDTH,
            height=WORDCLOUD_HEIGHT,
            background_color='white',
            max_words=WORDCLOUD_MAX_WORDS,
            colormap=WORDCLOUD_COLORMAP,
            collocations=False,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)

        return wordcloud

    def create_comprehensive_dashboard(self, language_stats: pd.DataFrame,
                                       topic_stats: pd.DataFrame,
                                       predictions: Dict,
                                       clustering_results: Optional[Dict] = None) -> Dict[str, go.Figure]:
        """
        Create a comprehensive dashboard with multiple visualizations (optimized for speed)
        """
        dashboard = {}

        # Limit data size for faster plotting
        lang_stats = language_stats.head(15) if not language_stats.empty else language_stats
        topic_stats = topic_stats.head(20) if not topic_stats.empty else topic_stats

        # Generate charts only if data is available
        if not lang_stats.empty:
            dashboard['language_popularity'] = self.create_language_popularity_chart(lang_stats, 15)
        if not topic_stats.empty:
            dashboard['topic_distribution'] = self.create_topic_distribution_chart(topic_stats, 'bar', 20)
        if not self.df.empty:
            dashboard['stars_vs_forks'] = self.create_stars_vs_forks_scatter('language', True)
        if predictions:
            dashboard['trend_predictions'] = self.create_trend_prediction_chart(predictions)
        if clustering_results:
            dashboard['cluster_analysis'] = self.create_cluster_visualization(clustering_results)
        if not self.df.empty and 'created_at' in self.df.columns:
            dashboard['creation_timeline'] = self._create_timeline_chart()

        return dashboard

    def _create_timeline_chart(self) -> go.Figure:
        """Create a repository creation timeline chart"""
        if self.df.empty or 'created_at' not in self.df.columns:
            return self._create_empty_chart("No creation date data available")

        df_copy = self.df.copy()
        df_copy['created_at'] = pd.to_datetime(df_copy['created_at'])
        df_copy['creation_year'] = df_copy['created_at'].dt.year

        yearly_counts = df_copy['creation_year'].value_counts().sort_index()

        fig = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            title='Repository Creation Timeline',
            labels={'x': 'Year', 'y': 'Number of Repositories Created'}
        )

        fig.update_traces(mode='lines+markers')
        return fig

    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=message,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig

    def save_charts(self, dashboard: Dict[str, go.Figure],
                    output_dir: str = "charts", formats: List[str] = ['html', 'png']):
        """
        Save dashboard charts to files

        Args:
            dashboard: Dictionary of charts
            output_dir: Output directory
            formats: List of formats to save ('html', 'png', 'pdf')
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for chart_name, fig in dashboard.items():
            for fmt in formats:
                filename = f"{output_dir}/{chart_name}.{fmt}"
                if fmt == 'html':
                    fig.write_html(filename)
                elif fmt == 'png':
                    fig.write_image(filename, width=1200, height=800)
                elif fmt == 'pdf':
                    fig.write_image(filename, format='pdf')

        print(f"Charts saved to {output_dir}/ in formats: {formats}")