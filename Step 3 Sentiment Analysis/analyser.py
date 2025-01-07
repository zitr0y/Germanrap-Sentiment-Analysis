from typing import Tuple, List, Optional, Set
import os
import sqlite3
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


os.chdir(os.path.dirname(os.path.realpath(__file__)))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



class SentimentAnalysisResults:
    def __init__(self, 
                 db_path: str = '../Step 3.1 Tack together time data with database/rapper_sentiments_with_time.db',
                 excluded_rappers: Optional[List[str]] = None,
                 output_dir: str = 'sentiment_analysis_results',
                 dist_top_n: int = 10,
                 high_volume_threshold: int = 300):
        # Create output directory if it doesn't exist
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store configuration
        self.dist_top_n = dist_top_n
        self.high_volume_threshold = high_volume_threshold
        
        # Default excluded rappers that usually don't refer to the actual rapper
        default_excluded = ['boss', 'headie_one', 'woke', 'aitch', 'grandmaster_flash', 'kendrick_lamar', 'level']
        
        # Combine default and user-provided excluded rappers
        self.excluded_rappers = set(default_excluded)
        if excluded_rappers:
            self.excluded_rappers.update(excluded_rappers)
            
        self.conn = sqlite3.connect(db_path)
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load data from SQLite database including timestamps."""
        query = """
        SELECT *, datetime(original_timestamp, 'unixepoch') as datetime
        FROM sentiment_analysis 
        WHERE sentiment NOT IN ('NO_SENTIMENT', 'ERROR')
        AND sentiment IS NOT NULL
        """
        df = pd.read_sql_query(query, self.conn)
        df['sentiment'] = pd.to_numeric(df['sentiment'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Exclude specified rappers
        if self.excluded_rappers:
            df = df[~df['rapper_name'].isin(self.excluded_rappers)]
            
        return df
    
    def _calculate_stats(self, df: pd.DataFrame, min_mentions: int = 5) -> pd.DataFrame:
        """Calculate basic statistics for each rapper."""
        stats = df.groupby('rapper_name').agg({
            'sentiment': ['count', 'mean', 'std']
        }).round(3)
        
        stats.columns = ['mention_count', 'avg_sentiment', 'sentiment_std']
        stats = stats[stats['mention_count'] >= min_mentions]
        return stats
    
    def _get_top_n_by_mentions(self, n: Optional[int] = None) -> Set[str]:
        """Get set of top N most mentioned rappers."""
        if n is None:
            return set(self.df['rapper_name'].unique())
            
        mention_counts = self.df['rapper_name'].value_counts()
        return set(mention_counts.head(n).index)
    
    def calculate_composite_score(self, stats: pd.DataFrame,
                                mention_weight: float = 0.1, 
                                sentiment_weight: float = 0.7, 
                                std_weight: float = 0.2) -> pd.Series:
        """
        Calculate composite score considering mentions, sentiment and std deviation.
        
        Weights should sum to 1.0
        mention_weight: How much to weight number of mentions (normalized)
        sentiment_weight: How much to weight average sentiment
        std_weight: How much to weight consistency (inverse of std dev)
        """
        # Normalize mention count to 0-1 scale relative to max mentions
        mention_scores = stats['mention_count'] / stats['mention_count'].max()
        
        # Normalize sentiment to 0-1 scale (it's already 1-5)
        sentiment_scores = (stats['avg_sentiment'] - 1) / 4
        
        # Convert std dev to consistency score (lower std = higher consistency)
        # Normalize to 0-1 scale where 1 is most consistent
        max_std = 2.0  # Maximum possible std dev for 1-5 scale
        consistency_scores = 1 - (stats['sentiment_std'] / max_std)
        
        # Calculate weighted sum
        composite_scores = (mention_weight * mention_scores + 
                          sentiment_weight * sentiment_scores + 
                          std_weight * consistency_scores)
                         
        return composite_scores.round(3)
    
    def analyze_by_mention_threshold(self, mention_threshold: int) -> pd.DataFrame:
        """Analyze only rappers with more than threshold mentions."""
        mention_counts = self.df['rapper_name'].value_counts()
        qualified_rappers = set(mention_counts[mention_counts >= mention_threshold].index)
        
        filtered_df = self.df[self.df['rapper_name'].isin(qualified_rappers)]
        stats = self._calculate_stats(filtered_df, min_mentions=mention_threshold)
        
        # Add composite score
        stats['composite_score'] = self.calculate_composite_score(stats)
        
        return stats.sort_values('avg_sentiment', ascending=False)
    
    def analyze_top_n_rappers(self, top_n: int) -> pd.DataFrame:
        """Analyze only the top N most mentioned rappers."""
        top_rappers = self._get_top_n_by_mentions(top_n)
        filtered_df = self.df[self.df['rapper_name'].isin(top_rappers)]
        stats = self._calculate_stats(filtered_df, min_mentions=1)  # min_mentions=1 since we already filtered
        return stats.sort_values('mention_count', ascending=False)
    
    def get_most_mentioned(self, top_n: int = 20, min_mentions: int = 5) -> pd.DataFrame:
        """Get the most mentioned rappers with their sentiment stats."""
        stats = self._calculate_stats(self.df, min_mentions)
        return stats.sort_values('mention_count', ascending=False).head(top_n)
    
    def get_best_rated(self, top_n: int = 20, min_mentions: int = 5) -> pd.DataFrame:
        """Get the best rated rappers."""
        stats = self._calculate_stats(self.df, min_mentions)
        return stats.sort_values('avg_sentiment', ascending=False).head(top_n)
    
    def get_worst_rated(self, top_n: int = 20, min_mentions: int = 5) -> pd.DataFrame:
        """Get the worst rated rappers."""
        stats = self._calculate_stats(self.df, min_mentions)
        return stats.sort_values('avg_sentiment', ascending=True).head(top_n)
    
    def get_most_controversial(self, top_n: int = 20, min_mentions: int = 5) -> pd.DataFrame:
        """Get rappers with highest standard deviation in ratings."""
        stats = self._calculate_stats(self.df, min_mentions)
        return stats.sort_values('sentiment_std', ascending=False).head(top_n)
    
    def get_stats_without_neutral(self, top_n: int = 20, min_mentions: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get best and worst rated rappers excluding neutral (3) ratings."""
        df_no_neutral = self.df[self.df['sentiment'] != 3]
        stats = self._calculate_stats(df_no_neutral, min_mentions)
        
        best = stats.sort_values('avg_sentiment', ascending=False).head(top_n)
        worst = stats.sort_values('avg_sentiment', ascending=True).head(top_n)
        
        return best, worst
    
    def get_sentiment_distribution(self, top_n_rappers: int = 10) -> pd.DataFrame:
        """Get sentiment distribution for top mentioned rappers."""
        top_rappers = self.get_most_mentioned(top_n_rappers)['mention_count'].index
        
        # Create a list to store each rapper's distribution
        distributions = []
        
        for rapper in top_rappers:
            counts = self.df[self.df['rapper_name'] == rapper]['sentiment'].value_counts()
            dist_series = counts.reindex(range(1, 6)).fillna(0).astype(int)
            distributions.append(dist_series)
        
        # Concatenate all distributions at once
        dist = pd.concat(distributions, axis=1)
        dist.columns = top_rappers
    
        return dist    
    def analyze_popular_rappers(self, mention_threshold: int = 500) -> dict:
        """Detailed analysis of popular rappers (above threshold mentions)."""
        mention_counts = self.df['rapper_name'].value_counts()
        qualified_rappers = set(mention_counts[mention_counts >= mention_threshold].index)
        
        # For analysis with neutral ratings
        filtered_df = self.df[self.df['rapper_name'].isin(qualified_rappers)]
        stats_with_neutral = self._calculate_stats(filtered_df, min_mentions=mention_threshold)
        stats_with_neutral['composite_score'] = self.calculate_composite_score(stats_with_neutral)
        
        # For analysis without neutral ratings
        filtered_df_no_neutral = filtered_df[filtered_df['sentiment'] != 3]
        stats_without_neutral = self._calculate_stats(filtered_df_no_neutral, min_mentions=mention_threshold)
        stats_without_neutral['composite_score'] = self.calculate_composite_score(stats_without_neutral)
        
        return {
            'with_neutral': {
                'best_rated': stats_with_neutral.sort_values('avg_sentiment', ascending=False),
                'worst_rated': stats_with_neutral.sort_values('avg_sentiment', ascending=True),
                'most_controversial': stats_with_neutral.sort_values('sentiment_std', ascending=False),
                'most_consistent': stats_with_neutral.sort_values('sentiment_std', ascending=True),
                'highest_composite': stats_with_neutral.sort_values('composite_score', ascending=False)
            },
            'without_neutral': {
                'best_rated': stats_without_neutral.sort_values('avg_sentiment', ascending=False),
                'worst_rated': stats_without_neutral.sort_values('avg_sentiment', ascending=True),
                'most_controversial': stats_without_neutral.sort_values('sentiment_std', ascending=False),
                'most_consistent': stats_without_neutral.sort_values('sentiment_std', ascending=True),
                'highest_composite': stats_without_neutral.sort_values('composite_score', ascending=False)
            }
        }
    def create_timeline_analysis(self, top_n: int = 5) -> go.Figure:
        """Create timeline analysis for top N most mentioned rappers."""
        # Get top N rappers by mention count
        top_rappers = self.df['rapper_name'].value_counts().head(top_n).index

        # Create timeline data for each rapper
        fig = go.Figure()

        for rapper in top_rappers:
            rapper_data = self.df[self.df['rapper_name'] == rapper]
            
            # Resample by month and calculate metrics
            monthly_stats = rapper_data.set_index('datetime').resample('ME').agg({
                'sentiment': ['mean', 'count']
            })
            monthly_stats.columns = ['avg_sentiment', 'mentions']

            # Create hover text
            hover_text = [
                f"Date: {date.strftime('%Y-%m')}<br>" +
                f"Average Sentiment: {sent:.2f}<br>" +
                f"Mentions: {count}"
                for date, sent, count in zip(
                    monthly_stats.index,
                    monthly_stats['avg_sentiment'],
                    monthly_stats['mentions']
                )
            ]

            # Add lines for both sentiment and mentions
            fig.add_trace(
                go.Scatter(
                    x=monthly_stats.index,
                    y=monthly_stats['avg_sentiment'],
                    name=f"{rapper} (Sentiment)",
                    text=hover_text,
                    hoverinfo='text',
                    line=dict(width=2),
                    visible='legendonly' if len(fig.data) >= 10 else True
                )
            )

        fig.update_layout(
            title=f"Timeline Analysis of Top {top_n} Most Mentioned Rappers",
            xaxis_title="Time",
            yaxis_title="Average Sentiment",
            hovermode='closest',
            template='plotly_white',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def create_visualizations(self, top_n: int = 50):
        """Create various visualizations using Plotly."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Get top N rappers by mentions to ensure consistency across plots
        top_rappers = self.get_most_mentioned(top_n)
        top_rapper_names = set(top_rappers.index)

        # 1. Top N Most Mentioned Rappers bar chart
        fig_mentions = px.bar(
            top_rappers,
            x=top_rappers.index,
            y='mention_count',
            title=f'Top {top_n} Most Mentioned Rappers',
            labels={'mention_count': 'Number of Mentions', 'index': 'Rapper'},
            height=800
        )
        fig_mentions.update_layout(
            xaxis_tickangle=-45,
            template='plotly_white',
            showlegend=False,
            margin=dict(b=150)
        )
        fig_mentions.write_html(self.output_dir / f'top_{top_n}_mentions_{timestamp}.html')

        # 2. Sentiment Distribution Heatmap
        dist = self.get_sentiment_distribution(top_n)
        fig_heatmap = px.imshow(
            dist,
            title=f'Sentiment Distribution for Top {top_n} Rappers',
            labels=dict(x='Rapper', y='Sentiment Value', color='Count'),
            aspect='auto',
            height=800
        )
        fig_heatmap.update_layout(
            xaxis_tickangle=-45,
            template='plotly_white',
            margin=dict(b=150)
        )
        fig_heatmap.write_html(self.output_dir / f'sentiment_heatmap_{timestamp}.html')

        # 3. Average Sentiment vs Mentions Scatter Plot (excluding neutral ratings)
        df_no_neutral = self.df[self.df['sentiment'] != 3]
        # Filter for only top N rappers
        df_no_neutral_top = df_no_neutral[df_no_neutral['rapper_name'].isin(top_rapper_names)]
        stats_no_neutral = self._calculate_stats(df_no_neutral_top, min_mentions=5)
        
        fig_scatter = px.scatter(
            stats_no_neutral,
            x='mention_count',
            y='avg_sentiment',
            title=f'Average Sentiment vs Number of Mentions for Top {top_n} Rappers (Excluding Neutral Ratings)',
            labels={
                'mention_count': 'Number of Mentions',
                'avg_sentiment': 'Average Sentiment'
            },
            text=stats_no_neutral.index,
            height=800
        )
        fig_scatter.update_traces(
            textposition='top center',
            hovertemplate="<br>".join([
                "Rapper: %{text}",
                "Mentions: %{x}",
                "Avg Sentiment: %{y:.2f}"
            ])
        )
        fig_scatter.update_layout(
            template='plotly_white',
            xaxis_type="log",
            yaxis=dict(range=[1, 5])
        )
        fig_scatter.write_html(self.output_dir / f'sentiment_vs_mentions_{timestamp}.html')

        # 4. Controversy vs Popularity (excluding neutral ratings)
        fig_controversy = px.scatter(
            stats_no_neutral,  # Using same filtered stats as above
            x='mention_count',
            y='sentiment_std',
            title=f'Controversy vs Popularity for Top {top_n} Rappers (Excluding Neutral Ratings)',
            labels={
                'mention_count': 'Number of Mentions',
                'sentiment_std': 'Sentiment Standard Deviation'
            },
            text=stats_no_neutral.index,
            height=800
        )
        fig_controversy.update_traces(
            textposition='top center',
            hovertemplate="<br>".join([
                "Rapper: %{text}",
                "Mentions: %{x}",
                "Std Dev: %{y:.2f}"
            ])
        )
        fig_controversy.update_layout(
            template='plotly_white',
            xaxis_type="log",
            yaxis=dict(range=[0, stats_no_neutral['sentiment_std'].max() * 1.1])
        )
        fig_controversy.write_html(self.output_dir / f'controversy_vs_popularity_{timestamp}.html')

        # 5. Timeline Analysis
        fig_timeline = self.create_timeline_analysis(top_n=5)  # Keep this at 5 to avoid overcrowding
        fig_timeline.write_html(self.output_dir / f'timeline_analysis_{timestamp}.html')
    
    def get_overall_sentiment_distribution(self) -> pd.DataFrame:
        """Calculate the overall sentiment distribution across all rappers."""
        # Get overall counts
        overall_dist = self.df['sentiment'].value_counts().sort_index()
        
        # Calculate percentages
        total_ratings = len(self.df)
        overall_dist_pct = (overall_dist / total_ratings * 100).round(2)
        
        # Combine counts and percentages
        distribution = pd.DataFrame({
            'count': overall_dist,
            'percentage': overall_dist_pct
        })
        
        return distribution

    def plot_overall_sentiment_distribution(self) -> go.Figure:
        """Create a visualization of the overall sentiment distribution."""
        distribution = self.get_overall_sentiment_distribution()
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for counts
        fig.add_trace(
            go.Bar(
                x=distribution.index,
                y=distribution['count'],
                name="Count",
                marker_color='rgb(55, 83, 109)',
                opacity=0.7,
            ),
            secondary_y=False,
        )
        
        # Add line chart for percentages
        fig.add_trace(
            go.Scatter(
                x=distribution.index,
                y=distribution['percentage'],
                name="Percentage",
                mode='lines+markers',
                marker=dict(size=10),
                line=dict(width=3, color='rgb(200, 50, 50)'),
                yaxis='y2'
            ),
            secondary_y=True,
        )
        
        # Update layout
        fig.update_layout(
            title="Overall Sentiment Distribution",
            xaxis_title="Sentiment Rating",
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            bargap=0.2
        )
        
        # Update yaxis labels
        fig.update_yaxes(title_text="Number of Ratings", secondary_y=False)
        fig.update_yaxes(title_text="Percentage of Total (%)", secondary_y=True)
        
        return fig

    def analyze_overall_sentiment(self) -> dict:
        """Analyze overall sentiment statistics."""
        distribution = self.get_overall_sentiment_distribution()
        
        # Calculate additional statistics
        stats = {
            'total_ratings': len(self.df),
            'mean_sentiment': self.df['sentiment'].mean().round(3),
            'median_sentiment': self.df['sentiment'].median(),
            'std_sentiment': self.df['sentiment'].std().round(3),
            'distribution': distribution.to_dict(),
            'most_common_rating': distribution['count'].idxmax(),
            'most_common_count': distribution['count'].max(),
            'most_common_percentage': distribution['percentage'].max().round(2)
        }
        
        # Calculate polarity ratios (excluding neutral)
        non_neutral = self.df[self.df['sentiment'] != 3]
        positive_ratings = non_neutral[non_neutral['sentiment'] > 3]['sentiment'].count()
        negative_ratings = non_neutral[non_neutral['sentiment'] < 3]['sentiment'].count()
        total_non_neutral = positive_ratings + negative_ratings
        
        stats.update({
            'positive_ratio': (positive_ratings / total_non_neutral * 100).round(2) if total_non_neutral > 0 else 0,
            'negative_ratio': (negative_ratings / total_non_neutral * 100).round(2) if total_non_neutral > 0 else 0,
            'neutral_percentage': (self.df[self.df['sentiment'] == 3]['sentiment'].count() / len(self.df) * 100).round(2)
        })
        
        return stats
    
    def generate_report(self, min_mentions: int = 5, save_to_csv: bool = True):
        """Generate a comprehensive report of all analyses."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        print("=== Sentiment Analysis Report ===\n")
        
        print("1. Most Mentioned Rappers (Top 20)")
        print(self.get_most_mentioned(20, min_mentions))
        print("\n" + "="*50 + "\n")
        
        print("2. Best Rated Rappers (Top 20)")
        print(self.get_best_rated(20, min_mentions))
        print("\n" + "="*50 + "\n")
        
        print("3. Worst Rated Rappers (Top 20)")
        print(self.get_worst_rated(20, min_mentions))
        print("\n" + "="*50 + "\n")
        
        print("4. Most Controversial Rappers (Highest Rating Variance)")
        print(self.get_most_controversial(20, min_mentions))
        print("\n" + "="*50 + "\n")
        
        print("5. Analysis Excluding Neutral Ratings")
        best_no_neutral, worst_no_neutral = self.get_stats_without_neutral(20, min_mentions)
        print("\nBest Rated (Excluding Neutral):")
        print(best_no_neutral)
        print("\nWorst Rated (Excluding Neutral):")
        print(worst_no_neutral)
        print("\n" + "="*50 + "\n")
        
        print(f"6. Sentiment Distribution for Top {self.dist_top_n} Most Mentioned Rappers")
        dist = self.get_sentiment_distribution(self.dist_top_n)
        print("Number of ratings per sentiment value (1-5):")
        print(dist)
        print("\n" + "="*50 + "\n")
        
        print(f"7. Analysis of Popular Rappers ({self.high_volume_threshold}+ mentions)")
        high_volume_analysis = self.analyze_popular_rappers(self.high_volume_threshold)
        
        print("\nA. Analysis Including All Ratings:")
        print("\nBest Rated Popular Rappers:")
        print(high_volume_analysis['with_neutral']['best_rated'])
        print("\nMost Controversial Popular Rappers:")
        print(high_volume_analysis['with_neutral']['most_controversial'])
        print("\nHighest Composite Score (Mentions + Sentiment + Consistency):")
        print(high_volume_analysis['with_neutral']['highest_composite'])
        
        print("\nB. Analysis Excluding Neutral Ratings:")
        print("\nBest Rated Popular Rappers (No Neutral):")
        print(high_volume_analysis['without_neutral']['best_rated'])
        print("\nMost Controversial Popular Rappers (No Neutral):")
        print(high_volume_analysis['without_neutral']['most_controversial'])
        print("\nHighest Composite Score (No Neutral):")
        print(high_volume_analysis['without_neutral']['highest_composite'])
        print("\n" + "="*50 + "\n")
        
        # Get overall sentiment distribution
        overall_dist = self.get_overall_sentiment_distribution()
        print("Overall Sentiment Distribution:")
        print(overall_dist)

        # Get comprehensive statistics
        overall_stats = self.analyze_overall_sentiment()
        print("\nOverall Sentiment Statistics:")
        print(f"Total ratings: {overall_stats['total_ratings']}")
        print(f"Mean sentiment: {overall_stats['mean_sentiment']}")
        print(f"Positive ratio: {overall_stats['positive_ratio']}%")
        print(f"Negative ratio: {overall_stats['negative_ratio']}%")
        print(f"Neutral percentage: {overall_stats['neutral_percentage']}%")

        # Create and save visualization
        fig = self.plot_overall_sentiment_distribution()
        fig.write_html("overall_sentiment_distribution.html")

        if save_to_csv:
            # Save results to CSV files in output directory
            self.get_most_mentioned(50, min_mentions).to_csv(self.output_dir / f'most_mentioned_{timestamp}.csv')
            self.get_best_rated(50, min_mentions).to_csv(self.output_dir / f'best_rated_{timestamp}.csv')
            self.get_worst_rated(50, min_mentions).to_csv(self.output_dir / f'worst_rated_{timestamp}.csv')
            high_volume_analysis['with_neutral']['highest_composite'].to_csv(
                self.output_dir / f'high_volume_composite_{timestamp}.csv')
            high_volume_analysis['without_neutral']['highest_composite'].to_csv(
                self.output_dir / f'high_volume_composite_no_neutral_{timestamp}.csv')
            
            # Create visualizations
            self.create_visualizations()
        
        return {
            'most_mentioned': self.get_most_mentioned(50, min_mentions),
            'best_rated': self.get_best_rated(50, min_mentions),
            'worst_rated': self.get_worst_rated(50, min_mentions),
            'most_controversial': self.get_most_controversial(50, min_mentions),
            'best_no_neutral': best_no_neutral,
            'worst_no_neutral': worst_no_neutral,
            'sentiment_distribution': dist,
            'high_volume_analysis': high_volume_analysis
        }

def main():
    # Example usage with customized parameters
    analyzer = SentimentAnalysisResults(
        output_dir='sentiment_analysis_results',
        dist_top_n=10,                 # Number of rappers in distribution analysis
        high_volume_threshold=300      # Minimum mentions for high-volume analysis
    )
    results = analyzer.generate_report(min_mentions=50)

if __name__ == "__main__":
    main()