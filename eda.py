"""
Exploratory Data Analysis module for tennis tournament predictor.
Generates comprehensive visualizations to understand the data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_missing_values(df, save_path='plots'):
    """Plot missing values heatmap and percentage bar chart."""
    print("Generating missing values visualizations...")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) == 0:
        print("  No missing values found!")
        return
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    top_missing = missing_df.head(20)
    ax.barh(range(len(top_missing)), top_missing['Percentage'].values)
    ax.set_yticks(range(len(top_missing)))
    ax.set_yticklabels(top_missing.index)
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title('Missing Values by Column (Top 20)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{save_path}/missing_values_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}/missing_values_bar.png")
    
    # Heatmap for subset of columns with most missing values
    cols_with_missing = missing_df.head(15).index.tolist()
    if len(cols_with_missing) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        sample_df = df[cols_with_missing].isnull()
        sns.heatmap(sample_df.head(1000), cbar=True, yticklabels=False, ax=ax, cmap='viridis')
        ax.set_title('Missing Values Heatmap (Sample of 1000 rows, Top 15 columns)')
        plt.tight_layout()
        plt.savefig(f'{save_path}/missing_values_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/missing_values_heatmap.png")


def plot_distributions(df, save_path='plots'):
    """Plot distributions of key numerical features."""
    print("Generating distribution plots...")
    
    # Rankings distribution
    if 'winner_rank' in df.columns and 'loser_rank' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df['winner_rank'].dropna().hist(bins=50, ax=axes[0], alpha=0.7, label='Winner')
        df['loser_rank'].dropna().hist(bins=50, ax=axes[1], alpha=0.7, label='Loser', color='orange')
        axes[0].set_xlabel('Rank')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Winner Rank Distribution')
        axes[1].set_xlabel('Rank')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Loser Rank Distribution')
        plt.tight_layout()
        plt.savefig(f'{save_path}/rank_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/rank_distributions.png")
    
    # Ranking points distribution
    if 'winner_rank_points' in df.columns and 'loser_rank_points' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df['winner_rank_points'].dropna().hist(bins=50, ax=axes[0], alpha=0.7)
        df['loser_rank_points'].dropna().hist(bins=50, ax=axes[1], alpha=0.7, color='orange')
        axes[0].set_xlabel('Ranking Points')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Winner Ranking Points Distribution')
        axes[1].set_xlabel('Ranking Points')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Loser Ranking Points Distribution')
        plt.tight_layout()
        plt.savefig(f'{save_path}/ranking_points_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/ranking_points_distributions.png")
    
    # Age distribution
    if 'winner_age' in df.columns and 'loser_age' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        df['winner_age'].dropna().hist(bins=30, ax=ax, alpha=0.5, label='Winner', color='blue')
        df['loser_age'].dropna().hist(bins=30, ax=ax, alpha=0.5, label='Loser', color='red')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution: Winners vs Losers')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/age_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/age_distribution.png")
    
    # Match duration
    if 'minutes' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        df['minutes'].dropna().hist(bins=50, ax=ax)
        ax.set_xlabel('Match Duration (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Match Duration Distribution')
        plt.tight_layout()
        plt.savefig(f'{save_path}/match_duration_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/match_duration_distribution.png")


def plot_categorical_features(df, save_path='plots'):
    """Plot distributions of categorical features."""
    print("Generating categorical feature plots...")
    
    # Surface distribution
    if 'surface' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        surface_counts = df['surface'].value_counts()
        axes[0].bar(surface_counts.index, surface_counts.values)
        axes[0].set_xlabel('Surface Type')
        axes[0].set_ylabel('Number of Matches')
        axes[0].set_title('Matches by Surface Type')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].pie(surface_counts.values, labels=surface_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Surface Type Distribution')
        plt.tight_layout()
        plt.savefig(f'{save_path}/surface_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/surface_distribution.png")
    
    # Tournament level
    if 'tourney_level' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        level_counts = df['tourney_level'].value_counts()
        ax.bar(level_counts.index, level_counts.values)
        ax.set_xlabel('Tournament Level')
        ax.set_ylabel('Number of Matches')
        ax.set_title('Matches by Tournament Level')
        plt.tight_layout()
        plt.savefig(f'{save_path}/tournament_level_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/tournament_level_distribution.png")
    
    # Round distribution
    if 'round' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        round_counts = df['round'].value_counts()
        ax.bar(round_counts.index, round_counts.values)
        ax.set_xlabel('Round')
        ax.set_ylabel('Number of Matches')
        ax.set_title('Matches by Round')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_path}/round_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/round_distribution.png")
    
    # Hand preference
    if 'winner_hand' in df.columns and 'loser_hand' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        hand_data = pd.concat([df['winner_hand'], df['loser_hand']]).value_counts()
        ax.bar(hand_data.index, hand_data.values)
        ax.set_xlabel('Hand Preference')
        ax.set_ylabel('Frequency')
        ax.set_title('Hand Preference Distribution (Winners + Losers)')
        plt.tight_layout()
        plt.savefig(f'{save_path}/hand_preference_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/hand_preference_distribution.png")


def plot_temporal_trends(df, save_path='plots'):
    """Plot temporal trends in the data."""
    print("Generating temporal trend plots...")
    
    if 'tourney_date' not in df.columns:
        print("  ⚠ tourney_date column not found, skipping temporal plots")
        return
    
    df_temp = df.copy()
    df_temp['tourney_date'] = pd.to_datetime(df_temp['tourney_date'], format='%Y%m%d', errors='coerce')
    df_temp = df_temp.dropna(subset=['tourney_date'])
    
    if len(df_temp) == 0:
        print("  ⚠ No valid dates found, skipping temporal plots")
        return
    
    df_temp['year'] = df_temp['tourney_date'].dt.year
    df_temp['month'] = df_temp['tourney_date'].dt.month
    
    # Matches per year
    fig, ax = plt.subplots(figsize=(12, 6))
    matches_per_year = df_temp.groupby('year').size()
    ax.plot(matches_per_year.index, matches_per_year.values, marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Matches')
    ax.set_title('Number of Matches per Year')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/matches_per_year.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}/matches_per_year.png")
    
    # Matches per month (seasonal patterns)
    fig, ax = plt.subplots(figsize=(12, 6))
    matches_per_month = df_temp.groupby('month').size()
    ax.bar(matches_per_month.index, matches_per_month.values)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Matches')
    ax.set_title('Matches per Month (Seasonal Patterns)')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig(f'{save_path}/matches_per_month.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {save_path}/matches_per_month.png")
    
    # Average match duration over time
    if 'minutes' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        avg_duration = df_temp.groupby('year')['minutes'].mean()
        ax.plot(avg_duration.index, avg_duration.values, marker='o', color='green')
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Match Duration (minutes)')
        ax.set_title('Average Match Duration Over Time')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/avg_duration_over_time.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/avg_duration_over_time.png")


def plot_ranking_analysis(df, save_path='plots'):
    """Plot ranking-related analysis."""
    print("Generating ranking analysis plots...")
    
    # Rank difference distribution
    if 'winner_rank' in df.columns and 'loser_rank' in df.columns:
        df_ranks = df[['winner_rank', 'loser_rank']].dropna()
        if len(df_ranks) > 0:
            df_ranks['rank_diff'] = df_ranks['winner_rank'] - df_ranks['loser_rank']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(df_ranks['rank_diff'], bins=50, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Equal Rank')
            ax.set_xlabel('Rank Difference (Winner Rank - Loser Rank)')
            ax.set_ylabel('Frequency')
            ax.set_title('Rank Difference Distribution')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{save_path}/rank_difference_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved: {save_path}/rank_difference_distribution.png")
            
            # Win rate by rank difference bins
            df_ranks['rank_diff_bin'] = pd.cut(df_ranks['rank_diff'], bins=20)
            win_rate_by_bin = df_ranks.groupby('rank_diff_bin').size()
            
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(range(len(win_rate_by_bin)), win_rate_by_bin.values)
            ax.set_xlabel('Rank Difference Bin')
            ax.set_ylabel('Number of Matches')
            ax.set_title('Match Frequency by Rank Difference Bins')
            ax.set_xticks(range(0, len(win_rate_by_bin), 2))
            ax.set_xticklabels([str(win_rate_by_bin.index[i]) for i in range(0, len(win_rate_by_bin), 2)], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'{save_path}/win_rate_by_rank_diff.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved: {save_path}/win_rate_by_rank_diff.png")
    
    # Seeded vs unseeded
    if 'winner_seed' in df.columns and 'loser_seed' in df.columns:
        df_seeds = df[['winner_seed', 'loser_seed']].copy()
        df_seeds['winner_seeded'] = df_seeds['winner_seed'].notna()
        df_seeds['loser_seeded'] = df_seeds['loser_seed'].notna()
        
        seeded_counts = pd.DataFrame({
            'Winner Seeded': df_seeds['winner_seeded'].sum(),
            'Winner Unseeded': (~df_seeds['winner_seeded']).sum(),
            'Loser Seeded': df_seeds['loser_seeded'].sum(),
            'Loser Unseeded': (~df_seeds['loser_seeded']).sum()
        }, index=[0])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        seeded_counts.T.plot(kind='bar', ax=ax)
        ax.set_ylabel('Number of Matches')
        ax.set_title('Seeded vs Unseeded Players')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(['Count'])
        plt.tight_layout()
        plt.savefig(f'{save_path}/seeded_vs_unseeded.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/seeded_vs_unseeded.png")


def plot_match_statistics(df, save_path='plots'):
    """Plot match statistics distributions and correlations."""
    print("Generating match statistics plots...")
    
    # Aces distribution
    if 'w_ace' in df.columns and 'l_ace' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df['w_ace'].dropna().hist(bins=30, ax=axes[0], alpha=0.7, color='blue')
        df['l_ace'].dropna().hist(bins=30, ax=axes[1], alpha=0.7, color='orange')
        axes[0].set_xlabel('Aces')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Winner Aces Distribution')
        axes[1].set_xlabel('Aces')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Loser Aces Distribution')
        plt.tight_layout()
        plt.savefig(f'{save_path}/aces_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/aces_distribution.png")
    
    # Double faults distribution
    if 'w_df' in df.columns and 'l_df' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df['w_df'].dropna().hist(bins=30, ax=axes[0], alpha=0.7, color='blue')
        df['l_df'].dropna().hist(bins=30, ax=axes[1], alpha=0.7, color='orange')
        axes[0].set_xlabel('Double Faults')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Winner Double Faults Distribution')
        axes[1].set_xlabel('Double Faults')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Loser Double Faults Distribution')
        plt.tight_layout()
        plt.savefig(f'{save_path}/double_faults_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/double_faults_distribution.png")
    
    # Correlation heatmap
    stat_cols = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon']
    available_stat_cols = [col for col in stat_cols if col in df.columns]
    
    if len(available_stat_cols) > 2:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[available_stat_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)
        ax.set_title('Match Statistics Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(f'{save_path}/match_stats_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Saved: {save_path}/match_stats_correlation.png")


def plot_key_insights(df, save_path='plots'):
    """Plot key insights: upsets, surface specialization, etc."""
    print("Generating key insights plots...")
    
    # Upsets analysis (lower ranked player beating higher ranked)
    if 'winner_rank' in df.columns and 'loser_rank' in df.columns:
        df_ranks = df[['winner_rank', 'loser_rank']].dropna()
        if len(df_ranks) > 0:
            df_ranks['upset'] = df_ranks['winner_rank'] > df_ranks['loser_rank']
            upset_rate = df_ranks['upset'].mean() * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            upset_counts = df_ranks['upset'].value_counts()
            labels = ['Higher Rank Wins', 'Upset (Lower Rank Wins)']
            ax.pie(upset_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Upset Analysis\n(Upset Rate: {upset_rate:.2f}%)')
            plt.tight_layout()
            plt.savefig(f'{save_path}/upset_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved: {save_path}/upset_analysis.png")
    
    # Surface specialization (win rate by surface for top players)
    if 'surface' in df.columns and 'winner_name' in df.columns:
        top_players = df['winner_name'].value_counts().head(10).index.tolist()
        surface_win_rates = []
        
        for surface in df['surface'].dropna().unique():
            surface_matches = df[df['surface'] == surface]
            if len(surface_matches) > 0:
                for player in top_players:
                    player_wins = len(surface_matches[surface_matches['winner_name'] == player])
                    player_losses = len(surface_matches[surface_matches['loser_name'] == player])
                    total = player_wins + player_losses
                    if total > 0:
                        win_rate = player_wins / total * 100
                        surface_win_rates.append({
                            'Player': player,
                            'Surface': surface,
                            'Win Rate': win_rate,
                            'Matches': total
                        })
        
        if len(surface_win_rates) > 0:
            surface_df = pd.DataFrame(surface_win_rates)
            pivot_df = surface_df.pivot(index='Player', columns='Surface', values='Win Rate')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Win Rate (%)'})
            ax.set_title('Top 10 Players: Win Rate by Surface')
            ax.set_xlabel('Surface')
            ax.set_ylabel('Player')
            plt.tight_layout()
            plt.savefig(f'{save_path}/surface_specialization.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [OK] Saved: {save_path}/surface_specialization.png")


def generate_all_eda_plots(df, save_path='plots'):
    """
    Generate all EDA plots.
    
    Args:
        df: DataFrame with match data
        save_path: Directory to save plots
    """
    print("\n" + "="*60)
    print("GENERATING EXPLORATORY DATA ANALYSIS PLOTS")
    print("="*60)
    
    # Ensure save directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    plot_missing_values(df, save_path)
    plot_distributions(df, save_path)
    plot_categorical_features(df, save_path)
    plot_temporal_trends(df, save_path)
    plot_ranking_analysis(df, save_path)
    plot_match_statistics(df, save_path)
    plot_key_insights(df, save_path)
    
    print("\n" + "="*60)
    print("[OK] All EDA plots generated successfully!")
    print(f"  Plots saved to: {save_path}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test the EDA module
    from data_loader import load_data
    
    df = load_data()
    generate_all_eda_plots(df)
