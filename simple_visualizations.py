#!/usr/bin/env python3
"""
Simple COVID-19 Cluster Analysis Visualizations
This script creates basic visualizations of the COVID-19 cluster analysis results
using only standard libraries that are commonly available.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap

# Create output directory if it doesn't exist
output_dir = 'visualization_outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def load_data():
    """Load the clustering results and cleaned data."""
    print("Loading data...")
    
    # Load clustering results
    clusters_df = pd.read_csv('covid_clustering_results.csv')
    
    # Load cleaned data for time series analysis
    cleaned_df = pd.read_csv('cleaned_covid_data.csv')
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
    
    print(f"Loaded data for {len(clusters_df)} countries")
    return clusters_df, cleaned_df

def create_cluster_scatter(df, cluster_col, x_col, y_col, cluster_labels, output_dir):
    """Create a scatter plot showing the clusters."""
    print(f"Creating scatter plot for {x_col} vs {y_col}...")
    
    # Create a copy of the dataframe with cluster labels
    scatter_df = df.copy()
    scatter_df['cluster_label'] = scatter_df[cluster_col].map(cluster_labels)
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    
    # Get unique clusters
    unique_clusters = scatter_df[cluster_col].unique()
    
    # Create a colormap
    colors = plt.cm.tab10(range(len(unique_clusters)))
    
    # Plot each cluster
    for i, cluster in enumerate(unique_clusters):
        cluster_data = scatter_df[scatter_df[cluster_col] == cluster]
        plt.scatter(
            cluster_data[x_col], 
            cluster_data[y_col], 
            c=[colors[i]], 
            label=f"{cluster_labels.get(cluster, f'Cluster {cluster}')} (n={len(cluster_data)})",
            alpha=0.7,
            s=80
        )
    
    # Add labels for specific countries of interest
    countries_to_label = ['United States', 'China', 'Italy', 'Brazil', 'India', 'United Kingdom']
    for country in countries_to_label:
        if country in scatter_df['location'].values:
            country_data = scatter_df[scatter_df['location'] == country]
            plt.annotate(
                country,
                (country_data[x_col].values[0], country_data[y_col].values[0]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )
    
    plt.title(f"COVID-19 Cluster Analysis: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}")
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'cluster_scatter_{x_col}_vs_{y_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {output_path}")

def create_cluster_comparison(df, cluster_col, metrics, cluster_labels, output_dir):
    """Create a bar chart comparing cluster metrics."""
    print(f"Creating cluster comparison for {len(metrics)} metrics...")
    
    # Calculate the mean of each metric for each cluster
    comparison_df = df.groupby(cluster_col)[metrics].mean().reset_index()
    comparison_df['cluster_label'] = comparison_df[cluster_col].map(cluster_labels)
    comparison_df['cluster_size'] = df.groupby(cluster_col).size().values
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    
    # If there's only one metric, axes will not be an array
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Sort by cluster number
        sorted_df = comparison_df.sort_values(cluster_col)
        
        # Create the bar chart
        bars = ax.bar(
            sorted_df[cluster_col].astype(str), 
            sorted_df[metric],
            color=plt.cm.tab10(range(len(sorted_df)))
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01 * max(sorted_df[metric]),
                f'{height:.2f}',
                ha='center', 
                va='bottom',
                fontsize=8
            )
        
        # Add cluster size as text below x-axis
        for j, (cluster, size) in enumerate(zip(sorted_df[cluster_col], sorted_df['cluster_size'])):
            ax.text(
                j,
                -0.05 * max(sorted_df[metric]),
                f'n={size}',
                ha='center',
                va='top',
                fontsize=8
            )
        
        ax.set_title(f"{metric.replace('_', ' ').title()} by Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add cluster labels as a second x-axis
        ax2 = ax.twiny()
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticklabels([cluster_labels.get(int(c), f"Cluster {c}") for c in sorted_df[cluster_col]])
        ax2.tick_params(axis='x', labelsize=8, rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'cluster_comparison_{cluster_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster comparison to {output_path}")

def create_time_series(df, countries, metric, start_date, end_date, output_dir):
    """Create a time series plot for selected countries."""
    print(f"Creating time series for {metric} across {len(countries)} countries...")
    
    # Check if the metric exists in the dataframe
    if metric not in df.columns:
        print(f"Warning: {metric} not found in the dataframe. Available metrics: {', '.join(df.columns)}")
        return
    
    # Filter the data
    ts_df = df[df['location'].isin(countries)].copy()
    ts_df = ts_df[(ts_df['date'] >= start_date) & (ts_df['date'] <= end_date)]
    
    if ts_df.empty:
        print(f"Warning: No data found for the selected countries and date range.")
        return
    
    # Create the time series plot
    plt.figure(figsize=(12, 8))
    
    # Plot each country
    for country in countries:
        country_data = ts_df[ts_df['location'] == country]
        if not country_data.empty:
            plt.plot(
                country_data['date'], 
                country_data[metric], 
                label=country,
                linewidth=2,
                marker='o',
                markersize=1
            )
    
    plt.title(f"COVID-19 Time Series: {metric.replace('_', ' ').title()}")
    plt.xlabel("Date")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend(title="Countries", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'time_series_{metric}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved time series to {output_path}")
    
    # Create a smoothed version
    print(f"Creating smoothed time series...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot each country with a rolling average
    for country in countries:
        country_data = ts_df[ts_df['location'] == country].copy()
        if not country_data.empty:
            # Create a rolling average
            country_data = country_data.sort_values('date')
            country_data[f"{metric}_smoothed"] = country_data[metric].rolling(7, min_periods=1).mean()
            
            plt.plot(
                country_data['date'], 
                country_data[f"{metric}_smoothed"], 
                label=country,
                linewidth=2
            )
    
    plt.title(f"COVID-19 Time Series (7-day Rolling Average): {metric.replace('_', ' ').title()}")
    plt.xlabel("Date")
    plt.ylabel(f"{metric.replace('_', ' ').title()} (7-day Avg)")
    plt.legend(title="Countries", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'time_series_{metric}_smoothed.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved smoothed time series to {output_path}")

def create_cluster_distribution(df, cluster_col, cluster_labels, output_dir):
    """Create a pie chart showing the distribution of countries across clusters."""
    print(f"Creating cluster distribution chart...")
    
    # Count the number of countries in each cluster
    cluster_counts = df[cluster_col].value_counts().reset_index()
    cluster_counts.columns = [cluster_col, 'count']
    
    # Add cluster labels
    cluster_counts['cluster_label'] = cluster_counts[cluster_col].map(cluster_labels)
    
    # Sort by cluster
    cluster_counts = cluster_counts.sort_values(cluster_col)
    
    # Create the pie chart
    plt.figure(figsize=(10, 8))
    
    # Create custom labels with cluster number, label, and count
    labels = [f"Cluster {row[cluster_col]} - {row['cluster_label']}\n({row['count']} countries)" 
              for _, row in cluster_counts.iterrows()]
    
    plt.pie(
        cluster_counts['count'],
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.tab10(range(len(cluster_counts))),
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    plt.title('Distribution of Countries Across Clusters')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save the figure
    output_path = os.path.join(output_dir, f'cluster_distribution_{cluster_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cluster distribution chart to {output_path}")

def create_cluster_profiles(df, cluster_col, cluster_labels, output_dir):
    """Create a visualization of cluster profiles using radar charts."""
    print(f"Creating cluster profiles...")
    
    # Get the key features for profiling
    profile_features = [
        'case_rate_per_100k', 'mortality_rate_per_100k', 'testing_rate_per_1k',
        'vaccination_rate_pct', 'hospitalization_rate_per_100k', 'icu_rate_per_100k', 
        'recovery_speed', 'case_fatality_rate_pct', 'vax_acceleration'
    ]
    
    # Filter to features that exist in the dataframe
    profile_features = [col for col in profile_features if col in df.columns]
    
    if len(profile_features) < 3:
        print("Warning: Not enough features for radar chart. Skipping cluster profiles.")
        return
    
    # Create a copy of the dataframe with cluster labels
    profile_df = df.copy()
    profile_df['cluster_label'] = profile_df[cluster_col].map(cluster_labels)
    
    # Calculate the mean of each feature for each cluster
    cluster_profiles = profile_df.groupby(cluster_col)[profile_features].mean()
    
    # Normalize the features for radar chart
    normalized_profiles = cluster_profiles.copy()
    for feature in profile_features:
        min_val = normalized_profiles[feature].min()
        max_val = normalized_profiles[feature].max()
        if max_val > min_val:  # Avoid division by zero
            normalized_profiles[feature] = (normalized_profiles[feature] - min_val) / (max_val - min_val)
        else:
            normalized_profiles[feature] = 0  # Set to 0 if all values are the same
    
    # Create radar charts for each cluster
    for cluster in normalized_profiles.index:
        # Get the data for this cluster
        cluster_data = normalized_profiles.loc[cluster].values
        
        # Number of variables
        N = len(profile_features)
        
        # Create a figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Repeat the first value to close the polygon
        values = np.append(cluster_data, cluster_data[0])
        
        # Compute angle for each feature
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label=cluster_labels.get(cluster, f"Cluster {cluster}"))
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace('_', ' ').title() for f in profile_features], fontsize=8)
        
        # Draw y-axis lines from center to edge
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
        ax.set_ylim(0, 1)
        
        plt.title(f"Profile for {cluster_labels.get(cluster, f'Cluster {cluster}')}")
        
        # Save the figure
        output_path = os.path.join(output_dir, f'cluster_profile_{cluster}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cluster profile for cluster {cluster} to {output_path}")
    
    # Create a combined radar chart for all clusters
    # This might be too cluttered, so we'll create a separate chart for each cluster category
    
    # Group clusters by their category
    cluster_categories = {
        "High-Risk Regions": [0, 3],
        "Healthcare-Strained Regions": [1],
        "High-Mortality, High-Resource Region": [2],
        "High-Case, Low-Mortality Regions": [4, 5, 6, 7]
    }
    
    for category_name, category_clusters in cluster_categories.items():
        # Filter to clusters in this category
        category_profiles = normalized_profiles.loc[normalized_profiles.index.isin(category_clusters)]
        
        if category_profiles.empty:
            continue
        
        # Create a figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(profile_features)
        
        # Compute angle for each feature
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each cluster in this category
        for cluster in category_profiles.index:
            # Get the data for this cluster
            cluster_data = category_profiles.loc[cluster].values
            
            # Repeat the first value to close the polygon
            values = np.append(cluster_data, cluster_data[0])
            
            # Plot data
            ax.plot(angles, values, 'o-', linewidth=2, label=f"Cluster {cluster}: {cluster_labels.get(cluster, '')}")
            ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace('_', ' ').title() for f in profile_features], fontsize=8)
        
        # Draw y-axis lines from center to edge
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
        ax.set_ylim(0, 1)
        
        plt.title(f"Profiles for {category_name}")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save the figure
        output_path = os.path.join(output_dir, f'cluster_profiles_{category_name.replace(" ", "_").lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined cluster profiles for {category_name} to {output_path}")

def create_heatmap(df, cluster_col, features, cluster_labels, output_dir):
    """Create a heatmap showing the mean values of features for each cluster."""
    print(f"Creating feature heatmap...")
    
    # Calculate the mean of each feature for each cluster
    heatmap_df = df.groupby(cluster_col)[features].mean()
    
    # Create a copy with cluster labels as the index
    labeled_df = heatmap_df.copy()
    labeled_df.index = [cluster_labels.get(cluster, f"Cluster {cluster}") for cluster in labeled_df.index]
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Normalize the data for better visualization
    normalized_df = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min())
    
    # Create the heatmap
    sns.heatmap(
        normalized_df, 
        annot=True, 
        cmap="YlGnBu", 
        linewidths=.5,
        fmt=".2f",
        cbar_kws={'label': 'Normalized Value'}
    )
    
    plt.title("Feature Heatmap by Cluster (Normalized Values)")
    plt.xlabel("Features")
    plt.ylabel("Clusters")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'feature_heatmap_{cluster_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature heatmap to {output_path}")
    
    # Also create a non-normalized version with actual values
    plt.figure(figsize=(14, 10))
    
    # Create a custom formatter to handle different scales
    def custom_formatter(x, pos):
        if abs(x) < 0.01:
            return f"{x:.4f}"
        elif abs(x) < 1:
            return f"{x:.2f}"
        elif abs(x) < 10:
            return f"{x:.1f}"
        else:
            return f"{int(x)}"
    
    # Create the heatmap with actual values
    sns.heatmap(
        labeled_df, 
        annot=True, 
        cmap="YlGnBu", 
        linewidths=.5,
        fmt=".2f",
        cbar_kws={'label': 'Value'}
    )
    
    plt.title("Feature Heatmap by Cluster (Actual Values)")
    plt.xlabel("Features")
    plt.ylabel("Clusters")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'feature_heatmap_actual_{cluster_col}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved actual value feature heatmap to {output_path}")

def create_html_index(output_dir, cluster_labels):
    """Create an HTML index file to easily access all visualizations."""
    print("Creating HTML index file...")
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>COVID-19 Cluster Analysis Visualizations</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                .viz-group { margin-bottom: 30px; }
                ul { list-style-type: none; padding-left: 20px; }
                li { margin: 10px 0; }
                a { color: #2980b9; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .cluster-info { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .cluster-category { font-weight: bold; color: #2c3e50; }
                .viz-container { display: flex; flex-wrap: wrap; }
                .viz-item { margin: 10px; text-align: center; }
                .viz-item img { max-width: 300px; border: 1px solid #ddd; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>COVID-19 Cluster Analysis Visualizations</h1>
            
            <div class="cluster-info">
                <h2>Cluster Categories</h2>
                <p><span class="cluster-category">High-Risk Regions (Clusters 0 and 3):</span> 73 countries including Argentina, Australia, Bolivia, Indonesia with high case rates and mortality rates.</p>
                <p><span class="cluster-category">Healthcare-Strained Regions (Cluster 1):</span> 141 countries including Afghanistan, Albania, Algeria with high hospitalization/ICU utilization.</p>
                <p><span class="cluster-category">High-Mortality, High-Resource Region (Cluster 2):</span> United States as a single-country cluster with high mortality despite substantial healthcare resources.</p>
                <p><span class="cluster-category">High-Case, Low-Mortality Regions (Clusters 4, 5, 6, 7):</span> 19 countries including China, Italy, Hong Kong, Malaysia, and several island nations that successfully managed mortality despite high case rates.</p>
            </div>
        """)
        
        # Add all PNG files in the output directory
        f.write("""
            <h2>All Visualizations</h2>
            <div class="viz-container">
        """)
        
        # Get all PNG files
        png_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        png_files.sort()
        
        for png_file in png_files:
            # Create a nice title from the filename
            title = png_file.replace('.png', '').replace('_', ' ').title()
            
            f.write(f"""
                <div class="viz-item">
                    <a href="{png_file}" target="_blank">
                        <img src="{png_file}" alt="{title}">
                        <p>{title}</p>
                    </a>
                </div>
            """)
        
        f.write("""
            </div>
        </body>
        </html>
        """)
    
    print(f"Created index.html in the '{output_dir}' directory for easy access to all visualizations.")

def main():
    """Main function to generate visualizations."""
    print("Starting COVID-19 visualization generation...")
    
    # Load the data
    clusters_df, cleaned_df = load_data()
    
    # Define cluster labels
    cluster_labels = {
        0: "High-Risk Regions",
        1: "Healthcare-Strained Regions",
        2: "High-Mortality, High-Resource Region",
        3: "High-Risk Regions",
        4: "High-Case, Low-Mortality Regions",
        5: "High-Case, Low-Mortality Regions",
        6: "High-Case, Low-Mortality Regions",
        7: "High-Case, Low-Mortality Regions"
    }
    
    # Generate visualizations
    
    # 1. Cluster Distribution
    create_cluster_distribution(clusters_df, 'gmm_cluster', cluster_labels, output_dir)
    
    # 2. Cluster Scatter Plots
    create_cluster_scatter(clusters_df, 'gmm_cluster', 'case_rate_per_100k', 'mortality_rate_per_100k', cluster_labels, output_dir)
    create_cluster_scatter(clusters_df, 'gmm_cluster', 'vaccination_rate_pct', 'case_fatality_rate_pct', cluster_labels, output_dir)
    
    # 3. Cluster Comparison
    metrics = ['case_rate_per_100k', 'mortality_rate_per_100k', 'testing_rate_per_1k', 
               'vaccination_rate_pct', 'hospitalization_rate_per_100k', 'icu_rate_per_100k']
    create_cluster_comparison(clusters_df, 'gmm_cluster', metrics, cluster_labels, output_dir)
    
    # 4. Time Series
    countries = ['United States', 'United Kingdom', 'China', 'Italy', 'Brazil']
    start_date = pd.to_datetime('2020-03-01')
    end_date = pd.to_datetime('2022-12-31')
    
    # Check available metrics in cleaned_df
    time_series_metrics = [col for col in ['new_cases_per_million', 'new_deaths_per_million', 
                                          'total_cases_per_million', 'total_deaths_per_million'] 
                          if col in cleaned_df.columns]
    
    for metric in time_series_metrics:
        create_time_series(cleaned_df, countries, metric, start_date, end_date, output_dir)
    
    # 5. Cluster Profiles
    create_cluster_profiles(clusters_df, 'gmm_cluster', cluster_labels, output_dir)
    
    # 6. Feature Heatmap
    features = ['case_rate_per_100k', 'mortality_rate_per_100k', 'testing_rate_per_1k', 
                'vaccination_rate_pct', 'hospitalization_rate_per_100k', 'icu_rate_per_100k',
                'recovery_speed', 'case_fatality_rate_pct', 'vax_acceleration']
    features = [f for f in features if f in clusters_df.columns]
    create_heatmap(clusters_df, 'gmm_cluster', features, cluster_labels, output_dir)
    
    # 7. Create HTML index
    create_html_index(output_dir, cluster_labels)
    
    print("\nVisualization generation complete!")
    print(f"All visualizations have been saved to the '{output_dir}' directory.")
    print(f"Open {os.path.join(output_dir, 'index.html')} in your web browser to view all visualizations.")

if __name__ == "__main__":
    main()
