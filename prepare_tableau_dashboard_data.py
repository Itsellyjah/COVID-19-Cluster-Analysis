#!/usr/bin/env python3
"""
Prepare Data for Tableau Dashboard - COVID-19 Cluster Analysis

This script prepares the necessary datasets for creating an interactive Tableau dashboard
to visualize the results of the COVID-19 cluster analysis project.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def load_data():
    """Load all necessary datasets for the Tableau dashboard."""
    print("Loading datasets...")
    
    # Load clustering results
    clustering_results = pd.read_csv('covid_clustering_results.csv')
    print(f"Loaded clustering results for {len(clustering_results)} countries")
    
    # Load cleaned COVID-19 time series data
    cleaned_covid_data = pd.read_csv('data/cleaned_covid_data.csv')
    cleaned_covid_data['date'] = pd.to_datetime(cleaned_covid_data['date'])
    print(f"Loaded cleaned COVID-19 data with {len(cleaned_covid_data)} records")
    
    return clustering_results, cleaned_covid_data

def prepare_main_dataset(clustering_results):
    """Prepare the main dataset with clustering results and features for each country."""
    print("\nPreparing main dataset for Tableau...")
    
    # Create a copy of the clustering results
    main_df = clustering_results.copy()
    
    # Add cluster labels
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
    
    # Add cluster category column (grouping similar clusters)
    cluster_categories = {
        0: "High-Risk Regions",
        1: "Healthcare-Strained Regions",
        2: "High-Mortality, High-Resource Region",
        3: "High-Risk Regions",
        4: "High-Case, Low-Mortality Regions",
        5: "High-Case, Low-Mortality Regions",
        6: "High-Case, Low-Mortality Regions",
        7: "High-Case, Low-Mortality Regions"
    }
    
    # We'll use GMM clusters as they provided the most nuanced results
    main_df['cluster_label'] = main_df['gmm_cluster'].map(cluster_labels)
    main_df['cluster_category'] = main_df['gmm_cluster'].map(cluster_categories)
    
    # Calculate additional metrics that might be useful for the dashboard
    # (if they don't already exist in the dataset)
    if 'case_to_test_ratio' not in main_df.columns and 'testing_rate_per_1k' in main_df.columns and 'case_rate_per_100k' in main_df.columns:
        main_df['case_to_test_ratio'] = (main_df['case_rate_per_100k'] / 100) / main_df['testing_rate_per_1k']
    
    # Add continent information if available
    try:
        # Try to get continent information from the original OWID dataset
        owid_data = pd.read_csv('data/owid-covid-data.csv')
        continent_map = owid_data[['location', 'continent']].drop_duplicates().set_index('location')['continent']
        main_df['continent'] = main_df['location'].map(continent_map)
        print("Added continent information to the main dataset")
    except Exception as e:
        print(f"Could not add continent information: {e}")
    
    # Create tableau_dashboard directory if it doesn't exist
    tableau_dir = 'tableau_dashboard'
    if not os.path.exists(tableau_dir):
        os.makedirs(tableau_dir)
    
    # Save the main dataset
    output_path = os.path.join(tableau_dir, 'tableau_dashboard_main.csv')
    main_df.to_csv(output_path, index=False)
    print(f"Saved main dataset with {len(main_df)} countries to {output_path}")
    
    return main_df

def prepare_time_series_dataset(cleaned_covid_data, clustering_results):
    """Prepare a time series dataset for temporal analysis in Tableau."""
    print("\nPreparing time series dataset for Tableau...")
    
    # Create a mapping from country to cluster
    country_to_cluster = clustering_results[['location', 'gmm_cluster']].set_index('location')['gmm_cluster'].to_dict()
    
    # Create a mapping from country to cluster label
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
    
    country_to_cluster_label = {
        country: cluster_labels.get(cluster, f"Cluster {cluster}")
        for country, cluster in country_to_cluster.items()
    }
    
    # Create a mapping from country to cluster category
    cluster_categories = {
        0: "High-Risk Regions",
        1: "Healthcare-Strained Regions",
        2: "High-Mortality, High-Resource Region",
        3: "High-Risk Regions",
        4: "High-Case, Low-Mortality Regions",
        5: "High-Case, Low-Mortality Regions",
        6: "High-Case, Low-Mortality Regions",
        7: "High-Case, Low-Mortality Regions"
    }
    
    country_to_cluster_category = {
        country: cluster_categories.get(cluster, f"Cluster {cluster}")
        for country, cluster in country_to_cluster.items()
    }
    
    # Filter the time series data to include only countries in the clustering results
    time_series_df = cleaned_covid_data[cleaned_covid_data['location'].isin(country_to_cluster.keys())].copy()
    
    # Add cluster information to the time series data
    time_series_df['gmm_cluster'] = time_series_df['location'].map(country_to_cluster)
    time_series_df['cluster_label'] = time_series_df['location'].map(country_to_cluster_label)
    time_series_df['cluster_category'] = time_series_df['location'].map(country_to_cluster_category)
    
    # Calculate 7-day rolling averages for key metrics
    metrics_to_smooth = [
        'new_cases_per_million', 'new_deaths_per_million',
        'total_cases_per_million', 'total_deaths_per_million'
    ]
    
    for metric in metrics_to_smooth:
        if metric in time_series_df.columns:
            time_series_df[f'{metric}_7day_avg'] = time_series_df.groupby('location')[metric].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
    
    # Save the time series dataset
    # To avoid creating a very large file, we'll sample the data
    # Let's keep data at weekly intervals for a more manageable file size
    time_series_df['week'] = time_series_df['date'].dt.isocalendar().week
    time_series_df['year'] = time_series_df['date'].dt.isocalendar().year
    weekly_data = time_series_df.groupby(['location', 'year', 'week']).agg({
        'date': 'first',  # Keep the first date of each week
        'gmm_cluster': 'first',
        'cluster_label': 'first',
        'cluster_category': 'first',
        'new_cases_per_million': 'mean',
        'new_deaths_per_million': 'mean',
        'total_cases_per_million': 'last',
        'total_deaths_per_million': 'last',
        'new_cases_per_million_7day_avg': 'mean',
        'new_deaths_per_million_7day_avg': 'mean'
    }).reset_index()
    
    # Save the weekly time series dataset
    output_path = os.path.join(tableau_dir, 'tableau_dashboard_time_series.csv')
    weekly_data.to_csv(output_path, index=False)
    print(f"Saved time series dataset with {len(weekly_data)} records to {output_path}")
    
    return weekly_data

def prepare_feature_comparison_dataset(main_df):
    """Prepare a feature comparison dataset for detailed feature analysis in Tableau."""
    print("\nPreparing feature comparison dataset for Tableau...")
    
    # Identify the feature columns (excluding metadata and cluster assignments)
    feature_cols = [col for col in main_df.columns if col not in [
        'location', 'population', 'kmeans_cluster', 'hierarchical_cluster', 
        'dbscan_cluster', 'gmm_cluster', 'cluster_label', 'cluster_category',
        'continent', 'case_to_test_ratio'
    ]]
    
    # Create a long-format dataset for easier comparison in Tableau
    feature_comparison_rows = []
    
    for _, row in main_df.iterrows():
        country = row['location']
        cluster = row['gmm_cluster']
        cluster_label = row['cluster_label']
        cluster_category = row['cluster_category']
        
        for feature in feature_cols:
            feature_comparison_rows.append({
                'location': country,
                'gmm_cluster': cluster,
                'cluster_label': cluster_label,
                'cluster_category': cluster_category,
                'feature': feature,
                'value': row[feature]
            })
    
    feature_comparison_df = pd.DataFrame(feature_comparison_rows)
    
    # Save the feature comparison dataset
    output_path = os.path.join(tableau_dir, 'tableau_dashboard_feature_comparison.csv')
    feature_comparison_df.to_csv(output_path, index=False)
    print(f"Saved feature comparison dataset with {len(feature_comparison_df)} records to {output_path}")
    
    return feature_comparison_df

def main():
    """Main function to prepare all datasets for the Tableau dashboard."""
    print("Starting preparation of datasets for Tableau dashboard...")
    
    # Load the data
    clustering_results, cleaned_covid_data = load_data()
    
    # Prepare the main dataset
    main_df = prepare_main_dataset(clustering_results)
    
    # Prepare the time series dataset
    time_series_df = prepare_time_series_dataset(cleaned_covid_data, clustering_results)
    
    # Prepare the feature comparison dataset
    feature_comparison_df = prepare_feature_comparison_dataset(main_df)
    
    print("\nAll datasets for the Tableau dashboard have been prepared successfully!")
    print(f"All files have been saved to the '{tableau_dir}' directory.")
    print("You can now use these datasets to create your interactive dashboard in Tableau:")
    print(f"1. {tableau_dir}/tableau_dashboard_main.csv - Main dataset with clustering results and features")
    print(f"2. {tableau_dir}/tableau_dashboard_time_series.csv - Time series dataset for temporal analysis")
    print(f"3. {tableau_dir}/tableau_dashboard_feature_comparison.csv - Feature comparison dataset for detailed analysis")

if __name__ == "__main__":
    main()
