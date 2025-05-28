#!/usr/bin/env python3
"""
Prepare COVID-19 data for Tableau visualization
This script formats the clustering results and time series data for easy import into Tableau.
"""

import pandas as pd
import numpy as np
import os

def prepare_cluster_data():
    """Prepare the cluster data for Tableau."""
    print("Preparing cluster data for Tableau...")
    
    # Load clustering results
    clusters_df = pd.read_csv('covid_clustering_results.csv')
    
    # Define the cluster labels
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
    
    # Add cluster labels
    clusters_df['cluster_label'] = clusters_df['gmm_cluster'].map(cluster_labels)
    
    # Group clusters into broader categories for easier visualization
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
    
    clusters_df['cluster_category'] = clusters_df['gmm_cluster'].map(cluster_categories)
    
    # Save the data for Tableau
    clusters_df.to_csv('tableau_cluster_data.csv', index=False)
    print(f"Saved cluster data to tableau_cluster_data.csv")
    
    return clusters_df

def prepare_time_series_data():
    """Prepare time series data for Tableau."""
    print("Preparing time series data for Tableau...")
    
    # Load cleaned data
    cleaned_df = pd.read_csv('cleaned_covid_data.csv')
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
    
    # Load cluster data to get cluster assignments
    clusters_df = pd.read_csv('tableau_cluster_data.csv')
    
    # Create a mapping of countries to clusters
    country_to_cluster = dict(zip(clusters_df['location'], clusters_df['cluster_category']))
    
    # Add cluster information to the time series data
    cleaned_df['cluster_category'] = cleaned_df['location'].map(country_to_cluster)
    
    # Filter to relevant columns for time series analysis
    ts_columns = [
        'iso_code', 'location', 'date', 'total_cases', 'new_cases', 
        'total_deaths', 'new_deaths', 'total_cases_per_million', 
        'new_cases_per_million', 'total_deaths_per_million', 
        'new_deaths_per_million', 'cluster_category'
    ]
    
    # Keep only columns that exist in the dataframe
    ts_columns = [col for col in ts_columns if col in cleaned_df.columns]
    
    ts_df = cleaned_df[ts_columns].copy()
    
    # Add month and year columns for easier aggregation in Tableau
    ts_df['year'] = ts_df['date'].dt.year
    ts_df['month'] = ts_df['date'].dt.month
    ts_df['quarter'] = ts_df['date'].dt.quarter
    
    # Save the time series data for Tableau
    # To avoid creating a huge file, let's sample the data
    # Take data for every 7 days (weekly data)
    ts_df['week'] = ts_df['date'].dt.isocalendar().week
    weekly_df = ts_df.groupby(['location', 'year', 'week']).last().reset_index()
    
    # Save the weekly data for Tableau
    weekly_df.to_csv('tableau_time_series_data.csv', index=False)
    print(f"Saved time series data to tableau_time_series_data.csv")
    
    return weekly_df

def prepare_feature_data():
    """Prepare feature data for Tableau."""
    print("Preparing feature data for Tableau...")
    
    # Load the original features
    features_df = pd.read_csv('covid_features_original.csv')
    
    # Load cluster data to get cluster assignments
    clusters_df = pd.read_csv('tableau_cluster_data.csv')
    
    # Merge the datasets
    merged_df = pd.merge(clusters_df[['location', 'gmm_cluster', 'cluster_category']], 
                         features_df, on='location', how='inner')
    
    # Save the feature data for Tableau
    merged_df.to_csv('tableau_feature_data.csv', index=False)
    print(f"Saved feature data to tableau_feature_data.csv")
    
    return merged_df

def main():
    """Main function to prepare data for Tableau."""
    # Create a directory for Tableau data if it doesn't exist
    tableau_dir = 'tableau_data'
    if not os.path.exists(tableau_dir):
        os.makedirs(tableau_dir)
    
    # Prepare the datasets
    cluster_df = prepare_cluster_data()
    time_series_df = prepare_time_series_data()
    feature_df = prepare_feature_data()
    
    print("\nData preparation for Tableau completed!")
    print("You can now import these CSV files into Tableau to create visualizations.")
    print("\nSuggested Tableau Visualizations:")
    print("1. Choropleth Map: Use tableau_cluster_data.csv to create a world map colored by cluster_category")
    print("2. Time Series: Use tableau_time_series_data.csv to create line charts of cases/deaths over time by cluster")
    print("3. Feature Comparison: Use tableau_feature_data.csv to create bar charts comparing key metrics across clusters")
    print("4. Scatter Plot: Create a scatter plot of case_rate_per_100k vs mortality_rate_per_100k colored by cluster")
    print("5. Dashboard: Combine these visualizations into an interactive dashboard with filters for countries and time periods")

if __name__ == "__main__":
    main()
