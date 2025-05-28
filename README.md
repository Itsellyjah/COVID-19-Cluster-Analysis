# COVID-19 Cluster Analysis

This repository contains a comprehensive analysis of COVID-19 data using clustering techniques to identify patterns in how countries responded to the pandemic.

## Project Structure

- **clean_covid_data.py**: Script to clean the original COVID-19 dataset
- **feature_engineering.py**: Script to create features for clustering analysis
- **clustering_analysis.py**: Script to perform clustering using various algorithms
- **cluster_profiling.py**: Script to analyze and interpret clusters
- **simple_visualizations.py**: Script to generate visualizations of the clustering results
- **prepare_tableau_data.py**: Script to prepare data for Tableau visualization

## Visualizations

The `visualization_outputs` directory contains all visualizations generated from the analysis, including:

- Cluster distribution charts
- Scatter plots comparing key metrics
- Time series visualizations
- Cluster profile radar charts
- Feature heatmaps

## Data

The analysis is based on the "Our World in Data" COVID-19 dataset, which can be downloaded from: https://github.com/owid/covid-19-data/tree/master/public/data

Due to size constraints, the original data files are not included in this repository. To run the scripts, download the dataset and place it in the project directory.

## Cluster Analysis Results

The analysis identified four main categories of regions based on their COVID-19 response patterns:

1. **High-Risk Regions** (73 countries): High case and mortality rates
2. **Healthcare-Strained Regions** (141 countries): High hospitalization/ICU utilization
3. **High-Mortality, High-Resource Region** (United States): Unique pattern despite resources
4. **High-Case, Low-Mortality Regions** (19 countries): Successfully managed mortality despite high case rates
