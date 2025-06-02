# COVID-19 Cluster Analysis

This repository contains a comprehensive analysis of COVID-19 data using clustering techniques to identify patterns in how countries responded to the pandemic.

## Project Structure

- **clean_covid_data.py**: Script to clean the original COVID-19 dataset
- **feature_engineering.py**: Script to create features for clustering analysis
- **clustering_analysis.py**: Script to perform clustering using various algorithms
- **cluster_profiling.py**: Script to analyze and interpret clusters
- **simple_visualizations.py**: Script to generate visualizations of the clustering results
- **prepare_tableau_dashboard_data.py**: Script to prepare data for the interactive Tableau dashboard
- **COVID19_Cluster_Analysis_Report.md**: Comprehensive report of analysis findings and implications

## Visualizations

The `visualization_outputs` directory contains all visualizations generated from the analysis, including:

- Cluster distribution charts
- Scatter plots comparing key metrics
- Time series visualizations
- Cluster profile radar charts
- Feature heatmaps

## Interactive Tableau Dashboard

An interactive Tableau dashboard has been created to visualize the results of this analysis. The dashboard provides:

- Interactive global map showing cluster distribution by country
- Comparative analysis of key metrics across the four cluster categories
- Time series trends showing COVID-19 progression by cluster
- Detailed feature analysis showing what distinguishes each cluster
- Country-level data exploration with cluster context
- Public health implications based on cluster membership

### Dashboard Datasets

The following datasets have been prepared specifically for the Tableau dashboard and are stored in the `tableau_dashboard` folder:

1. **tableau_dashboard_main.csv**: Main dataset with clustering results and features for each country
2. **tableau_dashboard_time_series.csv**: Weekly time series data for temporal analysis
3. **tableau_dashboard_feature_comparison.csv**: Feature comparison dataset in long format for detailed analysis

### Creating Your Own Dashboard

To create or modify the Tableau dashboard:

1. Run `prepare_tableau_dashboard_data.py` to generate the necessary datasets (saved to the `tableau_dashboard` folder)
2. Import these datasets from the `tableau_dashboard` folder into Tableau
3. Use the structure outlined in the script comments to build your visualizations
4. Connect the visualizations with interactive filters and actions
5. Publish to Tableau Public or Tableau Server to share your insights

View the dashboard on Tableau Public: [COVID-19 Cluster Analysis Dashboard](https://public.tableau.com/views/COVID-19ClusterAnalysisDashboard/Dashboard1?:language=en-GB&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Data

The analysis is based on the "Our World in Data" COVID-19 dataset, which can be downloaded from: https://github.com/owid/covid-19-data/tree/master/public/data

Due to size constraints, the original data files are not included in this repository. To run the scripts, download the dataset and place it in the `data` directory.

## Cluster Analysis Results

The analysis identified four main categories of regions based on their COVID-19 response patterns:

1. **High-Risk Regions** (73 countries): High case and mortality rates with significant healthcare burden
2. **Healthcare-Strained Regions** (141 countries): High hospitalization/ICU utilization relative to resources
3. **High-Mortality, High-Resource Region** (United States): Unique pattern showing high mortality despite abundant resources
4. **High-Case, Low-Mortality Regions** (19 countries): Successfully managed mortality despite high case rates through effective healthcare management

For a detailed analysis of these clusters and their public health implications, see the comprehensive report in `COVID19_Cluster_Analysis_Report.md`.
