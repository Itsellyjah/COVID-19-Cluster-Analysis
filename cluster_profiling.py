#!/usr/bin/env python3
"""
COVID-19 Cluster Profiling Script
This script analyzes and profiles the clusters identified in the clustering analysis,
assigns meaningful labels, and derives public health insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

def load_data():
    """Load the clustering results and original features."""
    print("Loading clustering results and feature data...")
    
    # Load clustering results
    clusters_df = pd.read_csv('covid_clustering_results.csv')
    
    # We don't need to load the original features separately since the clustering results
    # already contain all the necessary features
    return clusters_df

def perform_statistical_tests(df, cluster_col, feature_cols):
    """Perform statistical tests to confirm significant differences between clusters."""
    print(f"\nPerforming statistical tests for {cluster_col}...")
    
    # Create a directory for the profiling visualizations if it doesn't exist
    viz_dir = 'cluster_profiling'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Get the number of clusters
    clusters = sorted(df[cluster_col].unique())
    
    # Filter feature_cols to only include columns that exist in the dataframe
    valid_features = [col for col in feature_cols if col in df.columns]
    
    # Initialize a DataFrame to store test results
    test_results = pd.DataFrame(index=valid_features, columns=['test_statistic', 'p_value', 'significant'])
    
    # Perform Kruskal-Wallis H-test for each feature
    # (Non-parametric alternative to ANOVA, doesn't assume normal distribution)
    for feature in valid_features:
        # Skip if the feature has all NaN values
        if df[feature].isna().all():
            test_results.loc[feature] = [np.nan, np.nan, False]
            continue
        
        # Prepare data for the test
        groups = [df[df[cluster_col] == cluster][feature].dropna() for cluster in clusters]
        
        # Skip if any group is empty
        if any(len(group) == 0 for group in groups):
            test_results.loc[feature] = [np.nan, np.nan, False]
            continue
        
        # Perform the test
        try:
            h_stat, p_value = stats.kruskal(*groups)
            significant = p_value < 0.05
            test_results.loc[feature] = [h_stat, p_value, significant]
        except Exception as e:
            print(f"Error performing Kruskal-Wallis test for {feature}: {e}")
            test_results.loc[feature] = [np.nan, np.nan, False]
    
    # Sort the results by p-value
    test_results = test_results.sort_values('p_value')
    
    # Print the significant features
    significant_features = test_results[test_results['significant']].index.tolist()
    print(f"Statistically significant features ({len(significant_features)}):")
    for feature in significant_features:
        print(f"  - {feature}: H = {test_results.loc[feature, 'test_statistic']:.2f}, p = {test_results.loc[feature, 'p_value']:.4f}")
    
    # Visualize the significant features across clusters
    if len(significant_features) > 0:
        # Select top features for visualization (to avoid cluttered plots)
        top_features = significant_features[:min(10, len(significant_features))]
        
        # Create box plots for each significant feature
        fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 4 * len(top_features)))
        if len(top_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(top_features):
            sns.boxplot(x=cluster_col, y=feature, data=df, ax=axes[i])
            axes[i].set_title(f"{feature} by Cluster")
            axes[i].set_xlabel("Cluster")
            axes[i].set_ylabel(feature)
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/{cluster_col}_significant_features.png')
        plt.close()
    
    return test_results, significant_features

def profile_clusters(df, cluster_col, feature_cols, significant_features=None):
    """Create detailed profiles for each cluster."""
    print(f"\nProfiling {cluster_col} clusters...")
    
    # Create a directory for the profiling visualizations if it doesn't exist
    viz_dir = 'cluster_profiling'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Get the number of clusters
    clusters = sorted(df[cluster_col].unique())
    
    # If significant features are provided, use them for profiling
    if significant_features is not None and len(significant_features) > 0:
        profile_features = [f for f in significant_features if f in feature_cols]
    else:
        profile_features = feature_cols
    
    # Calculate the mean of each feature for each cluster
    cluster_means = df.groupby(cluster_col)[profile_features].mean()
    
    # Calculate the global mean of each feature
    global_means = df[profile_features].mean()
    
    # Calculate the relative difference from the global mean
    # Handle division by zero by using np.where
    relative_diff = pd.DataFrame(index=cluster_means.index, columns=cluster_means.columns)
    
    for col in cluster_means.columns:
        # Avoid division by zero or very small numbers
        if abs(global_means[col]) > 0.001:
            relative_diff[col] = ((cluster_means[col] - global_means[col]) / global_means[col]) * 100
        else:
            # If global mean is very close to zero, use absolute difference instead
            relative_diff[col] = cluster_means[col] - global_means[col]
    
    # Replace infinities and extreme values with more reasonable bounds
    relative_diff = relative_diff.replace([np.inf, -np.inf], np.nan)
    
    # Cap extreme values to make visualization more meaningful
    for col in relative_diff.columns:
        relative_diff[col] = relative_diff[col].clip(lower=-500, upper=500)
    
    # Plot heatmap of relative differences
    plt.figure(figsize=(15, 10))
    sns.heatmap(relative_diff, annot=True, cmap='RdBu_r', center=0, fmt='.1f')
    plt.title(f'Relative Difference (%) from Global Mean for {cluster_col}')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/{cluster_col}_relative_diff_heatmap.png')
    plt.close()
    
    # Create cluster profiles
    cluster_profiles = {}
    
    for cluster in clusters:
        # Get the countries in this cluster
        countries = df[df[cluster_col] == cluster]['location'].tolist()
        
        # Calculate the size and percentage of this cluster
        size = len(countries)
        percentage = (size / len(df)) * 100
        
        # Get the top 3 features where this cluster is highest relative to the global mean
        if cluster in relative_diff.index:
            top_features = relative_diff.loc[cluster].sort_values(ascending=False).head(3)
            bottom_features = relative_diff.loc[cluster].sort_values().head(3)
        else:
            # Handle the case where the cluster is not in the index (e.g., for noise points)
            top_features = pd.Series()
            bottom_features = pd.Series()
        
        # Create a profile for this cluster
        profile = {
            'size': size,
            'percentage': percentage,
            'countries': countries,
            'top_features': top_features,
            'bottom_features': bottom_features
        }
        
        cluster_profiles[cluster] = profile
    
    return cluster_profiles, relative_diff

def assign_cluster_labels(cluster_profiles, relative_diff, cluster_col):
    """Assign meaningful labels to each cluster based on their profiles."""
    print(f"\nAssigning labels to {cluster_col} clusters...")
    
    # Define a function to generate a label based on the profile
    def generate_label(cluster, profile, rel_diff):
        if cluster == -1:
            return "Outlier Regions"
        
        # Get the top features for this cluster
        if cluster in rel_diff.index:
            # Filter out columns with _x and _y suffixes (duplicates from the merge)
            clean_cols = [col for col in rel_diff.columns if not (col.endswith('_x') or col.endswith('_y'))]
            
            # If we have clean columns, use them
            if clean_cols:
                rel_diff_clean = rel_diff[clean_cols]
                top_features = rel_diff_clean.loc[cluster].sort_values(ascending=False).head(3)
                bottom_features = rel_diff_clean.loc[cluster].sort_values().head(3)
            else:
                # Otherwise use columns ending with _x (original features)
                x_cols = [col for col in rel_diff.columns if col.endswith('_x')]
                rel_diff_x = rel_diff[x_cols]
                
                # Remove the _x suffix for readability
                rel_diff_x.columns = [col.replace('_x', '') for col in x_cols]
                
                top_features = rel_diff_x.loc[cluster].sort_values(ascending=False).head(3)
                bottom_features = rel_diff_x.loc[cluster].sort_values().head(3)
        else:
            return f"Cluster {cluster}"
        
        # Define feature categories (without suffixes)
        case_features = ['case_rate_per_100k', 'peak_case_rate_per_100k', 'recent_case_growth']
        death_features = ['mortality_rate_per_100k', 'case_fatality_rate_pct', 'recent_death_growth']
        response_features = ['testing_rate_per_1k', 'vaccination_rate_pct', 'days_to_50pct_vax', 'vax_acceleration']
        healthcare_features = ['hospitalization_rate_per_100k', 'icu_rate_per_100k']
        
        # Strip any suffixes from feature names for comparison
        def strip_suffix(feature):
            if feature.endswith('_x') or feature.endswith('_y'):
                return feature[:-2]
            return feature
        
        # Check which categories are prominent in the top features
        high_cases = any(strip_suffix(feature) in case_features for feature in top_features.index)
        high_deaths = any(strip_suffix(feature) in death_features for feature in top_features.index)
        high_response = any(strip_suffix(feature) in response_features for feature in top_features.index)
        high_healthcare = any(strip_suffix(feature) in healthcare_features for feature in top_features.index)
        
        # Check which categories are prominent in the bottom features
        low_cases = any(strip_suffix(feature) in case_features for feature in bottom_features.index)
        low_deaths = any(strip_suffix(feature) in death_features for feature in bottom_features.index)
        low_response = any(strip_suffix(feature) in response_features for feature in bottom_features.index)
        low_healthcare = any(strip_suffix(feature) in healthcare_features for feature in bottom_features.index)
        
        # Generate label based on the prominent categories
        if high_cases and high_deaths:
            return "High-Risk Regions"
        elif high_cases and low_deaths:
            return "High-Case, Low-Mortality Regions"
        elif low_cases and high_response:
            return "Controlled Outbreak Regions"
        elif high_response and low_deaths:
            return "Vaccination Leaders"
        elif high_healthcare:
            return "Healthcare-Strained Regions"
        elif low_response and high_deaths:
            return "Vulnerable Regions"
        elif low_cases and low_deaths:
            return "Low-Impact Regions"
        else:
            # For specific single-country clusters, assign more specific labels
            if len(profile['countries']) == 1:
                country = profile['countries'][0]
                
                # Special cases for notable countries
                if country == 'United States':
                    return "High-Mortality, High-Resource Region"
                elif country == 'Italy':
                    return "Healthcare-Strained Region"
                elif country == 'China':
                    return "Strict Containment Region"
                else:
                    return f"Unique Pattern Region: {country}"
            
            # For small clusters with island nations
            if len(profile['countries']) <= 15 and any('Island' in country for country in profile['countries']):
                return "Island Nation Cluster"
                
            return "Moderate-Impact Region"
    
    # Assign labels to each cluster
    cluster_labels = {}
    for cluster, profile in cluster_profiles.items():
        label = generate_label(cluster, profile, relative_diff)
        cluster_labels[cluster] = label
    
    return cluster_labels

def derive_public_health_insights(cluster_profiles, cluster_labels, relative_diff, cluster_col):
    """Derive public health insights for each cluster."""
    print(f"\nDeriving public health insights for {cluster_col} clusters...")
    
    # Define a function to generate insights based on the profile and label
    def generate_insights(cluster, profile, label, rel_diff):
        insights = []
        
        if cluster == -1:
            insights.append("These regions have unique COVID-19 patterns that require individualized analysis and response strategies.")
            return insights
        
        # Get the top and bottom features for this cluster
        if cluster in rel_diff.index:
            # Filter out columns with _x and _y suffixes (duplicates from the merge)
            clean_cols = [col for col in rel_diff.columns if not (col.endswith('_x') or col.endswith('_y'))]
            
            # If we have clean columns, use them
            if clean_cols:
                rel_diff_clean = rel_diff[clean_cols]
                top_features = rel_diff_clean.loc[cluster].sort_values(ascending=False).head(5)
                bottom_features = rel_diff_clean.loc[cluster].sort_values().head(5)
            else:
                # Otherwise use columns ending with _x (original features)
                x_cols = [col for col in rel_diff.columns if col.endswith('_x')]
                rel_diff_x = rel_diff[x_cols]
                
                # Remove the _x suffix for readability
                rel_diff_x.columns = [col.replace('_x', '') for col in x_cols]
                
                top_features = rel_diff_x.loc[cluster].sort_values(ascending=False).head(5)
                bottom_features = rel_diff_x.loc[cluster].sort_values().head(5)
        else:
            return [f"No specific insights available for Cluster {cluster}."]
        
        # Strip any suffixes from feature names for comparison
        def strip_suffix(feature):
            if feature.endswith('_x') or feature.endswith('_y'):
                return feature[:-2]
            return feature
            
        # Generate insights based on the label and features
        if "High-Risk" in label:
            insights.append("These regions face severe COVID-19 impact with both high case rates and mortality.")
            insights.append("Urgent intervention is needed including increased hospital capacity and accelerated vaccination campaigns.")
            insights.append("Strict containment measures may be necessary to reduce transmission.")
        
        elif "High-Case, Low-Mortality" in label:
            insights.append("Despite high case rates, these regions have managed to keep mortality relatively low.")
            insights.append("Focus should be on maintaining healthcare quality while continuing vaccination efforts.")
            insights.append("Investigate factors contributing to lower mortality despite high case rates for potential best practices.")
        
        elif "Controlled Outbreak" in label:
            insights.append("These regions have successfully contained COVID-19 spread through effective measures.")
            insights.append("Gradual relaxation of restrictions could be considered while maintaining vigilance.")
            insights.append("Document and share successful containment strategies with other regions.")
        
        elif "Vaccination Leaders" in label:
            insights.append("These regions have excelled in vaccination rollout, contributing to better outcomes.")
            insights.append("Focus on reaching remaining unvaccinated populations and consider booster programs.")
            insights.append("Share vaccination strategy best practices with other regions.")
        
        elif "Healthcare-Strained" in label or "Healthcare-Strained Region" in label:
            insights.append("Healthcare systems in these regions are under significant pressure.")
            insights.append("Urgent resource allocation is needed to expand hospital and ICU capacity.")
            insights.append("Consider requesting international assistance or resource sharing between regions.")
        
        elif "Vulnerable" in label:
            insights.append("These regions show concerning patterns of high mortality despite lower case rates.")
            insights.append("Urgent investigation into healthcare quality, testing adequacy, and vulnerable populations is needed.")
            insights.append("Targeted interventions for high-risk groups and healthcare system strengthening are priorities.")
        
        elif "Low-Impact" in label:
            insights.append("These regions have experienced relatively mild COVID-19 impact.")
            insights.append("Focus on maintaining vigilance and preparedness for potential future waves.")
            insights.append("Study factors contributing to lower impact for lessons applicable elsewhere.")
            
        elif "Island Nation Cluster" in label:
            insights.append("These small island nations show unique COVID-19 patterns, often with high case rates but lower mortality.")
            insights.append("Geographic isolation may have helped control initial spread but created challenges for medical resources.")
            insights.append("Focus on sustainable containment strategies that balance public health with economic needs in tourism-dependent economies.")
            
        elif "High-Mortality, High-Resource Region" in label:
            insights.append("Despite substantial healthcare resources, this region experienced high mortality rates.")
            insights.append("Analysis of policy implementation, healthcare access disparities, and population vulnerabilities is needed.")
            insights.append("Focus on addressing healthcare inequities and improving pandemic response coordination.")
            
        elif "Strict Containment Region" in label:
            insights.append("This region implemented extensive containment measures that successfully limited case spread.")
            insights.append("The zero-COVID approach demonstrated effectiveness but with significant economic and social costs.")
            insights.append("Focus on developing sustainable long-term strategies that balance public health with other societal needs.")
            
        elif "Moderate-Impact" in label:
            insights.append("These regions experienced moderate COVID-19 impact with mixed outcomes across different metrics.")
            insights.append("Targeted improvements in specific areas of weakness could yield significant benefits.")
            insights.append("Balanced approach to containment, vaccination, and healthcare capacity is recommended.")
        
        # Add feature-specific insights
        case_rate_features = ['case_rate_per_100k', 'case_rate_per_100k_x', 'case_rate_per_100k_y']
        mortality_features = ['mortality_rate_per_100k', 'mortality_rate_per_100k_x', 'mortality_rate_per_100k_y']
        testing_features = ['testing_rate_per_1k', 'testing_rate_per_1k_x', 'testing_rate_per_1k_y']
        vaccination_features = ['vaccination_rate_pct', 'vaccination_rate_pct_x', 'vaccination_rate_pct_y']
        vax_speed_features = ['days_to_50pct_vax', 'days_to_50pct_vax_x', 'days_to_50pct_vax_y']
        hospital_features = ['hospitalization_rate_per_100k', 'hospitalization_rate_per_100k_x', 'hospitalization_rate_per_100k_y', 
                            'icu_rate_per_100k', 'icu_rate_per_100k_x', 'icu_rate_per_100k_y']
        
        if any(feature in top_features for feature in case_rate_features):
            insights.append("High case rates indicate ongoing community transmission requiring continued containment efforts.")
        
        if any(feature in top_features for feature in mortality_features):
            insights.append("High mortality rates suggest the need for improved clinical protocols and healthcare capacity.")
        
        if any(feature in bottom_features for feature in testing_features):
            insights.append("Low testing rates may be masking the true extent of the outbreak; expanded testing is recommended.")
        
        if any(feature in bottom_features for feature in vaccination_features):
            insights.append("Low vaccination rates highlight the need for targeted vaccination campaigns and addressing hesitancy.")
        
        if any(feature in top_features for feature in vax_speed_features):
            insights.append("Slow vaccine rollout indicates the need for improved distribution systems and public outreach.")
        
        if any(feature in top_features for feature in hospital_features):
            insights.append("High hospitalization/ICU rates indicate healthcare system strain requiring capacity expansion.")
        
        return insights
    
    # Generate insights for each cluster
    cluster_insights = {}
    for cluster, profile in cluster_profiles.items():
        label = cluster_labels[cluster]
        insights = generate_insights(cluster, profile, label, relative_diff)
        cluster_insights[cluster] = insights
    
    return cluster_insights

def create_region_profiles(cluster_profiles, cluster_labels, cluster_insights, cluster_col):
    """Create comprehensive region profiles with narratives."""
    print(f"\nCreating region profiles for {cluster_col} clusters...")
    
    # Create a directory for the profiling documents if it doesn't exist
    profile_dir = 'cluster_profiles'
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    
    # Create a markdown file for the region profiles
    with open(f'{profile_dir}/{cluster_col}_region_profiles.md', 'w') as f:
        f.write(f"# COVID-19 Region Profiles: {cluster_col}\n\n")
        f.write("This document provides detailed profiles of the different region types identified through cluster analysis of COVID-19 data.\n\n")
        
        # Write profiles for each cluster
        for cluster in sorted(cluster_profiles.keys()):
            profile = cluster_profiles[cluster]
            label = cluster_labels[cluster]
            insights = cluster_insights[cluster]
            
            f.write(f"## {label} (Cluster {cluster})\n\n")
            
            # Basic information
            f.write(f"**Size:** {profile['size']} countries ({profile['percentage']:.1f}% of total)\n\n")
            
            # Representative countries
            f.write("**Representative Countries:** ")
            if len(profile['countries']) > 5:
                f.write(", ".join(profile['countries'][:5]) + f", and {len(profile['countries']) - 5} others\n\n")
            else:
                f.write(", ".join(profile['countries']) + "\n\n")
            
            # Distinctive features
            f.write("**Distinctive Characteristics:**\n\n")
            
            if not profile['top_features'].empty:
                f.write("*Higher than average:*\n")
                for feature, value in profile['top_features'].items():
                    f.write(f"- {feature}: {value:.1f}% above global average\n")
                f.write("\n")
            
            if not profile['bottom_features'].empty:
                f.write("*Lower than average:*\n")
                for feature, value in profile['bottom_features'].items():
                    f.write(f"- {feature}: {abs(value):.1f}% below global average\n")
                f.write("\n")
            
            # Public health insights
            f.write("**Public Health Insights:**\n\n")
            for insight in insights:
                f.write(f"- {insight}\n")
            f.write("\n")
            
            # Narrative
            f.write("**Narrative Summary:**\n\n")
            
            # Generate a narrative based on the profile
            narrative = generate_narrative(cluster, profile, label, insights)
            f.write(narrative + "\n\n")
            
            f.write("---\n\n")
    
    print(f"Region profiles saved to {profile_dir}/{cluster_col}_region_profiles.md")
    
    return f'{profile_dir}/{cluster_col}_region_profiles.md'

def generate_narrative(cluster, profile, label, insights):
    """Generate a narrative summary for a cluster."""
    if cluster == -1:
        return ("These regions exhibit unique COVID-19 patterns that don't fit well with other clusters. "
                "Each requires individualized analysis to understand its specific challenges and appropriate responses. "
                "The diversity of these outliers highlights the complex and multifaceted nature of the pandemic's impact across different contexts.")
    
    # Start with a basic description
    narrative = f"The {label} "
    
    if profile['size'] == 1:
        narrative += f"consists of a single country, {profile['countries'][0]}. "
    elif profile['size'] <= 5:
        narrative += f"encompasses a small group of {profile['size']} countries including {', '.join(profile['countries'])}. "
    elif profile['size'] <= 20:
        narrative += f"represents a moderate-sized group of {profile['size']} countries, including {', '.join(profile['countries'][:3])} and others. "
    else:
        narrative += f"represents a large group of {profile['size']} countries ({profile['percentage']:.1f}% of all regions analyzed). "
    
    # Add distinctive characteristics
    if not profile['top_features'].empty:
        top_feature = profile['top_features'].index[0]
        top_value = profile['top_features'].iloc[0]
        
        narrative += f"This group is particularly characterized by {top_feature} levels that are {abs(top_value):.1f}% "
        narrative += "above the global average. "
    
    # Add a summary of insights
    if insights:
        narrative += insights[0] + " "
        if len(insights) > 1:
            narrative += insights[1] + " "
    
    # Add a concluding statement
    if "High-Risk" in label:
        narrative += ("These regions face the most severe challenges from COVID-19 and require comprehensive and urgent interventions "
                     "to address both the immediate healthcare crisis and long-term public health implications.")
    elif "Controlled" in label:
        narrative += ("The success of these regions in containing COVID-19 provides valuable lessons for pandemic response, "
                     "though continued vigilance remains essential to maintain their favorable position.")
    elif "Vaccination" in label:
        narrative += ("The effective vaccination campaigns in these regions demonstrate the impact of proactive public health measures "
                     "and provide a pathway for other regions to follow as they work to control the pandemic.")
    elif "Vulnerable" in label:
        narrative += ("The concerning patterns in these regions highlight the importance of addressing underlying healthcare system "
                     "weaknesses and protecting vulnerable populations as essential components of pandemic response.")
    elif "Low-Impact" in label:
        narrative += ("While these regions have experienced relatively mild impacts from COVID-19, understanding the factors "
                     "contributing to their resilience could provide valuable insights for global pandemic preparedness.")
    else:
        narrative += ("The specific characteristics of this group highlight the importance of tailored approaches to COVID-19 "
                     "response that address the unique challenges and leverage the particular strengths of each region.")
    
    return narrative

def link_to_outcomes(df, cluster_col):
    """Link clusters to real-world outcomes using available data."""
    print(f"\nLinking {cluster_col} clusters to real-world outcomes...")
    
    # Create a directory for the outcome visualizations if it doesn't exist
    viz_dir = 'cluster_profiling'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Define potential outcome variables
    outcome_vars = [
        'case_fatality_rate_pct',  # Healthcare system effectiveness
        'hospitalization_rate_per_100k',  # Healthcare system strain
        'icu_rate_per_100k',  # Critical care capacity
        'days_to_50pct_vax'  # Vaccination campaign effectiveness
    ]
    
    # Check which outcome variables are available
    available_outcomes = [var for var in outcome_vars if var in df.columns]
    
    if not available_outcomes:
        print("No outcome variables available for analysis.")
        return None
    
    # Create a figure for the outcome analysis
    fig, axes = plt.subplots(len(available_outcomes), 1, figsize=(12, 5 * len(available_outcomes)))
    if len(available_outcomes) == 1:
        axes = [axes]
    
    # Analyze each outcome variable
    outcome_analysis = {}
    
    for i, outcome in enumerate(available_outcomes):
        # Create a box plot for this outcome by cluster
        sns.boxplot(x=cluster_col, y=outcome, data=df, ax=axes[i])
        axes[i].set_title(f"{outcome} by Cluster")
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel(outcome)
        
        # Perform Kruskal-Wallis test to check for significant differences
        try:
            groups = [df[df[cluster_col] == cluster][outcome].dropna() for cluster in sorted(df[cluster_col].unique())]
            # Filter out empty groups
            groups = [group for group in groups if len(group) > 0]
            
            if len(groups) >= 2:  # Need at least 2 groups for the test
                h_stat, p_value = stats.kruskal(*groups)
                significant = p_value < 0.05
            else:
                h_stat, p_value, significant = np.nan, np.nan, False
        except Exception as e:
            print(f"Error performing Kruskal-Wallis test for {outcome}: {e}")
            h_stat, p_value, significant = np.nan, np.nan, False
        
        # Store the analysis
        outcome_analysis[outcome] = {
            'h_statistic': h_stat,
            'p_value': p_value,
            'significant': significant
        }
        
        # Add the test result to the plot
        if not np.isnan(p_value):
            axes[i].annotate(
                f"Kruskal-Wallis: H = {h_stat:.2f}, p = {p_value:.4f}{' *' if significant else ''}",
                xy=(0.02, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/{cluster_col}_outcome_analysis.png')
    plt.close()
    
    # Print the outcome analysis
    print("\nOutcome Analysis:")
    for outcome, analysis in outcome_analysis.items():
        if analysis['significant']:
            print(f"  - {outcome}: Significant differences between clusters (H = {analysis['h_statistic']:.2f}, p = {analysis['p_value']:.4f})")
        elif not np.isnan(analysis['p_value']):
            print(f"  - {outcome}: No significant differences between clusters (H = {analysis['h_statistic']:.2f}, p = {analysis['p_value']:.4f})")
        else:
            print(f"  - {outcome}: Could not perform statistical test")
    
    return outcome_analysis

def main():
    """Main function to run the cluster profiling analysis."""
    # Load the data
    df = load_data()
    
    # Print column names for debugging
    print(f"\nDataframe has {len(df.columns)} columns")
    
    # Define the feature columns (excluding metadata and cluster assignments)
    feature_cols = [col for col in df.columns if col not in [
        'location', 'population', 'kmeans_cluster', 'hierarchical_cluster', 
        'dbscan_cluster', 'gmm_cluster'
    ]]
    
    print(f"Found {len(feature_cols)} feature columns")
    
    # Define the clustering methods to analyze
    cluster_cols = ['gmm_cluster']  # Using GMM as it provided the most nuanced clusters
    
    # Process each clustering method
    for cluster_col in cluster_cols:
        if cluster_col not in df.columns:
            print(f"Clustering column {cluster_col} not found in the data.")
            continue
        
        # Perform statistical tests
        test_results, significant_features = perform_statistical_tests(df, cluster_col, feature_cols)
        
        # Profile the clusters
        cluster_profiles, relative_diff = profile_clusters(df, cluster_col, feature_cols, significant_features)
        
        # Assign labels to clusters
        cluster_labels = assign_cluster_labels(cluster_profiles, relative_diff, cluster_col)
        
        # Derive public health insights
        cluster_insights = derive_public_health_insights(cluster_profiles, cluster_labels, relative_diff, cluster_col)
        
        # Create region profiles
        profile_path = create_region_profiles(cluster_profiles, cluster_labels, cluster_insights, cluster_col)
        
        # Link clusters to outcomes
        outcome_analysis = link_to_outcomes(df, cluster_col)
        
        # Print the cluster labels
        print("\nCluster Labels:")
        for cluster, label in sorted(cluster_labels.items()):
            countries = cluster_profiles[cluster]['countries']
            if len(countries) > 3:
                countries_str = f"{', '.join(countries[:3])} and {len(countries) - 3} others"
            else:
                countries_str = ', '.join(countries)
            print(f"  - Cluster {cluster}: {label} ({len(countries)} countries, e.g., {countries_str})")
    
    print("\nCluster profiling completed successfully!")
    print(f"Detailed region profiles are available in the cluster_profiles directory.")
    print(f"Visualizations are available in the cluster_profiling directory.")

if __name__ == "__main__":
    main()
