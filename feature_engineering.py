#!/usr/bin/env python3
"""
COVID-19 Feature Engineering Script
This script creates features for clustering analysis from the cleaned COVID-19 dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the cleaned COVID-19 dataset."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def extract_external_data(df):
    """Extract GDP, population density, and other external data from the dataset."""
    print("Extracting external data...")
    
    # Filter out aggregate regions like 'World', 'Europe', etc. to focus on countries
    # First, identify which locations are actual countries (have iso_code with length 3)
    countries_mask = df['iso_code'].str.len() == 3
    
    # Get the latest record for each country to extract static data
    latest_records = df[countries_mask].groupby('location').last().reset_index()
    
    # Extract external data columns if they exist
    external_columns = [
        'location', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older',
        'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence',
        'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
        'life_expectancy', 'human_development_index'
    ]
    
    # Check which columns actually exist in the dataset
    existing_columns = [col for col in external_columns if col in df.columns]
    
    # Create a dataframe with the external data
    external_data = latest_records[existing_columns].copy()
    
    print(f"Extracted {len(existing_columns)} external data columns for {len(external_data)} countries.")
    return external_data

def create_features(df):
    """Create features for clustering analysis."""
    print("Creating features for clustering...")
    
    # Filter out aggregate regions like 'World', 'Europe', etc. to focus on countries
    # First, identify which locations are actual countries (have iso_code with length 3)
    countries_mask = df['iso_code'].str.len() == 3
    countries_list = df.loc[countries_mask, 'location'].unique()
    
    # Filter the dataframe to include only actual countries
    df_countries = df[df['location'].isin(countries_list)].copy()
    
    print(f"Filtered to {len(countries_list)} individual countries.")
    
    # Get the latest available data for each country
    latest_date = df_countries['date'].max()
    print(f"Latest date in the dataset: {latest_date}")
    
    # We'll use data from the last month of the dataset for stability
    one_month_ago = latest_date - pd.Timedelta(days=30)
    df_recent = df_countries[df_countries['date'] >= one_month_ago].copy()
    
    # Group by country and aggregate
    df_agg = df_recent.groupby('location').agg({
        'total_cases': 'max',
        'total_deaths': 'max',
        'total_tests': 'max',
        'people_fully_vaccinated': 'max',
        'population': 'first',
        'hosp_patients': 'mean',
        'icu_patients': 'mean',
        'positive_rate': 'mean',
        'new_cases': 'mean',
        'new_deaths': 'mean'
    }).reset_index()
    
    # Create the requested features
    
    # 1. Case Rate: Cases per 100,000 population
    df_agg['case_rate_per_100k'] = (df_agg['total_cases'] / df_agg['population']) * 100000
    
    # 2. Mortality Rate: Deaths per 100,000 population
    df_agg['mortality_rate_per_100k'] = (df_agg['total_deaths'] / df_agg['population']) * 100000
    
    # 3. Testing Rate: Tests per 1,000 population
    df_agg['testing_rate_per_1k'] = (df_agg['total_tests'] / df_agg['population']) * 1000
    
    # 4. Vaccination Rate: Percentage of population fully vaccinated
    df_agg['vaccination_rate_pct'] = (df_agg['people_fully_vaccinated'] / df_agg['population']) * 100
    
    # 5. Hospitalization Rate: Hospitalizations per 100,000 (if available)
    df_agg['hospitalization_rate_per_100k'] = (df_agg['hosp_patients'] / df_agg['population']) * 100000
    
    # 6. ICU Rate: ICU patients per 100,000 (if available)
    df_agg['icu_rate_per_100k'] = (df_agg['icu_patients'] / df_agg['population']) * 100000
    
    # 7. Case Fatality Rate: Deaths per confirmed case
    df_agg['case_fatality_rate_pct'] = (df_agg['total_deaths'] / df_agg['total_cases']) * 100
    
    # 8. Recent Case Growth: Average new cases in the last month
    df_agg['recent_case_growth'] = df_agg['new_cases']
    
    # 9. Recent Death Growth: Average new deaths in the last month
    df_agg['recent_death_growth'] = df_agg['new_deaths']
    
    # 10. Case-to-Test Ratio: Proportion of tests that are positive
    df_agg['case_to_test_ratio'] = df_agg['total_cases'] / df_agg['total_tests']
    
    # Extract external data (GDP, population density, etc.)
    external_data = extract_external_data(df)
    
    # Now let's calculate time-based features
    # For this, we need to go back to the original data
    
    time_features = []
    
    for country in countries_list:
        country_data = df_countries[df_countries['location'] == country].copy()
        
        if len(country_data) < 30:  # Skip countries with insufficient data
            continue
            
        # Ensure data is sorted by date
        country_data = country_data.sort_values('date')
        
        # 11. Days to peak: Number of days from first case to peak cases
        if 'new_cases' in country_data.columns and not country_data['new_cases'].isna().all():
            first_case_date = country_data[country_data['new_cases'] > 0]['date'].min()
            peak_case_date = country_data[country_data['new_cases'] == country_data['new_cases'].max()]['date'].min()
            
            if not pd.isna(first_case_date) and not pd.isna(peak_case_date):
                days_to_peak = (peak_case_date - first_case_date).days
            else:
                days_to_peak = np.nan
        else:
            days_to_peak = np.nan
            
        # 12. Peak case rate: Maximum daily cases per 100,000
        if 'new_cases' in country_data.columns and 'population' in country_data.columns:
            peak_case_rate = (country_data['new_cases'].max() / country_data['population'].iloc[0]) * 100000
        else:
            peak_case_rate = np.nan
            
        # 13. Recovery speed: Rate of decline after peak (if applicable)
        # Simplified as the average daily decrease in cases after peak
        if not pd.isna(peak_case_date) and 'new_cases' in country_data.columns:
            post_peak_data = country_data[country_data['date'] > peak_case_date]
            
            if len(post_peak_data) > 30:  # Ensure we have enough data after peak
                # Calculate the slope of the trend line for new cases after peak
                post_peak_data['days_after_peak'] = (post_peak_data['date'] - peak_case_date).dt.days
                
                if not post_peak_data['new_cases'].isna().all():
                    # Use simple linear regression to get the slope
                    x = post_peak_data['days_after_peak'].values
                    y = post_peak_data['new_cases'].values
                    
                    # Remove NaN values
                    mask = ~np.isnan(y)
                    x = x[mask]
                    y = y[mask]
                    
                    if len(x) > 0 and len(y) > 0:
                        recovery_speed = np.polyfit(x, y, 1)[0]  # Slope of the line
                    else:
                        recovery_speed = np.nan
                else:
                    recovery_speed = np.nan
            else:
                recovery_speed = np.nan
        else:
            recovery_speed = np.nan
            
        # 14. Speed of vaccine rollout: Days to reach 50% vaccination
        if 'people_fully_vaccinated' in country_data.columns and 'population' in country_data.columns:
            # Calculate vaccination percentage for each day
            country_data['vaccination_pct'] = (country_data['people_fully_vaccinated'] / country_data['population'].iloc[0]) * 100
            
            # Find the first date when vaccination reached 50%
            vax_50pct_data = country_data[country_data['vaccination_pct'] >= 50]
            
            if len(vax_50pct_data) > 0:
                first_vax_date = country_data[country_data['people_fully_vaccinated'] > 0]['date'].min()
                vax_50pct_date = vax_50pct_data['date'].min()
                
                if not pd.isna(first_vax_date) and not pd.isna(vax_50pct_date):
                    days_to_50pct_vax = (vax_50pct_date - first_vax_date).days
                else:
                    days_to_50pct_vax = np.nan
            else:
                # If country never reached 50% vaccination, use a high value or NaN
                days_to_50pct_vax = np.nan
        else:
            days_to_50pct_vax = np.nan
            
        # 15. Vaccination acceleration: Rate of increase in vaccination percentage
        if 'people_fully_vaccinated' in country_data.columns and 'population' in country_data.columns:
            # Get data where vaccination has started
            vax_data = country_data[country_data['people_fully_vaccinated'] > 0].copy()
            
            if len(vax_data) > 30:  # Ensure we have enough vaccination data
                # Calculate vaccination percentage
                vax_data['vaccination_pct'] = (vax_data['people_fully_vaccinated'] / vax_data['population'].iloc[0]) * 100
                
                # Calculate days since first vaccination
                first_vax_date = vax_data['date'].min()
                vax_data['days_since_first_vax'] = (vax_data['date'] - first_vax_date).dt.days
                
                # Use simple linear regression to get the slope (percentage points per day)
                x = vax_data['days_since_first_vax'].values
                y = vax_data['vaccination_pct'].values
                
                # Remove NaN values
                mask = ~np.isnan(y)
                x = x[mask]
                y = y[mask]
                
                if len(x) > 0 and len(y) > 0:
                    vax_acceleration = np.polyfit(x, y, 1)[0]  # Slope of the line (percentage points per day)
                else:
                    vax_acceleration = np.nan
            else:
                vax_acceleration = np.nan
        else:
            vax_acceleration = np.nan
            
        # Add the time-based features to our list
        time_features.append({
            'location': country,
            'days_to_peak': days_to_peak,
            'peak_case_rate_per_100k': peak_case_rate,
            'recovery_speed': recovery_speed,
            'days_to_50pct_vax': days_to_50pct_vax,
            'vax_acceleration': vax_acceleration
        })
    
    # Convert the time features list to a DataFrame
    df_time = pd.DataFrame(time_features)
    
    # Merge the time-based features with our aggregated features
    df_features = pd.merge(df_agg, df_time, on='location', how='left')
    
    # Merge with external data (GDP, population density, etc.)
    external_data = extract_external_data(df)
    df_features = pd.merge(df_features, external_data, on='location', how='left')
    
    # Drop the intermediate columns we don't need for clustering
    cols_to_drop = ['total_cases', 'total_deaths', 'total_tests', 'people_fully_vaccinated', 
                    'hosp_patients', 'icu_patients', 'new_cases', 'new_deaths']
    df_features = df_features.drop(columns=[col for col in cols_to_drop if col in df_features.columns])
    
    # Define all feature columns
    feature_cols = [
        # Epidemiological features
        'case_rate_per_100k', 'mortality_rate_per_100k', 'testing_rate_per_1k',
        'vaccination_rate_pct', 'hospitalization_rate_per_100k', 'icu_rate_per_100k',
        'case_fatality_rate_pct', 'recent_case_growth', 'recent_death_growth',
        'case_to_test_ratio',
        
        # Time-based features
        'days_to_peak', 'peak_case_rate_per_100k', 'recovery_speed',
        'days_to_50pct_vax', 'vax_acceleration',
        
        # External data features
        'population_density', 'median_age', 'aged_65_older', 'aged_70_older',
        'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence',
        'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
        'life_expectancy', 'human_development_index'
    ]
    
    # Filter to only include columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df_features.columns]
    
    # Handle missing values using KNNImputer for more sophisticated imputation
    print("Imputing missing values using KNN...")
    
    # First, handle any infinities
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    
    # Create a copy of the features for imputation
    features_for_imputation = df_features[feature_cols].copy()
    
    # Check if we have enough data for KNN imputation
    if len(df_features) > 5 and not features_for_imputation.isna().all().any():
        try:
            # Use KNNImputer to fill missing values
            imputer = KNNImputer(n_neighbors=min(5, len(df_features)-1))
            imputed_features = imputer.fit_transform(features_for_imputation)
            
            # Replace the original features with the imputed values
            for i, col in enumerate(feature_cols):
                df_features[col] = imputed_features[:, i]
        except Exception as e:
            print(f"KNN imputation failed: {e}")
            print("Falling back to median imputation...")
            # Fall back to median imputation
            for col in feature_cols:
                if col in df_features.columns and df_features[col].isna().any():
                    median_val = df_features[col].median()
                    if pd.isna(median_val):  # If median is also NaN, use 0
                        df_features[col] = df_features[col].fillna(0)
                    else:
                        df_features[col] = df_features[col].fillna(median_val)
    else:
        print("Not enough data for KNN imputation, using median imputation instead...")
        # Fall back to median imputation
        for col in feature_cols:
            if col in df_features.columns and df_features[col].isna().any():
                median_val = df_features[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    df_features[col] = df_features[col].fillna(0)
                else:
                    df_features[col] = df_features[col].fillna(median_val)
    
    print(f"Created features for {len(df_features)} countries.")
    return df_features

def standardize_features(df):
    """Standardize the features for clustering."""
    print("Standardizing features...")
    
    # Select only the numeric feature columns
    feature_cols = [
        # Epidemiological features
        'case_rate_per_100k', 'mortality_rate_per_100k', 'testing_rate_per_1k',
        'vaccination_rate_pct', 'hospitalization_rate_per_100k', 'icu_rate_per_100k',
        'case_fatality_rate_pct', 'recent_case_growth', 'recent_death_growth',
        'case_to_test_ratio',
        
        # Time-based features
        'days_to_peak', 'peak_case_rate_per_100k', 'recovery_speed',
        'days_to_50pct_vax', 'vax_acceleration',
        
        # External data features
        'population_density', 'median_age', 'aged_65_older', 'aged_70_older',
        'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence',
        'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand',
        'life_expectancy', 'human_development_index'
    ]
    
    # Filter to only include columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Create a copy of the dataframe with only the features we want to standardize
    df_features = df[feature_cols].copy()
    
    # Handle any remaining missing values or infinities
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.fillna(df_features.median())
    
    # Standardize the features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_features),
        columns=feature_cols,
        index=df.index
    )
    
    # Add back the country name
    df_scaled['location'] = df['location']
    
    # Also keep the population for reference
    if 'population' in df.columns:
        df_scaled['population'] = df['population']
    
    print(f"Standardized {len(feature_cols)} features.")
    return df_scaled, scaler

def visualize_features(df_original, df_scaled):
    """Visualize the distribution of features before and after standardization."""
    print("Generating feature distribution visualizations...")
    
    # Select only the numeric feature columns
    feature_cols = [col for col in df_scaled.columns 
                   if col not in ['location', 'population'] 
                   and col in df_original.columns]
    
    # Create a directory for the visualizations if it doesn't exist
    viz_dir = 'feature_visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Create a correlation heatmap for the original features
    plt.figure(figsize=(14, 12))
    corr_matrix = df_original[feature_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/correlation_matrix.png')
    plt.close()
    
    # Create histograms for each feature before and after standardization
    for feature in feature_cols:
        plt.figure(figsize=(12, 6))
        
        # Original feature distribution
        plt.subplot(1, 2, 1)
        sns.histplot(df_original[feature].dropna(), kde=True)
        plt.title(f'Original {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        
        # Standardized feature distribution
        plt.subplot(1, 2, 2)
        sns.histplot(df_scaled[feature].dropna(), kde=True)
        plt.title(f'Standardized {feature}')
        plt.xlabel(f'{feature} (standardized)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/{feature}_distribution.png')
        plt.close()
    
    print(f"Saved visualizations to {viz_dir}/ directory.")

def save_features(df_original, df_scaled):
    """Save the feature datasets to CSV files."""
    print("Saving feature datasets...")
    
    # Save the original features
    df_original.to_csv('covid_features_original.csv', index=False)
    print(f"Saved original features to covid_features_original.csv")
    
    # Save the standardized features
    df_scaled.to_csv('covid_features_standardized.csv', index=False)
    print(f"Saved standardized features to covid_features_standardized.csv")

def main():
    """Main function to run the feature engineering process."""
    # Load the cleaned data
    df = load_data('cleaned_covid_data.csv')
    
    # Create features for clustering
    df_features = create_features(df)
    
    # Standardize the features
    df_scaled, scaler = standardize_features(df_features)
    
    # Visualize the features
    visualize_features(df_features, df_scaled)
    
    # Save the features
    save_features(df_features, df_scaled)
    
    # Print a summary of the features
    print("\nFeature Summary:")
    print(f"Total countries: {len(df_features)}")
    print(f"Total features: {df_scaled.shape[1] - 2}")
    print("\nKey features created:")
    print("- Epidemiological: case rates, mortality rates, testing rates, vaccination rates")
    print("- Time-based: days to peak, recovery speed, vaccine rollout speed")
    print("- External: GDP per capita, population density, healthcare indicators")
    
    print("\nFeature engineering completed successfully!")

if __name__ == "__main__":
    main()
