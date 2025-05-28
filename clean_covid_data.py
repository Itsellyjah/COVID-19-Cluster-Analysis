#!/usr/bin/env python3
"""
COVID-19 Data Cleaning Script
This script cleans the Our World in Data COVID-19 dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def load_data(file_path):
    """Load the COVID-19 dataset."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the COVID-19 dataset."""
    print(f"Original dataset shape: {df.shape}")
    
    # Make a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Step 1: Check for and remove duplicates
    duplicates = cleaned_df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing {duplicates} duplicate rows...")
        cleaned_df = cleaned_df.drop_duplicates()
    else:
        print("No duplicates found.")
    
    # Step 2: Check for missing values
    missing_values = cleaned_df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0].sort_values(ascending=False).head(10))
    
    # Step 3: Handle missing values for key columns
    key_columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths']
    for col in key_columns:
        if col in cleaned_df.columns:
            # Replace missing values with 0 for case and death counts
            missing_count = cleaned_df[col].isnull().sum()
            if missing_count > 0:
                print(f"Replacing {missing_count} missing values in {col} with 0...")
                cleaned_df[col] = cleaned_df[col].fillna(0)
    
    # Step 4: Convert date column to datetime
    print("Converting date column to datetime format...")
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
    
    # Step 5: Select only the most relevant columns
    relevant_columns = [
        'iso_code', 'continent', 'location', 'date', 
        'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
        'total_cases_per_million', 'new_cases_per_million', 
        'total_deaths_per_million', 'new_deaths_per_million',
        'icu_patients', 'hosp_patients', 'total_tests', 'new_tests',
        'positive_rate', 'total_vaccinations', 'people_vaccinated',
        'people_fully_vaccinated', 'total_boosters', 'population'
    ]
    
    # Only keep columns that exist in the dataframe
    relevant_columns = [col for col in relevant_columns if col in cleaned_df.columns]
    
    print(f"Selecting {len(relevant_columns)} relevant columns...")
    cleaned_df = cleaned_df[relevant_columns]
    
    # Step 6: Handle outliers in numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Skip certain columns where negative values might be meaningful
        if col not in ['new_cases', 'new_deaths', 'new_tests']:
            # Replace negative values with 0
            neg_count = (cleaned_df[col] < 0).sum()
            if neg_count > 0:
                print(f"Replacing {neg_count} negative values in {col} with 0...")
                cleaned_df.loc[cleaned_df[col] < 0, col] = 0
    
    # Step 7: Sort the data by location and date
    print("Sorting data by location and date...")
    cleaned_df = cleaned_df.sort_values(['location', 'date'])
    
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    return cleaned_df

def save_data(df, output_path):
    """Save the cleaned dataset to a CSV file."""
    print(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully! File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

def generate_summary(df):
    """Generate a summary of the cleaned dataset."""
    print("\nDataset Summary:")
    print(f"Time period: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of countries/regions: {df['location'].nunique()}")
    print(f"Number of continents: {df['continent'].nunique()}")
    
    # Get the top 5 countries by total cases
    if 'total_cases' in df.columns:
        top_cases = df.groupby('location')['total_cases'].max().sort_values(ascending=False).head(5)
        print("\nTop 5 countries by total cases:")
        print(top_cases)
    
    # Get the top 5 countries by total deaths
    if 'total_deaths' in df.columns:
        top_deaths = df.groupby('location')['total_deaths'].max().sort_values(ascending=False).head(5)
        print("\nTop 5 countries by total deaths:")
        print(top_deaths)

def main():
    """Main function to run the data cleaning process."""
    start_time = time.time()
    
    input_file = "owid-covid-data.csv"
    output_file = "cleaned_covid_data.csv"
    
    # Load the data
    df = load_data(input_file)
    
    # Clean the data
    cleaned_df = clean_data(df)
    
    # Generate summary
    generate_summary(cleaned_df)
    
    # Save the cleaned data
    save_data(cleaned_df, output_file)
    
    end_time = time.time()
    print(f"\nData cleaning completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
