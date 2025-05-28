#!/usr/bin/env python3
"""
Cleanup Script for GitHub Repository
This script organizes the COVID-19 project files and removes unnecessary ones.
"""

import os
import shutil
import glob

# Create directories if they don't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Files to keep (essential files)
essential_files = [
    # Core analysis scripts
    "clean_covid_data.py",
    "feature_engineering.py",
    "clustering_analysis.py",
    "cluster_profiling.py",
    "simple_visualizations.py",
    "prepare_tableau_data.py",
    
    # Data files
    "covid_clustering_results.csv",
    "covid_features_original.csv",
    "covid_features_standardized.csv",
    
    # Tableau files
    "tableau_cluster_data.csv",
    "tableau_feature_data.csv",
    "tableau_time_series_data.csv",
    
    # README and other documentation
    "README.md"
]

# Directories to keep
essential_dirs = [
    "visualization_outputs",
    "cluster_profiles",
    "feature_visualizations"
]

# Files to be archived (not deleted but moved to an archive folder)
archive_files = [
    "covid_dashboard.py",
    "simple_covid_dashboard.py",
    "generate_visualizations.py",
    "generate_visualizations_fixed.py"
]

# Large data files to be excluded from GitHub but kept locally
large_data_files = [
    "owid-covid-data.csv",
    "cleaned_covid_data.csv"
]

def main():
    print("Starting project cleanup for GitHub...")
    
    # Create archive directory
    archive_dir = "archive"
    ensure_dir(archive_dir)
    
    # Create data directory for large files
    data_dir = "data"
    ensure_dir(data_dir)
    
    # Create .gitignore file
    with open(".gitignore", "w") as f:
        f.write("# Large data files\n")
        f.write("data/\n")
        f.write("archive/\n")
        f.write("\n# Python cache files\n")
        f.write("__pycache__/\n")
        f.write("*.py[cod]\n")
        f.write("*$py.class\n")
        f.write("\n# Jupyter Notebook\n")
        f.write(".ipynb_checkpoints\n")
        f.write("\n# Mac OS files\n")
        f.write(".DS_Store\n")
    
    print("Created .gitignore file")
    
    # Create README.md if it doesn't exist
    if not os.path.exists("README.md"):
        with open("README.md", "w") as f:
            f.write("# COVID-19 Cluster Analysis\n\n")
            f.write("This repository contains a comprehensive analysis of COVID-19 data using clustering techniques to identify patterns in how countries responded to the pandemic.\n\n")
            f.write("## Project Structure\n\n")
            f.write("- **clean_covid_data.py**: Script to clean the original COVID-19 dataset\n")
            f.write("- **feature_engineering.py**: Script to create features for clustering analysis\n")
            f.write("- **clustering_analysis.py**: Script to perform clustering using various algorithms\n")
            f.write("- **cluster_profiling.py**: Script to analyze and interpret clusters\n")
            f.write("- **simple_visualizations.py**: Script to generate visualizations of the clustering results\n")
            f.write("- **prepare_tableau_data.py**: Script to prepare data for Tableau visualization\n\n")
            f.write("## Visualizations\n\n")
            f.write("The `visualization_outputs` directory contains all visualizations generated from the analysis, including:\n\n")
            f.write("- Cluster distribution charts\n")
            f.write("- Scatter plots comparing key metrics\n")
            f.write("- Time series visualizations\n")
            f.write("- Cluster profile radar charts\n")
            f.write("- Feature heatmaps\n\n")
            f.write("## Data\n\n")
            f.write("The analysis is based on the \"Our World in Data\" COVID-19 dataset, which can be downloaded from: https://github.com/owid/covid-19-data/tree/master/public/data\n\n")
            f.write("Due to size constraints, the original data files are not included in this repository. To run the scripts, download the dataset and place it in the project directory.\n\n")
            f.write("## Cluster Analysis Results\n\n")
            f.write("The analysis identified four main categories of regions based on their COVID-19 response patterns:\n\n")
            f.write("1. **High-Risk Regions** (73 countries): High case and mortality rates\n")
            f.write("2. **Healthcare-Strained Regions** (141 countries): High hospitalization/ICU utilization\n")
            f.write("3. **High-Mortality, High-Resource Region** (United States): Unique pattern despite resources\n")
            f.write("4. **High-Case, Low-Mortality Regions** (19 countries): Successfully managed mortality despite high case rates\n")
        
        print("Created README.md file")
    
    # Move files to appropriate directories
    all_files = os.listdir(".")
    
    for file in all_files:
        # Skip directories
        if os.path.isdir(file) and file not in [archive_dir, data_dir]:
            continue
            
        # Move large data files to data directory
        if file in large_data_files:
            try:
                shutil.move(file, os.path.join(data_dir, file))
                print(f"Moved large data file to data directory: {file}")
            except Exception as e:
                print(f"Error moving {file}: {e}")
                
        # Move archive files to archive directory
        elif file in archive_files:
            try:
                shutil.move(file, os.path.join(archive_dir, file))
                print(f"Moved to archive: {file}")
            except Exception as e:
                print(f"Error archiving {file}: {e}")
    
    print("\nCleanup complete!")
    print("\nFiles ready for GitHub:")
    for file in essential_files:
        if os.path.exists(file):
            print(f"- {file}")
    
    print("\nDirectories ready for GitHub:")
    for directory in essential_dirs:
        if os.path.exists(directory):
            print(f"- {directory}")
    
    print("\nLarge data files moved to 'data' directory (excluded from GitHub):")
    for file in os.listdir(data_dir):
        print(f"- {file}")
    
    print("\nArchived files (excluded from GitHub):")
    for file in os.listdir(archive_dir):
        print(f"- {file}")
    
    print("\nTo initialize Git repository and push to GitHub:")
    print("1. git init")
    print("2. git add .")
    print("3. git commit -m \"Initial commit\"")
    print("4. git branch -M main")
    print("5. git remote add origin <your-github-repo-url>")
    print("6. git push -u origin main")

if __name__ == "__main__":
    main()
