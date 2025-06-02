#!/usr/bin/env python3
"""
Script to remove Tableau-related files and directories from the COVID-19 Cluster Analysis project.
This helps clean up the project for GitHub by removing unnecessary files.
"""

import os
import shutil
import sys

def print_colored(text, color_code):
    """Print colored text to the console."""
    print(f"\033[{color_code}m{text}\033[0m")

def print_success(text):
    """Print success message in green."""
    print_colored(text, "92")

def print_info(text):
    """Print info message in blue."""
    print_colored(text, "94")

def print_warning(text):
    """Print warning message in yellow."""
    print_colored(text, "93")

def remove_file(file_path):
    """Remove a file if it exists."""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print_success(f"Removed file: {file_path}")
            return True
        except Exception as e:
            print_warning(f"Failed to remove {file_path}: {e}")
            return False
    else:
        print_info(f"File not found: {file_path}")
        return False

def remove_directory(dir_path):
    """Remove a directory and all its contents if it exists."""
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print_success(f"Removed directory: {dir_path}")
            return True
        except Exception as e:
            print_warning(f"Failed to remove directory {dir_path}: {e}")
            return False
    else:
        print_info(f"Directory not found: {dir_path}")
        return False

def main():
    """Main function to remove Tableau-related files and directories."""
    # Get the project directory (current directory)
    project_dir = os.getcwd()
    
    print_info("Starting cleanup of Tableau-related files...")
    
    # List of Tableau-related files to remove
    tableau_files = [
        "tableau_cluster_data.csv",
        "tableau_feature_data.csv",
        "tableau_time_series_data.csv",
        "prepare_tableau_data.py"
    ]
    
    # List of Tableau-related directories to remove
    tableau_dirs = [
        "tableau_data"
    ]
    
    # Remove files
    files_removed = 0
    for file_name in tableau_files:
        file_path = os.path.join(project_dir, file_name)
        if remove_file(file_path):
            files_removed += 1
    
    # Remove directories
    dirs_removed = 0
    for dir_name in tableau_dirs:
        dir_path = os.path.join(project_dir, dir_name)
        if remove_directory(dir_path):
            dirs_removed += 1
    
    # Print summary
    print_info("\nCleanup Summary:")
    print_info(f"Files removed: {files_removed}")
    print_info(f"Directories removed: {dirs_removed}")
    
    if files_removed == 0 and dirs_removed == 0:
        print_warning("No Tableau-related files or directories were found or removed.")
    else:
        print_success("Successfully cleaned up Tableau-related files and directories!")
    
    print_info("\nNext steps:")
    print_info("1. Run 'git status' to see the changes")
    print_info("2. Run 'git add .' to stage all changes")
    print_info("3. Run 'git commit -m \"Remove Tableau files\"' to commit the changes")
    print_info("4. Run 'git push' to push the changes to GitHub")

if __name__ == "__main__":
    main()
