#!/usr/bin/env python3
"""
COVID-19 Clustering Analysis Script
This script performs clustering analysis on the engineered COVID-19 features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the standardized COVID-19 features."""
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

def perform_kmeans(df, feature_cols, n_clusters_range=range(2, 11)):
    """Perform K-means clustering and determine the optimal number of clusters."""
    print("\nPerforming K-means clustering...")
    
    # Create a directory for the clustering visualizations if it doesn't exist
    viz_dir = 'clustering_visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Extract the features for clustering
    X = df[feature_cols].values
    
    # Calculate silhouette scores and inertia for different numbers of clusters
    silhouette_scores = []
    inertia_values = []
    
    for n_clusters in n_clusters_range:
        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        if len(np.unique(cluster_labels)) > 1:  # Silhouette score requires at least 2 clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)
        
        print(f"  K-means with {n_clusters} clusters - Silhouette Score: {silhouette_scores[-1]:.3f}, Inertia: {inertia_values[-1]:.3f}")
    
    # Plot the elbow curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(list(n_clusters_range), inertia_values, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(list(n_clusters_range), silhouette_scores, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/kmeans_optimal_clusters.png')
    plt.close()
    
    # Find the optimal number of clusters based on silhouette score
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_n_clusters}")
    
    # Perform K-means with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
    df['kmeans_cluster'] = kmeans.fit_predict(X)
    
    # Calculate Davies-Bouldin index
    db_score = davies_bouldin_score(X, df['kmeans_cluster'])
    print(f"Davies-Bouldin Index for K-means: {db_score:.3f} (lower is better)")
    
    return df, kmeans, optimal_n_clusters

def perform_dbscan(df, feature_cols, eps_range=np.arange(0.5, 2.1, 0.1), min_samples_range=range(3, 11)):
    """Perform DBSCAN clustering and determine the optimal parameters."""
    print("\nPerforming DBSCAN clustering...")
    
    # Create a directory for the clustering visualizations if it doesn't exist
    viz_dir = 'clustering_visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Extract the features for clustering
    X = df[feature_cols].values
    
    # Find optimal parameters for DBSCAN
    best_silhouette = -1
    best_eps = None
    best_min_samples = None
    best_labels = None
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            # Fit DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Count the number of clusters (excluding noise points labeled as -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Calculate silhouette score if there are at least 2 clusters and not all points are noise
            if n_clusters >= 2 and n_noise < len(labels):
                # Only include non-noise points in silhouette calculation
                non_noise_mask = labels != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette_avg = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                    
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette': silhouette_avg
                    })
                    
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_eps = eps
                        best_min_samples = min_samples
                        best_labels = labels
    
    # If we found at least one valid configuration
    if best_eps is not None:
        print(f"Optimal DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")
        print(f"Number of clusters: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")
        print(f"Number of noise points: {list(best_labels).count(-1)}")
        print(f"Silhouette Score: {best_silhouette:.3f}")
        
        # Create a DataFrame with the results for plotting
        results_df = pd.DataFrame(results)
        
        # Plot the results
        if len(results_df) > 0:
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            pivot_silhouette = results_df.pivot_table(
                index='min_samples', columns='eps', values='silhouette', aggfunc='mean'
            )
            sns.heatmap(pivot_silhouette, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Silhouette Score for DBSCAN Parameters')
            
            plt.subplot(1, 2, 2)
            pivot_clusters = results_df.pivot_table(
                index='min_samples', columns='eps', values='n_clusters', aggfunc='mean'
            )
            sns.heatmap(pivot_clusters, annot=True, cmap='viridis', fmt='g')
            plt.title('Number of Clusters for DBSCAN Parameters')
            
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/dbscan_parameter_tuning.png')
            plt.close()
        
        # Perform DBSCAN with the optimal parameters
        dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        df['dbscan_cluster'] = dbscan.fit_predict(X)
        
        # Replace -1 with a more descriptive label for noise points
        df['dbscan_cluster'] = df['dbscan_cluster'].replace(-1, -1)
        
        return df, dbscan
    else:
        print("Could not find optimal DBSCAN parameters. Try a different range of eps and min_samples.")
        df['dbscan_cluster'] = -1  # All points are treated as noise
        return df, None

def perform_hierarchical_clustering(df, feature_cols, n_clusters=None):
    """Perform hierarchical clustering and visualize the dendrogram."""
    print("\nPerforming hierarchical clustering...")
    
    # Create a directory for the clustering visualizations if it doesn't exist
    viz_dir = 'clustering_visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Extract the features for clustering
    X = df[feature_cols].values
    
    # Compute the linkage matrix
    Z = linkage(X, method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(12, 8))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Countries')
    plt.ylabel('Distance')
    
    # If we have too many samples, we'll truncate the dendrogram
    if len(df) > 50:
        dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.)
    else:
        dendrogram(Z, labels=df['location'].values, leaf_rotation=90., leaf_font_size=10.)
    
    plt.savefig(f'{viz_dir}/hierarchical_dendrogram.png')
    plt.close()
    
    # If n_clusters is not specified, use the same as K-means
    if n_clusters is None:
        # Find the optimal number of clusters using the elbow method on the dendrogram
        last = Z[-10:, 2]
        acceleration = np.diff(last, 2)
        n_clusters = len(last) - np.argmax(acceleration) + 1
        print(f"Optimal number of clusters for hierarchical clustering: {n_clusters}")
    
    # Perform hierarchical clustering with the optimal number of clusters
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df['hierarchical_cluster'] = hierarchical.fit_predict(X)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, df['hierarchical_cluster'])
    print(f"Silhouette Score for hierarchical clustering: {silhouette_avg:.3f}")
    
    return df, hierarchical

def perform_gmm(df, feature_cols, n_components_range=range(2, 11)):
    """Perform Gaussian Mixture Model clustering."""
    print("\nPerforming Gaussian Mixture Model clustering...")
    
    # Create a directory for the clustering visualizations if it doesn't exist
    viz_dir = 'clustering_visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Extract the features for clustering
    X = df[feature_cols].values
    
    # Calculate BIC and AIC for different numbers of components
    bic_values = []
    aic_values = []
    silhouette_scores = []
    
    for n_components in n_components_range:
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        
        # Predict clusters
        cluster_labels = gmm.predict(X)
        
        # Calculate BIC and AIC
        bic_values.append(gmm.bic(X))
        aic_values.append(gmm.aic(X))
        
        # Calculate silhouette score
        if len(np.unique(cluster_labels)) > 1:  # Silhouette score requires at least 2 clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
        
        print(f"  GMM with {n_components} components - BIC: {bic_values[-1]:.3f}, AIC: {aic_values[-1]:.3f}, Silhouette: {silhouette_scores[-1]:.3f}")
    
    # Plot the BIC and AIC curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(list(n_components_range), bic_values, 'o-', label='BIC')
    plt.plot(list(n_components_range), aic_values, 's-', label='AIC')
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion')
    plt.title('BIC and AIC for Optimal Components')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(list(n_components_range), silhouette_scores, 'o-')
    plt.xlabel('Number of Components')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal Components')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/gmm_optimal_components.png')
    plt.close()
    
    # Find the optimal number of components based on BIC
    optimal_n_components = n_components_range[np.argmin(bic_values)]
    print(f"Optimal number of components based on BIC: {optimal_n_components}")
    
    # Perform GMM with the optimal number of components
    gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
    df['gmm_cluster'] = gmm.fit_predict(X)
    
    return df, gmm, optimal_n_components

def perform_dimensionality_reduction(df, feature_cols, cluster_col, method='pca'):
    """Perform dimensionality reduction for visualization."""
    print(f"\nPerforming {method.upper()} for dimensionality reduction...")
    
    # Create a directory for the clustering visualizations if it doesn't exist
    viz_dir = 'clustering_visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Extract the features
    X = df[feature_cols].values
    
    # Perform dimensionality reduction
    if method.lower() == 'pca':
        # PCA for 2D visualization
        reducer = PCA(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        
        # Calculate explained variance
        explained_variance = reducer.explained_variance_ratio_.sum()
        print(f"Explained variance by 2 principal components: {explained_variance:.2%}")
        
        # Get the feature importance
        feature_importance = pd.DataFrame(
            reducer.components_.T,
            columns=[f'PC{i+1}' for i in range(2)],
            index=feature_cols
        )
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.heatmap(feature_importance, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Importance in Principal Components')
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/pca_feature_importance.png')
        plt.close()
        
    elif method.lower() == 'tsne':
        # t-SNE for 2D visualization
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_reduced = reducer.fit_transform(X)
        
        print("t-SNE does not provide explained variance or feature importance.")
    
    # Create a DataFrame with the reduced dimensions
    df_reduced = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'cluster': df[cluster_col],
        'location': df['location']
    })
    
    # Plot the reduced dimensions
    plt.figure(figsize=(12, 10))
    
    # Get the number of clusters
    clusters = df_reduced['cluster'].unique()
    
    # Define a colormap
    cmap = plt.cm.get_cmap('tab10', len(clusters))
    
    # Plot each cluster
    for i, cluster in enumerate(sorted(clusters)):
        if cluster == -1:  # Noise points in DBSCAN
            plt.scatter(
                df_reduced[df_reduced['cluster'] == cluster]['x'],
                df_reduced[df_reduced['cluster'] == cluster]['y'],
                s=50, c='black', marker='x', label=f'Noise'
            )
        else:
            plt.scatter(
                df_reduced[df_reduced['cluster'] == cluster]['x'],
                df_reduced[df_reduced['cluster'] == cluster]['y'],
                s=50, c=[cmap(i)], label=f'Cluster {cluster}'
            )
    
    # Add country labels for the points
    for i, row in df_reduced.iterrows():
        plt.annotate(
            row['location'],
            (row['x'], row['y']),
            fontsize=8,
            alpha=0.7,
            ha='center',
            va='center'
        )
    
    plt.title(f'2D Visualization of Clusters using {method.upper()}')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/{method.lower()}_{cluster_col}.png')
    plt.close()
    
    return df_reduced

def analyze_clusters(df, cluster_col, feature_cols):
    """Analyze the characteristics of each cluster."""
    print(f"\nAnalyzing {cluster_col} clusters...")
    
    # Create a directory for the clustering visualizations if it doesn't exist
    viz_dir = 'clustering_visualizations'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Get the number of clusters
    clusters = sorted(df[cluster_col].unique())
    
    # Calculate the mean of each feature for each cluster
    cluster_means = df.groupby(cluster_col)[feature_cols].mean()
    
    # Plot the cluster profiles
    plt.figure(figsize=(15, 10))
    
    # Normalize the means for better visualization
    cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # Plot heatmap of cluster profiles
    sns.heatmap(cluster_means_normalized, annot=False, cmap='coolwarm', center=0.5)
    plt.title(f'Cluster Profiles for {cluster_col}')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/{cluster_col}_profiles_heatmap.png')
    plt.close()
    
    # Create a more detailed profile for each cluster
    cluster_profiles = pd.DataFrame(index=clusters)
    
    for feature in feature_cols:
        for cluster in clusters:
            cluster_profiles.loc[cluster, feature] = df[df[cluster_col] == cluster][feature].mean()
    
    # Calculate the size of each cluster
    cluster_sizes = df[cluster_col].value_counts().sort_index()
    cluster_profiles['size'] = cluster_sizes
    cluster_profiles['percentage'] = (cluster_sizes / len(df)) * 100
    
    # List the countries in each cluster
    cluster_countries = {}
    for cluster in clusters:
        countries = df[df[cluster_col] == cluster]['location'].tolist()
        cluster_countries[cluster] = countries
    
    # Print the cluster profiles
    print("\nCluster Profiles:")
    print(cluster_profiles)
    
    # Print the top 5 countries in each cluster
    print("\nTop Countries in Each Cluster:")
    for cluster in clusters:
        if cluster == -1:
            print(f"Noise Points: {', '.join(cluster_countries[cluster][:5])}")
        else:
            print(f"Cluster {cluster}: {', '.join(cluster_countries[cluster][:5])}")
    
    # Interpret the clusters based on their profiles
    print("\nCluster Interpretation:")
    for cluster in clusters:
        if cluster == -1:
            print(f"Noise Points: Countries with unique COVID-19 patterns that don't fit well into any cluster.")
        else:
            # Get the top 3 highest and lowest features for this cluster
            cluster_profile = cluster_profiles.loc[cluster, feature_cols]
            top_features = cluster_profile.nlargest(3).index.tolist()
            bottom_features = cluster_profile.nsmallest(3).index.tolist()
            
            print(f"Cluster {cluster} ({cluster_profiles.loc[cluster, 'size']} countries, {cluster_profiles.loc[cluster, 'percentage']:.1f}%):")
            print(f"  High: {', '.join(top_features)}")
            print(f"  Low: {', '.join(bottom_features)}")
    
    return cluster_profiles, cluster_countries

def main():
    """Main function to run the clustering analysis."""
    # Load the standardized data
    df = load_data('covid_features_standardized.csv')
    
    # Select the features for clustering
    feature_cols = [col for col in df.columns if col not in ['location', 'population']]
    
    print(f"Using {len(feature_cols)} features for clustering: {', '.join(feature_cols)}")
    
    # Perform K-means clustering
    df, kmeans, optimal_k = perform_kmeans(df, feature_cols)
    
    # Analyze K-means clusters
    kmeans_profiles, kmeans_countries = analyze_clusters(df, 'kmeans_cluster', feature_cols)
    
    # Visualize K-means clusters using PCA
    perform_dimensionality_reduction(df, feature_cols, 'kmeans_cluster', method='pca')
    
    # Visualize K-means clusters using t-SNE
    perform_dimensionality_reduction(df, feature_cols, 'kmeans_cluster', method='tsne')
    
    # Perform hierarchical clustering
    df, hierarchical = perform_hierarchical_clustering(df, feature_cols, n_clusters=optimal_k)
    
    # Analyze hierarchical clusters
    hierarchical_profiles, hierarchical_countries = analyze_clusters(df, 'hierarchical_cluster', feature_cols)
    
    # Visualize hierarchical clusters using PCA
    perform_dimensionality_reduction(df, feature_cols, 'hierarchical_cluster', method='pca')
    
    # Perform DBSCAN clustering
    df, dbscan = perform_dbscan(df, feature_cols)
    
    if dbscan is not None:
        # Analyze DBSCAN clusters
        dbscan_profiles, dbscan_countries = analyze_clusters(df, 'dbscan_cluster', feature_cols)
        
        # Visualize DBSCAN clusters using PCA
        perform_dimensionality_reduction(df, feature_cols, 'dbscan_cluster', method='pca')
    
    # Perform Gaussian Mixture Model clustering
    df, gmm, optimal_components = perform_gmm(df, feature_cols)
    
    # Analyze GMM clusters
    gmm_profiles, gmm_countries = analyze_clusters(df, 'gmm_cluster', feature_cols)
    
    # Visualize GMM clusters using PCA
    perform_dimensionality_reduction(df, feature_cols, 'gmm_cluster', method='pca')
    
    # Save the clustering results
    df.to_csv('covid_clustering_results.csv', index=False)
    print("\nClustering results saved to covid_clustering_results.csv")
    
    print("\nClustering analysis completed successfully!")

if __name__ == "__main__":
    main()
