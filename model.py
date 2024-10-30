import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Load the dataset
try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    print("The file 'Mall_Customers.csv' was not found. Please check the file path.")
else:
    # Display the first few rows of the dataset
    

    # Select features for clustering
    df_features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Scale the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_features)

    # Determine optimal k using the elbow method
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()

    # Set the optimal k manually or based on analysis
    optimal_k = 5  # Update based on elbow method observation

    # Perform KMeans clustering with the optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(df_scaled)

    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Evaluate the clustering using the silhouette score
    silhouette_avg = silhouette_score(df_scaled, kmeans.labels_)
    print(f'Silhouette Score: {silhouette_avg:.3f}')

    # Optional: Adjusted Rand Index (if ground-truth labels are available)
    if 'True_Label' in df.columns:  # Replace 'True_Label' with the actual column name if available
        ari_score = adjusted_rand_score(df['True_Label'], df['Cluster'])
        print(f'Adjusted Rand Index (ARI): {ari_score:.3f}')
    else:
        print("Adjusted Rand Index (ARI) cannot be computed as there are no ground-truth labels in the dataset.")

    # Enhanced visualization of the clusters
    plt.figure(figsize=(10, 8))

    # Scatter plot of customers colored by their cluster labels
    sns.scatterplot(
        x=df['Annual Income (k$)'],
        y=df['Spending Score (1-100)'],
        hue=df['Cluster'],
        palette='viridis',
        style=df['Cluster'],
        s=80,
        edgecolor='black'
    )

    # Overlay the cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(
        centers[:, 0], centers[:, 1],
        c='red',
        s=200,
        alpha=0.75,
        marker='X',
        label='Cluster Centers'
    )

    # Add titles and labels
    plt.title(f'Customer Segments Based on Income and Spending (k={optimal_k})', fontsize=16)
    plt.xlabel('Annual Income (k$)', fontsize=14)
    plt.ylabel('Spending Score (1-100)', fontsize=14)
    plt.legend(title='Cluster', fontsize=12, title_fontsize='13')
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.colorbar(label='Cluster')

    # Show the plot
    plt.show()
