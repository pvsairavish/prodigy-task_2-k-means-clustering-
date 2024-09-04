# prodigy-task_2-k-means-clustering-
Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.
1. Importing Necessary Libraries:
pandas: For data manipulation and analysis.
numpy: For numerical operations.
matplotlib.pyplot and seaborn: For data visualization.
KMeans: For performing K-Means clustering from the scikit-learn library.
StandardScaler: For standardizing the features before clustering.
zipfile: For handling ZIP files.
2. Data Reading:
The data is read from a ZIP file (archive.zip) containing a CSV file (Mall_Customers.csv).
df = pd.read_csv(f) reads the CSV file into a pandas DataFrame.
print(df.head()), print(df.info()), and print(df.describe()) provide an initial overview of the dataset, including the first few rows, data types, missing values, and summary statistics.
3. Data Preprocessing:
df = df.dropna() removes any rows with missing values.
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1}) converts the categorical Gender column into numerical values (0 for Male, 1 for Female).
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] selects the relevant features for clustering.
scaler = StandardScaler() initializes a StandardScaler object to standardize the features.
X_scaled = scaler.fit_transform(X) scales the selected features to have a mean of 0 and a standard deviation of 1. This is important for clustering algorithms to ensure that each feature contributes equally to the distance calculations.
4. Determining the Optimal Number of Clusters:
The Elbow Method is used to determine the optimal number of clusters (k).
wcss = [] initializes an empty list to store the within-cluster sum of squares (WCSS) for each value of k.
A for loop runs the K-Means algorithm for k values from 1 to 10:
kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) initializes the K-Means model with i clusters.
kmeans.fit(X_scaled) fits the model to the scaled data.
wcss.append(kmeans.inertia_) appends the WCSS for the current k to the list.
plt.plot(range(1, 11), wcss, marker='o') plots the WCSS values against the number of clusters to visualize the "elbow point" where the WCSS starts to decrease more slowly, indicating the optimal number of clusters.
5. Applying the K-Means Algorithm:
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0) initializes the K-Means model with the selected number of clusters (k=5).
y_kmeans = kmeans.fit_predict(X_scaled) fits the model to the scaled data and predicts the cluster labels for each data point.
6. Visualizing the Clusters:
sns.scatterplot(x=X_scaled[:, 1], y=X_scaled[:, 2], hue=y_kmeans, palette='viridis') creates a scatter plot of the customers based on their scaled annual income and spending score, coloring the points according to their assigned clusters.
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=300, c='red', label='Centroids') plots the centroids of the clusters as red points on the scatter plot.
The plot is labeled and displayed to visualize the customer segments identified by the K-Means algorithm.
Summary:
This code performs customer segmentation using K-Means clustering on a mall customer dataset. After data preprocessing and feature scaling, it determines the optimal number of clusters using the Elbow Method. It then applies the K-Means algorithm to segment the customers and visualizes the results, helping to understand distinct customer groups based on age, annual income, and spending score.
