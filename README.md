# K-Means Clustering Applications

A collection of data science and machine learning notebooks focused on **K-Means Clustering**, exploratory data analysis, cluster visualization, and introductory hierarchical clustering.

This repository was developed as part of the **Applications of Data Science and Artificial Intelligence (DS&AI)** course. It contains practical exercises and applications for learning unsupervised machine learning.

## Repository Overview

The repository includes three notebooks and one dataset:

1. A college dataset project that applies K-Means to explore whether colleges can be grouped into two clusters based on numerical institutional features.
2. A fundamentals notebook that demonstrates K-Means, centroid visualization, the Elbow Method, and an introductory Agglomerative Clustering example.
3. A synthetic-data notebook that generates four clusters and compares the K-Means results with the original generated labels.

## Repository Structure

```text
KMEANS-CLUSTERING-APPLICATIONS/
├── 02-K Means Clustering Project.ipynb
├── K_Means_Clustering(1).ipynb
├── K_MEANS_PYTHON.ipynb
└── College_Data
```

## Files and Their Purpose

| File | Description |
|---|---|
| `02-K Means Clustering Project.ipynb` | College clustering project using K-Means with two clusters, supported by exploratory data analysis and a comparison against the existing Private/Public labels. |
| `K_Means_Clustering(1).ipynb` | Introductory examples of K-Means clustering, centroid visualization, the Elbow Method for selecting K, and Agglomerative Clustering with a dendrogram. |
| `K_MEANS_PYTHON.ipynb` | Synthetic-data example using `make_blobs` to generate four clusters and compare K-Means assignments with the generated labels. |
| `College_Data` | Dataset used in the college clustering project. The notebook loads it as a CSV file and uses the first column as the DataFrame index. |

## 1. College Clustering Project

### Notebook

`02-K Means Clustering Project.ipynb`

This project applies K-Means clustering to a college dataset to investigate whether institutions can be grouped into two clusters that broadly correspond to private and public colleges.

### Dataset and Features

The dataset is loaded using Pandas:

```python
df = pd.read_csv("College_Data", index_col=0)
```

The notebook inspects the dataset using `head()`, `describe()`, and `info()`.

Exploratory analysis focuses on relationships and distributions involving features such as:

- `Room.Board`
- `Grad.Rate`
- `F.Undergrad`
- `Outstate`
- `Private`

### Exploratory Data Analysis

The notebook uses Matplotlib and Seaborn to create visualizations, including:

- A scatter plot comparing graduation rate and room-and-board costs, colored by the `Private` field.
- A scatter plot comparing the number of full-time undergraduates with out-of-state tuition.
- Histograms of out-of-state tuition grouped by private/public status.
- Histograms of graduation rates grouped by private/public status.

The notebook also checks for a graduation rate above 100 and adjusts the value for Cazenovia College to 100 for consistency with the expected percentage range.

### K-Means Model

The model is configured to create two clusters:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop("Private", axis=1))
```

The `Private` label is excluded from model fitting because K-Means is an unsupervised algorithm and should group the colleges using the numerical features rather than the existing category label.

The notebook then examines the cluster centers through:

```python
kmeans.cluster_centers_
```

### Cluster Evaluation

For this learning exercise, the existing `Private` column is converted into numeric labels:

- `Yes` becomes `1`
- Other values become `0`

The notebook compares these labels with the clusters produced by K-Means using a confusion matrix and classification report.

This comparison is for learning and evaluation only. Cluster IDs are arbitrary and do not inherently mean “private” or “public”; the correspondence between cluster numbers and the existing labels must be interpreted carefully. In real unsupervised tasks, ground-truth labels may not be available.

## 2. K-Means Fundamentals and Cluster Selection

### Notebook

`K_Means_Clustering(1).ipynb`

This notebook explores the main mechanics of K-Means using small, illustrative datasets.

### Basic K-Means Example

A small two-dimensional dataset is defined:

```python
x = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])
```

The model is created with two clusters and a fixed random state:

```python
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(x)
```

The notebook displays the assigned cluster labels and centroid coordinates, then visualizes the observations and centroids in a scatter plot.

### The Elbow Method

The notebook evaluates multiple values of K, from 1 through 9, using a simple one-dimensional dataset. For each K value, it fits a K-Means model and records its inertia.

Inertia represents the within-cluster sum of squared distances to the assigned cluster centers. The resulting Elbow plot helps identify a point where increasing the number of clusters produces diminishing reductions in inertia.

The Elbow Method is a practical heuristic rather than a guarantee that one specific K is optimal.

### Agglomerative Clustering

The notebook also introduces hierarchical clustering using SciPy and Scikit-learn.

The data is linked using Ward's method:

```python
linked = linkage(X, method="ward")
dendrogram(linked)
```

It then fits an Agglomerative Clustering model with two clusters and visualizes the resulting labels.

This section provides a comparison between centroid-based K-Means and a hierarchical clustering approach.

## 3. K-Means with Synthetic Data

### Notebook

`K_MEANS_PYTHON.ipynb`

This notebook uses Scikit-learn's `make_blobs` function to generate a synthetic dataset with four groups.

```python
from sklearn.datasets import make_blobs

data = make_blobs(
    n_samples=200,
    n_features=2,
    centers=4,
    cluster_std=1.8,
    random_state=101
)
```

The generated points are visualized using their original generated labels. A K-Means model is then fitted with four clusters:

```python
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
```

The notebook inspects the predicted cluster labels and cluster centers. A side-by-side visualization compares the K-Means assignments with the original labels produced by `make_blobs`.

The cluster numbers assigned by K-Means may differ from the original label numbers even when the grouping is similar, because cluster identifiers are arbitrary.

## Algorithms and Concepts Covered

### K-Means Clustering

K-Means is an unsupervised learning algorithm that partitions observations into K clusters by assigning each observation to a nearby centroid and iteratively updating the centroids.

### Centroids and Cluster Labels

The notebooks inspect:

- `cluster_centers_`: the learned centroid coordinates.
- `labels_`: the cluster assignment for each training observation.

### Choosing K with the Elbow Method

The Elbow Method compares model inertia across different values of K to help assess a reasonable number of clusters.

### Agglomerative Clustering

Agglomerative Clustering builds a hierarchy of clusters by repeatedly merging groups. The notebook uses Ward linkage and a dendrogram to illustrate the hierarchy.

### Exploratory Data Analysis

The college project uses scatter plots and histograms to examine feature relationships and differences between existing college categories before clustering.

### Cluster Evaluation

The college project compares the generated clusters with existing private/public labels using a confusion matrix and classification report. This is an illustrative external comparison, not a requirement of unsupervised learning.

### Synthetic Data Generation

The `make_blobs` function creates a controlled dataset with known generated groups, making it useful for visualizing and understanding clustering behavior.

## Workflow

The projects cover the following general workflow:

```text
Load or Generate Data
        |
        v
Inspect and Visualize Data
        |
        v
Prepare Features
        |
        v
Choose a Clustering Approach
        |
        v
Fit the Clustering Model
        |
        v
Inspect Labels and Centroids
        |
        v
Visualize and Interpret Clusters
        |
        v
Evaluate Where Reference Labels Exist
```

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Installation

Clone the repository:

```bash
git clone https://github.com/mennnaaaaa/KMEANS-CLUSTERING-APPLICATIONS.git
cd KMEANS-CLUSTERING-APPLICATIONS
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open a notebook and run its cells in order. Keep the `College_Data` file in the same directory as the college project notebook, unless you update the file path in the notebook.

## Learning Objectives

This repository provides practice with:

- Unsupervised machine learning
- K-Means model fitting
- Understanding centroids and cluster assignments
- Visualizing two-dimensional clusters
- Using the Elbow Method to explore K
- Introductory hierarchical clustering
- Reading dendrograms
- Exploratory data analysis
- Comparing clusters with reference labels when available
- Interpreting the limitations of clustering evaluation

## Course Context

These applications were developed as part of the **Applications of Data Science and Artificial Intelligence (DS&AI)** course. The notebooks are practical learning exercises focused on clustering algorithms, data exploration, visualization, and model interpretation.

## Future Improvements

Possible extensions include:

- Applying feature scaling to datasets where feature ranges differ substantially.
- Using a fixed `random_state` for reproducible K-Means results.
- Comparing cluster quality using suitable internal metrics such as silhouette score.
- Comparing K-Means with additional clustering algorithms.
- Documenting model settings and evaluation results consistently.
- Organizing notebooks and datasets into dedicated folders.

## Author

**Menna Hany Abdelaziz**

GitHub: [mennnaaaaa](https://github.com/mennnaaaaa)
