# Assignment of Advanced Multivariate Data Analysis
![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/coverpage.png?raw=true)


## Table of Contents
1. [Introduction](#1--introduction)
2. [Tools used in this Assignment](#2--tools-used-in-this-assignment)
3. Problem-1: PCA Analysis


## 1- Introduction
This repository contains the completed assignment for the course **Advanced Multivariate Techniques**. The primary objective of this assignment is to apply and demonstrate various multivariate statistical methods using Python, based on the topics covered in the course.  

The tasks include practical implementations of methods such as **Principal Component Analysis (PCA)**, **Canonical Correlation Analysis (CCA)**, **Factor Analysis**, **DBSCAN clustering**, **Multidimensional Scaling (MDS)**, and **Correspondence Analysis**. Each problem is solved using relevant datasets, with step-by-step coding, visualization, and interpretation of results.  

The assignment aims to reinforce theoretical concepts through computational practice, enhance analytical skills, and develop proficiency in applying multivariate techniques to real-world data scenarios.


## 2- Tools used in this Assignment
The following tools and libraries were used to complete the tasks in this assignment:

- **Jupyter Notebook** – Interactive environment for writing and executing Python code, documenting workflows, and visualizing results.
- **NumPy** – For efficient numerical computations and array manipulations.
- **Pandas** – For data manipulation, cleaning, and handling tabular datasets.
- **Matplotlib** – For creating basic data visualizations.
- **Seaborn** – For advanced and aesthetically pleasing statistical plots.
- **Scikit-learn** – For implementing machine learning algorithms such as PCA, DBSCAN, and MDS.
- **FactorAnalyzer** – For conducting factor analysis and related statistical methods.


## 3- Principle Component Analysis(PCA)
Principal Component Analysis (PCA) is a widely used dimensionality reduction technique in multivariate statistics. It transforms the original set of possibly correlated variables into a smaller set of uncorrelated variables known as principal components. These components are ordered such that the first captures the maximum possible variance in the data, the second captures the next highest variance orthogonal to the first, and so on. PCA is valuable for simplifying datasets, reducing noise, and enabling better visualization while preserving most of the important information in the original data.

**When to Use:**  
PCA is most useful when dealing with high-dimensional datasets, when multicollinearity exists among variables, or when a more compact representation is needed for visualization, pattern recognition, or as a preprocessing step before applying other machine learning algorithms.

In this analysis, a sample dataset with features—Height, Weight, and Age—was used to predict Gender (binary: Male/Female). The key steps were:

- **Data Preparation:** The dataset was created and converted into a pandas DataFrame. Features (Height, Weight, Age) were separated from the target variable (Gender).
- **Feature Scaling:** StandardScaler was applied to standardize the features, ensuring equal weighting.
- **Dimensionality Reduction:** PCA was performed to reduce the original three features into two principal components that capture the majority of variance.
- **Model Training:** Logistic Regression was trained on the PCA-transformed training data to classify Gender.
- **Model Evaluation:** The model’s performance was evaluated on the test set using a confusion matrix, visualized with a heatmap.
- **Visualization:** Scatter plots were generated to compare the data distribution before and after PCA, showing how PCA projects data into a lower-dimensional space while preserving class separation.

This process demonstrates how PCA can simplify the feature space while maintaining important information for classification tasks.<br>

![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-1_output.png)
The full code is available in the [Problem-1_PCA_analysis.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-1_PCA%20analysis.ipynb) file.<br>
<br>Go back to **Table of Contents** [here](#table-of-contents)<br>

## 4- Canonical Correlation analysis(CCA)
Canonical Correlation Analysis (CCA) is a multivariate statistical technique used to identify and quantify the relationships between two sets of variables. It finds pairs of linear combinations—called canonical variates—from each variable set that are maximally correlated with each other. These canonical correlations help uncover shared patterns or associations between the two datasets.

**When to Use:**  
CCA is useful when you want to explore the relationship between two different sets of variables measured on the same samples. It is commonly applied in fields like psychology, genomics, and social sciences, where understanding the interplay between multiple variable groups is essential.

In this analysis, synthetic datasets representing psychological scores (X) and physiological scores (Y) were created with multiple variables. The goal was to explore the relationships between these two variable groups using Canonical Correlation Analysis (CCA).

Key steps included:
- Generating and standardizing two datasets with correlated variables.
- Applying CCA to find pairs of canonical variates (linear combinations) from each dataset that maximize their correlation.
- Visualizing the first pair of canonical variates with a scatter plot to observe their association.
- Calculating canonical correlation coefficients, which quantify the strength of the relationships between the pairs of canonical variates.

The resulting canonical correlations were approximately 0.51 and 0.35 for the first two canonical variate pairs, indicating moderate associations between the psychological and physiological variables. This demonstrates how CCA can uncover underlying connections between two multivariate datasets.

![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-2_output.png) <br>
The canonical correlation analysis yielded two canonical correlations: **0.5118** and **0.3526**.  
The first canonical variate pair, with a correlation of approximately **51.2%**, represents the strongest relationship between the two variable sets, indicating that about **26%** (\(R^2 = 0.2619\)) of the variance in one canonical variate is explained by the other.  
The second canonical variate pair has a weaker correlation of about **35.3%**, corresponding to roughly **12%** (\(R^2 = 0.1243\)) shared variance, suggesting that it captures less of the common structure between the datasets compared to the first pair.<br>

The full code is available in the [Problem-2_ Canonical Correlation analysis.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-2_%20Canonical%20Correlation%20analysis.ipynb) file.<br>
<br>Go back to **Table of Contents** [here](#table-of-contents)<br>



## 5- Visualizing Covariance matrix
### What is a Covariance Matrix?
A covariance matrix shows how pairs of variables vary together:

* Positive covariance → variables increase together

* Negative covariance → one increases, the other decreases

* Zero covariance → variables are uncorrelated

It helps understand relationships between multiple variables.

**Iris dataset from seaborn is used**

This analysis involved computing and visualizing the covariance matrix of the well-known Iris dataset, focusing on its numeric features: sepal length, sepal width, petal length, and petal width.  

The covariance matrix quantifies how pairs of variables vary together:  
- Positive covariance indicates variables increase or decrease together.  
- Negative covariance indicates one variable increases while the other decreases.  
- Values near zero indicate little or no linear relationship.  

A heatmap was used to visualize these covariances, where red cells represent positive relationships and blue cells indicate negative relationships. Notably, petal length and sepal length exhibit a strong positive covariance (1.27), while sepal width and petal length show a negative covariance (-0.33). The diagonal elements represent variances, with petal length having the highest variance (3.12), reflecting its wide range of values.

This visualization helps in understanding the strength and direction of relationships among variables, which is critical for multivariate data analysis.
![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-3_output.png) <br>

The full code is available in the [Problem-3_Visualizing Covariance matrix.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-3_Visualizing%20Covariance%20matrix.ipynb) file.<br>
<br>Go back to **Table of Contents** [here](#table-of-contents)<br>




## 6- Factor analysis with student stress data
Factor Analysis is a multivariate statistical technique used to identify underlying latent factors that explain the patterns of correlations among observed variables. It assumes that observed variables are influenced by a smaller number of unobserved factors, which capture the shared variance.

The main objectives of Factor Analysis are to reduce dimensionality, detect structure in the relationships between variables, and identify latent constructs that may not be directly measured.

Factor Analysis is widely applied in fields such as psychology, social sciences, and marketing research, where complex data can be summarized by a few meaningful factors.

**When to Use:**  
Use Factor Analysis when you want to explore underlying dimensions or constructs in a dataset with many correlated variables, especially when those factors are not directly observable. It is particularly useful for data reduction and for developing theoretical models based on latent variables.

This analysis applied Factor Analysis to a student stress dataset containing 21 variables related to psychological, physical, and social factors affecting stress levels.

**Key Steps and Findings:**

- **Data Suitability:**
```python
# Suitability tests
kmo_all, kmo_model = calculate_kmo(X)
chi_square_value, bartlett_p = calculate_bartlett_sphericity(X)
print(f"\nKMO: {kmo_model:.3f}   Bartlett p-value: {bartlett_p:.5f}")

# Output:
# KMO: 0.968   Bartlett p-value: 0.00000
```
 
  The Kaiser-Meyer-Olkin (KMO) measure was excellent at 0.968, and Bartlett’s test was highly significant (p < 0.001), confirming the dataset’s suitability for factor analysis.

- **Factor Selection:**
  ![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-4_output.png)
  
  - The scree plot and eigenvalues indicated two meaningful factors explaining most variance.

- **Factor Loadings and Interpretation:**
  ![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-4.1_output.png)
  
  - *Factor 1 ("Psychological & Social Stress")* explains 49.24% of variance and is characterized by high positive loadings on anxiety, depression, bullying, peer pressure, and future career concerns, with strong negative loadings on self-esteem, sleep quality, safety, academic performance, and teacher-student relationship. This factor represents overall mental and emotional well-being.  
  - *Factor 2 ("Physical Health & Social Support")* explains 13.80% of variance and relates strongly to blood pressure and social support, capturing physical health and perceived social support dynamics.

- **Variance Explained:**  
  Together, the two factors explain approximately 63.05% of the variance, which is considered strong for social science data.

- **Factor Scores:**  
  Each participant’s scores on these latent factors indicate their relative standing. For example, Participant 0 showed slightly higher psychological stress but lower physical health/social support issues, whereas Participant 1 showed high stress on both factors.

This analysis effectively reduced complex, correlated variables into two interpretable latent factors, aiding in understanding the dimensions underlying student stress.<br>

The full code is available in the [Problem-4_Factor analysis with student stress data.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-4_Factor%20analysis%20with%20student%20stress%20data.ipynb) file.<br>
<br>Go back to **Table of Contents** [here](#table-of-contents)<br>



## 7- Factor analysis with Mtcars data
This analysis applied Factor Analysis to the mtcars dataset, which contains multiple numeric variables related to car attributes.

**Key Findings:**

- **Data Suitability:**
```python
# Suitability tests
 kmo_all, kmo_model = calculate_kmo(X)
 chi_square_value, p_value = calculate_bartlett_sphericity(X)
 print(f"KMO: {kmo_model:.3f}  |  Bartlett's p-value: {p_value:.5f}")
 KMO: 0.846  |  Bartlett's p-value: 0.0000
```
  The Kaiser-Meyer-Olkin (KMO) measure was 0.846, indicating good sampling adequacy. Bartlett’s test was highly significant (p < 0.001), confirming that the dataset is appropriate for factor analysis.

- **Factor Selection:**
![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-5.1_output.png)<br>

The scree plot and eigenvalues suggested two main factors with eigenvalues greater than 1, which were retained for further analysis.

- **Factor Loadings and Interpretation:**
<br> ![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-5.2_output.png) <br>

  - *Factor 1* loads positively on horsepower (hp), carburetors (carb), cylinders (cyl), and displacement (disp), and negatively on miles per gallon (mpg), quarter mile time (qsec), and engine shape (vs). This factor represents **Engine Size and Power versus Fuel Efficiency**—larger, more powerful engines tend to have lower fuel efficiency.  
  - *Factor 2* loads positively on transmission type (am), number of gears (gear), and rear axle ratio (drat), and negatively on weight (wt) and displacement (disp). This factor reflects **Performance and Transmission Type**, capturing lighter, sportier cars with manual transmissions and higher gear counts.

- **Variance Explained:**  
  Factor 1 explains 41.6% of the variance, Factor 2 explains 37.2%, and together they account for 78.8% of the total variability, summarizing most of the important patterns in the data.

This factor analysis effectively reduced the dimensionality of the mtcars dataset into two meaningful latent factors, aiding in the interpretation of the underlying car characteristics.<br>

The full code is available in the [Problem-5_Factor analysis with Mtcars data.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-5_Factor%20analysis%20with%20Mtcars%20data.ipynb) file.<br>

<br>Go back to **Table of Contents** [here](#table-of-contents)<br>



## 8- Problem-6_Factor Analysis with Iris data
This analysis performed Factor Analysis on the Iris dataset, which contains measurements of different flower attributes.

**Key Findings:**

- **Data Suitability:**
```python
# Suitability tests
 kmo_all, kmo_model = calculate_kmo(X)
 chi_square_value, p_value = calculate_bartlett_sphericity(X)
 print(f"KMO: {kmo_model:.3f}  |  Bartlett's p-value: {p_value:.5f}")
 KMO: 0.540  |  Bartlett's p-value: 0.00000
```  
  The Kaiser-Meyer-Olkin (KMO) measure was 0.540, indicating mediocre sampling adequacy. Bartlett’s test was highly significant (p < 0.001), confirming that factor analysis is appropriate despite the lower KMO.

- **Factor Selection:**
  <br>![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-6.1_output.png)<br>
  <br>The scree plot and eigenvalues suggested retaining only one factor with an eigenvalue greater than 1.<br>

- **Factor Loadings and Interpretation:**
  ![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-6.2_output.png) 
<br>The single factor showed strong negative loadings on sepal length, petal length, and petal width, and a moderate positive loading on sepal width. This factor likely represents a general size or shape dimension of the flowers.<br>

- **Variance Explained:**  
  This factor explains approximately 69.2% of the total variance in the dataset, capturing most of the variability in the measurements.

This factor analysis simplified the Iris dataset into one meaningful latent factor that summarizes the main variation in flower features.<br>

The full code is available in the [Problem-6_Factor Analysis with Iris data.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-6_Factor%20Analysis%20with%20Iris%20data.ipynb) file.<br>
<br>Go back to **Table of Contents** [here](#table-of-contents)<br>



## 9- DBSCAN using python
**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is an unsupervised clustering algorithm that groups together points that are closely packed while marking points in low-density regions as outliers or noise.

**Key Features:**

- **Density-based:** Clusters are formed based on areas of high point density.
- **No need to specify number of clusters:** Unlike k-means, DBSCAN does not require predefining the number of clusters.
- **Handles noise:** Effectively identifies outliers as noise points.
- **Clusters can have arbitrary shape:** Can find clusters of various shapes and sizes.

**When to use DBSCAN:**

- When you expect clusters to have irregular shapes.
- When your dataset contains noise or outliers.
- When you do not want to specify the number of clusters beforehand.


In this analysis, we applied the DBSCAN clustering algorithm to a synthetic dataset generated with two interleaving half circles ("moons"). 

- **Data Preparation:**  
  We generated 300 samples with some noise and standardized the features using `StandardScaler` to ensure all features contribute equally.

- **DBSCAN Application:**  
  DBSCAN was run with parameters `eps=0.3` (neighborhood radius) and `min_samples=5` (minimum points to form a dense region).

- **Results Visualization:**
<br> ![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-7_output.png)<br>
  The clusters assigned by DBSCAN were visualized in a scatter plot with different colors representing different clusters. Points labeled `-1` represent noise or outliers.

- **Insights:**  
  DBSCAN effectively identified the two distinct moon-shaped clusters and separated noise points, demonstrating its ability to find arbitrarily shaped clusters and handle noise.

The full code is available in the [Problem-7_DBSCAN using python.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-7_DBSCAN%20using%20python.ipynb) file.<br>
<br>Go back to **Table of Contents** [here](#table-of-contents)<br>




## 10- Multidimensional Scaling with Iris dataset

Multidimensional Scaling (MDS) is a dimensionality reduction technique used to visualize the similarity or dissimilarity of data points in a lower-dimensional space. It aims to preserve the pairwise distances between points as much as possible while projecting high-dimensional data into 2D or 3D for easier interpretation.

**When to use MDS:**  
- To explore and visualize the structure of complex datasets.  
- When you want to represent the similarity or dissimilarity (distance) between samples.  
- Useful in fields like psychology, ecology, and marketing to interpret relationships among objects or variables.


This analysis applies Multidimensional Scaling (MDS) to the Iris dataset to visualize its structure in two dimensions while preserving the pairwise Euclidean distances between samples.

**Key Points:**
<br> ![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-8_output.png) <br> 

- The original 4-dimensional Iris data was transformed into 2D coordinates using MDS.
- A scatter plot of the two MDS dimensions shows clear clustering by species (setosa, versicolor, virginica).
- This visualization helps reveal the natural grouping and similarity patterns among the different Iris species based on their measured features.

MDS effectively reduces dimensionality and provides an interpretable visual summary of complex, high-dimensional data.<br>
<br>The full code is available in the [Problem-8_Multidimensional Scaling with Iris dataset.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-8_Multidimensional%20Scaling%20with%20Iris%20dataset.ipynb) file.<br>
<br>Go back to **Table of Contents** [here](#table-of-contents)<br>



## 11- Correspondence Analysis with USArrests dataset

Correspondence Analysis is a multivariate statistical technique used to analyze and visualize relationships in categorical data, typically from contingency tables. It provides a way to explore associations between rows and columns by representing them as points in a low-dimensional space.

**When to Use:**  
- To examine relationships between two categorical variables.  
- To visualize patterns in frequency or count data.  
- When you want to reduce dimensionality and interpret associations in a contingency table.

CA helps reveal underlying structures and clusters by projecting categories into a simplified graphical form, making complex categorical relationships easier to understand.

### Correspondence Analysis with USArrests Dataset

This analysis applied Correspondence Analysis (CA) on the USArrests dataset to explore relationships between US states and crime variables.

**Key Findings:**

**Eigenvalues & Variance Explained:**
  ```python
# Eigenvalues (explained inertia)
eigenvalues = ca.eigenvalues_
print("Eigenvalues:", eigenvalues)
# Output:
# Eigenvalues: [0.04501357 0.00606546]
``` 
  The first two dimensions have eigenvalues of approximately 0.045 and 0.006, respectively, indicating the amount of explained inertia (variance) by each dimension.

- **Row and Column Coordinates:**  
  - Row coordinates represent the positions of states in the reduced 2D space.  
  - Column coordinates represent crime variables (Murder, Assault, UrbanPop, Rape).

- **Biplot Interpretation:**
<br> ![Alt Text](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/prob-9_output.png) <br>
  - States (blue dots) and variables (red crosses) positioned close together indicate strong associations.  
  - For example, *UrbanPop* and *Assault* are near states like Massachusetts and New Jersey, suggesting higher urbanization and assault rates there.  
  - *Murder* is closer to southern states such as Georgia, Tennessee, and Louisiana, showing stronger association with this crime.  
  - *Rape* aligns near Nevada and Colorado, indicating higher occurrence in these states.

The CA biplot reveals patterns of similarity among states and highlights which crime types are more prevalent in specific regions, helping to understand the multivariate relationships in the dataset.

<br>The full code is available in the [Problem-9_Correspondance Analysis with USArrests dataset.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-9_Correspondance%20Analysis%20with%20USArrests%20dataset.ipynb) file.<br>
<br>Go back to **Table of Contents** [here](#table-of-contents)<br>
