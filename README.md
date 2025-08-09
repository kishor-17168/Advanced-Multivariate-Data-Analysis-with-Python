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

This process demonstrates how PCA can simplify the feature space while maintaining important information for classification tasks.
The full code is available in the [Problem-1_PCA_analysis.ipynb](https://github.com/kishor-17168/Advanced-Multivariate-Data-Analysis-with-Python/blob/main/Problem-1_PCA%20analysis.ipynb) file.



