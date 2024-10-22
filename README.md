# Heart Disease Prediction Project

## Overview

This project provides an indication of the likelihood of developing heart disease using various analysis models, including Decision Tree, Logistic Regression, Naïve Bayes, and Support Vector Machine (SVM). Dimensionality reduction is performed using Principal Component Analysis (PCA), and the performance of the models is evaluated through various metrics such as Accuracy, Precision, Recall, ROC curve, and more.

## Features

- Utilizes multiple machine learning algorithms for prediction:
  - Decision Tree
  - Logistic Regression
  - Naïve Bayes
  - Support Vector Machine (SVM)
  
- Implements PCA for dimensionality reduction.

- Evaluates model performance with:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC Score

## Getting Started

### Prerequisites

To run this project, ensure you have the following libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly

Please make sure to adjust the path to the data files in the code. The dataset required for this project includes:

-AMI_GSE66360_series_matrix.csv: Contains the gene expression data
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE66360

-type.csv: Contains labels for the data (indicating the presence or absence of heart disease).

##Results
The performance of each model will be displayed in the console and will include:
Confusion matrices for each classifier.
Accuracy, Precision, Recall, F1 Score, and ROC AUC Score metrics.
Visualizations of the performance metrics.

##Conclusion
This project aims to explore and demonstrate the effectiveness of different machine learning algorithms in predicting heart disease. The results and insights gained can serve as a foundation for further research and improvement in health informatics.
