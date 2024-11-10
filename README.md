# Comparing_classifiers
Practical application Assignment  17.1 : comparing Classifiers

OverView :
In this third practical application assignment, your goal is to compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) you encountered in this section of the program. You will use a dataset related to the marketing of bank products over the telephone.

Data:
The dataset you will use comes from the UCI Machine Learning repository Links to an external site.. The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaigns. You can make use of the article Links to an external site.accompanying the dataset (in the .zip file) for more information on the data and features.

Deliverables : 
After understanding, preparing, and modeling your data, build a Jupyter Notebook that includes a clear statement demonstrating your understanding of the business problem, a correct and concise interpretation of descriptive and inferential statistics, your findings (including actionable insights), and next steps and recommendations.

Submission Instructions:
Submit the website URL to your public-facing GitHub repository here
Your Learning Facilitators will grade your submission according to the rubric below

The Analysis of the Logistic Regress, Decision Tree Classifier, KNeighborsClassifier and the Support Vector Machines was performed according to the following criteria

Imbalance Class Handling
Model Training Speed
Interpretable Results
Other criteria observed include

Accuracy
Precision
Recall
Specificity
Mean Squared Error
Logistic Regression Classifier

SMOTE was used to handle imbalanced classes
Speed of Training is moderately high at 0.58s
Train Score performs slightly better than the Test Score
Accuracy and Specificity are not too high at 59. and 33.
Precision is low at 56.%
Recall is high at 85.%
Train and Test MSEs are relatively equal
Decision Tree Classifier

SMOTE was used to handle imbalanced classes
Speed of Training is moderately high at 0.4s
Accuracy and Specificity are very high at 75.9% and 69.2% respectively
Train Score performs slightly better than the Test Score
Precision is high at 72.7%
Recall is low at 82.6%
However, the Decision Tree Classifier appears to overfit as Train MSE is lower than Test MSE
KNearest Neighbors Classifier

SMOTE was used to handle imbalanced classes
Speed of Training is high at 0.45s
Accuracy and Specificity are very high at 72% and 66% respectively
Train Score is slightly higher than the Test Score
Precision is high at 69.8%
Recall is low at 78.4%
However, the KNNeighbors Classifier appears to slightly overfit since Test MSE is higher than Train MSE
Support Vector Machine

SMOTE was used to handle imbalanced classes
Speed of Training is least at 12s
Accuracy and Specificity are very high at 62.8% and 48% respectively
Test Score is lower than Train Score
Precision is 58.%
Recall is high at 77.%
Train and Test MSE are relatively equal
Selecting Best Model: Support Vector Machine
The Support Vector Machine was selected as best model because the Train and Test MSEs are relatively equal.
Contents on Jupyter Notebook Steps
Jupyter Notebook Link
Information about the data
Dropping Unwanted Columns
Encode the whole categorical columns
Get the Correlation Matrix
Plot the Scatter Matrix
Define the Modelling Data
Perform Principal Component Analysis of the Scaled Data
Create four models - Logistic Regression, DecisionTree, KNN and SVC
Evaluate the four models
Summarize the results of the Evaluations
Select the Best Model out of the four based on the summary
Create a Report of the Analysis

Machine Learning Libraries Used on Jupyter Notebook
statsmodels.tsa.filters.filtertools as ft
sklearn.metrics import mean_squared_error
statsmodels.tsa.filters.filtertools import convolution_filter
sklearn.feature_selection import SequentialFeatureSelector
statsmodels.tsa.seasonal import _extrapolate_trend
pandas.testing as tm
statsmodels.tsa.arima_process as arima_process
statsmodels.graphics.tsaplots as tsaplots
numpy as np
pandas as pd
matplotlib.pyplot as plt
statsmodels.api as sm
statsmodels.tsa.seasonal import seasonal_decompose
sklearn.preprocessing import OneHotEncoder
sklearn.pipeline import Pipeline
sklearn.preprocessing import StandardScaler
sklearn.impute import SimpleImputer
sklearn.compose import ColumnTransformer
sklearn.decomposition import PCA
sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
sklearn.pipeline import make_pipeline
sklearn.model_selection import train_test_split, GridSearchCV
sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder
sklearn.linear_model import Ridge
scipy import stats
scipy.linalg import svd
warnings
warnings.filterwarnings('ignore')
