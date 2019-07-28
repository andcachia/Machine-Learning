# Predicting Company Bankruptcy with Machine Learning

The data  set  contains financial  information  of  a  number  of  bankrupt  and  non-bankrupt  Polish  companies analysed between the periods 2000-2012 and 2007-2013 respectively.

The available data comes in the form of five files. File 1 represents data for 1st year of the forecasting period.  Every  instance  represents  an  anonymised company  with  a  classification  label  indicating bankruptcy status after 5 years: 0 as non-bankrupt and 1 as bankrupt. Similarly file 2 represents data for 2nd year of the forecast period and each company is classified as non-bankrupt or bankrupt after 4 years, and so on for the other 3 files.

Each data file contains 64 features (or attributes) equivalent to 64 different financial metrics or ratios.

As pre-processing techniques, ive different strategies are applied for each of the two machine learning algorithms using the following methods:

Scaling: sklearn.preprocessing.RobustScaler and sklearn.preprocessing.MinMaxScaler
- Chi Squared Feature Selection
- Principal Component Analysis (PCA): sklearn.decomposition.PCA
- Recursive Feature Elimination (RFE): sklearn.feature_selection.RFE
- Synthetic Minority Over-sampling Technique (SMOTE)

In the 'Strategies-Scoring' notebook, Pipeline  and  GridSearchCV  are used to test out a number of different pre-processing combinations and determine  which techniques and parameters  work  best for this particular problem. For evaluating the results of the experiments, the following methods are used:
- Confusion Matrix 
- Area under ROC curve
- F-score
- Matthew’s Correlation Coefficient (MCC)

Finally, the tranformed input is fed into a custom Logistic Regression algorithm. The implementation managed to achieve 0.826 (year5) AUC by using RFE and SMOTE.