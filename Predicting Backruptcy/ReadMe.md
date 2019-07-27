# Predicting Company Bankruptcy with Machine Learning

The data  set  contains financial  information  of  a  number  of  bankrupt  and  non-bankrupt  Polish  companies analysed between the periods 2000-2012 and 2007-2013 respectively.

The available data comes in the form of five files. File 1 represents data for 1styear of the forecasting period.  Every  instance  represents  an  anonymisedcompany  with  a  classification  label  indicating bankruptcy status after 5 years: 0 as non-bankrupt and 1 as bankrupt. Similarly file 2 represents data for 2ndyear of the forecast period and each company is classified as non-bankrupt or bankrupt after 4 years, and so on for the other 3 files.

Each data file contains 64 features (or attributes) equivalent to 64 different financial metrics or ratios.

As pre-processing techniques, we apply five different strategies for each of the two machine learning algorithms using the following methods:

Scaling: sklearn.preprocessing.RobustScaler and sklearn.preprocessing.MinMaxScaler
- Chi Squared Feature Selection
- Principal Component Analysis (PCA): sklearn.decomposition.PCA
- Recursive Feature Elimination (RFE): sklearn.feature_selection.RFE
- Synthetic Minority Over-sampling Technique (SMOTE)

We  use  Pipeline  and  GridSearchCV  to  determine  which  parameters  work  best  as  explained  in  the Experiments section. For evaluating the results of the experiments, we use the following methods:
- Confusion Matrix 
- Area under ROCcurve
- F-score
- Matthew’s Correlation Coefficient (MCC)

Finally, the tranformed input is fed into a Logistic Regression algorithm. In our implementation we managed to achieve 0.826 (year5) AUC by using RFE and SMOTE.