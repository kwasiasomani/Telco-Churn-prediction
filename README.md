# Telco-Churn-prediction
A classification machine learning problem for predicting customers churn from the company based on customers who left within the last month labeled by 'yes' or 'no'

The dataset used in this project is obtained from Kaggle - Telco Customer Churn
The data set includes information about:

Customers who left within the last month – the column is called Churn
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents
Methodology
At first 20% of the data were splitted for final testing;  by the 'Churn' (target) column.

Data cleaning
Convert 'TotalCharges' column which is of object type to float type using pd.to_numeric() with errors parameter set to 'coerce' to parse invalid data to NaN.
Eight missing values were found in the 'TotalCharges' column and were imputed by the mean() value.
Data has  duplicates.
Exploratory data analysis
pie chart and barchart shows the distribution of the churn rate in the data which showed an imbalance in the data.

Data is evenly distributed between the two genders; males and females, which might be useful in further analysis.
No information added by 'No Internet Service' or 'No Phone Service' and 'No' categories. --> Replacing 'No Internet Service' and 'No Phone Service' entries with 'No'.
Histogram and box plot of continous features implies that:
No outliers exists.
'TotalCharges' feature is right skewed.
Scatter plot of 'MonthlyCharges' vs. 'TotalCharges' shows a positive correlation between both and also it affects the Churn rate positively.
Feature encoding
Several encoding techniques were tested on each categorical feature separately and One-Hot encoding all the categorical features gave the best results.

Feature engineering
Label encoder was used for the target variable and onehotencoder was used for the rest of the categorical data
Feature scaling
   StandaredScaler() was used for the numerical columns.

Data imbalance
Data imbalance affects machine learning models by tending only to predict the majority class and ignoting the minority class, hence, having major misclassification of the minority class in comparison with the majority class. Hence, we use techniques to balance class distribution in the data.

Even that our data here have severe class imbalance, but handling it shows results improvement. Using Oversampling Technique 
OverSampler creates new records of the minority class by randomly adjust the class distribution of a data set
Preparing a python function test_prep(dataframe) to combine and apply all previous preprocessing steps on the test data.
To handle any expected missing values in the test set, the missing values was few so we dropped it.
Models training
six different models were applied on the data and all results are reported with confusion matrix and classification report showing the precision, recall, and f1-score metrics.

Logistic Regression
RandomForest Classifier
XGBoost Classifier
K Nearest Neighbors
Support Vector Machines
DecisionTreeClassifier

Model evaluation wa done using K Fold  cross validation

The tuning was done using K fold and GridsearchCV
