import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#print working directory 
print("Current working directory: ", os.getcwd())
#using synthetic lung cancer data from:https://www.kaggle.com/datasets/rashadrmammadov/lung-cancer-prediction/data?select=lung_cancer_data.csv 

#pandas pd.dataFrame column: [rows], is a dictionary with keys as columns, with a list as the different 'rows' values
#Use index to name the row names

'''
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
'''

#pd.Series (creates a list - single column of a dataframe) 

lung_cancer_data = pd.read_csv("/Users/jason/Documents/Interests/Artifical Intelligence/Machine Learning/Modelling Practice/lung_cancer_data.csv")

#Specify index column with index_col = col_index


#Prelim look 
print(lung_cancer_data.head())
print(lung_cancer_data.columns)
print(lung_cancer_data.shape) #23658 row * 38 columns
print(lung_cancer_data.describe()) #summary statistics of numerical columns


#Data Cleaning and Preprocessing 

#missing variables -> there is none in this set
missing_counts = lung_cancer_data.isnull().sum()
print(missing_counts)
#If there was and you wanted to look at percentage of missing values
missing_percentage = lung_cancer_data.isnull().mean() * 100
print(missing_percentage)

print(lung_cancer_data.dtypes)
'''
Patient_ID                           object
Age                                   int64
Gender                               object
Smoking_History                      object
Tumor_Size_mm                       float64
Tumor_Location                       object
Stage                                object
Treatment                            object
Survival_Months                       int64
Ethnicity                            object
Insurance_Type                       object
Family_History                       object
Comorbidity_Diabetes                 object
Comorbidity_Hypertension             object
Comorbidity_Heart_Disease            object
Comorbidity_Chronic_Lung_Disease     object
Comorbidity_Kidney_Disease           object
Comorbidity_Autoimmune_Disease       object
Comorbidity_Other                    object
Performance_Status                    int64
Blood_Pressure_Systolic               int64
Blood_Pressure_Diastolic              int64
Blood_Pressure_Pulse                  int64
Hemoglobin_Level                    float64
White_Blood_Cell_Count              float64
Platelet_Count                      float64
Albumin_Level                       float64
Alkaline_Phosphatase_Level          float64
Alanine_Aminotransferase_Level      float64
Aspartate_Aminotransferase_Level    float64
Creatinine_Level                    float64
LDH_Level                           float64
Calcium_Level                       float64
Phosphorus_Level                    float64
Glucose_Level                       float64
Potassium_Level                     float64
Sodium_Level                        float64
Smoking_Pack_Years                  float64'''
#ContinuousVars 
continuous_cols = ['Age', 'Tumor_Size_mm', 'Survival_Months', 'Blood_Pressure_Systolic', 
                   'Blood_Pressure_Diastolic', 'Blood_Pressure_Pulse', 'Hemoglobin_Level', 
                   'White_Blood_Cell_Count', 'Platelet_Count', 'Albumin_Level', 
                   'Alkaline_Phosphatase_Level', 'Alanine_Aminotransferase_Level', 
                   'Aspartate_Aminotransferase_Level', 'Creatinine_Level', 'LDH_Level', 
                   'Calcium_Level', 'Phosphorus_Level', 'Glucose_Level', 
                   'Potassium_Level', 'Sodium_Level', 'Smoking_Pack_Years']
#Categorical Vars 
categorical_cols = ['Gender', 'Smoking_History', 'Tumor_Location', 'Stage', 'Treatment','Ethnicity', 'Insurance_Type', 'Family_History', 'Comorbidity_Diabetes', 
                     'Comorbidity_Hypertension', 'Comorbidity_Heart_Disease', 'Comorbidity_Chronic_Lung_Disease', 
                     'Comorbidity_Kidney_Disease', 'Comorbidity_Autoimmune_Disease', 'Comorbidity_Other', 'Performance_Status' ]

print (len(continuous_cols))
print(len(categorical_cols)) 



#Visualisation of relationships between outcome variable (survival in months) and other variables
 
#sns.histplot(lung_cancer_data['Survival_Months'], bins=30, kde=True)
#plt.xlabel('Survival Months')
#plt.ylabel('Number of Patients')
#plt.show()
'''for col in continuous_cols:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=lung_cancer_data[col], y=lung_cancer_data['Survival_Months'])
    sns.regplot(x=lung_cancer_data[col], y=lung_cancer_data['Survival_Months'], scatter=False, color='red')  # optional regression line
    plt.title(f'{col} vs Survival')
    plt.xlabel(col)
    plt.ylabel('Survival Months')
    plt.show()



for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=col, y='Survival_Months', data=lung_cancer_data)
    plt.title(f'{col} vs Survival')
    plt.xlabel(col)
    plt.ylabel('Survival Months')
    plt.show()
'''

#issue with this data, is that it is too perfectly balanced, i don't think i can visualise any trends in the preliminary plots


#Prepare data for modelling -> scale variables, drop first variable, avoids the dummy variable trap which allows linear regression to predict 
lung_cancer_data_encoded = pd.get_dummies(lung_cancer_data, columns = categorical_cols, drop_first=True)

for col in categorical_cols:
    print(f"{col} - Unique values: {lung_cancer_data[col].unique()}")
    print(f"Value counts: \n {lung_cancer_data[col].value_counts()}\n")


#now to split the testing data, scale after the split (to avoid data leakage!)
X = lung_cancer_data_encoded.drop(['Patient_ID', 'Survival_Months'], axis=1)
y = lung_cancer_data['Survival_Months']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

continuous_cols_scaled = [col for col in continuous_cols if col in X_train.columns]
print("X-train column names: ", X_train.columns )

scaler = StandardScaler()
X_train[continuous_cols_scaled] = scaler.fit_transform(X_train[continuous_cols_scaled])
X_test[continuous_cols_scaled] = scaler.transform(X_test[continuous_cols_scaled])


#Now build the model - using sci-kit learn first, then build manually and then compare 

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
mse = mean_squared_error (y_test,y_pred_lr)
print("Linear Regression MSE:", mse)
print("Linear Regression RMSE:", np.sqrt(mse))
print("R2:", r2_score(y_test, y_pred_lr))


#Build the function manually y = Xw +b 

def compute_cost (X, y, weights, bias):
    #X is vector
    #y is vector
    #weights is weight vector
    #bias = b 
    n = len(y)
    predictions = np.dot(X, weights) + bias
    cost = (1/(2*n)) * np.sum((predictions - y) ** 2)
    return cost

def compute_gradients(X, y, weights, bias):
    #X is a vector matrix
    #y is vector
    n = len(y)
    predictions = np.dot(X, weights) + bias
    dw = (1/n) * np.dot(X.T, (predictions - y))
    db = (1/n) * np.sum(predictions - y)
    return dw, db

#epoch = one complete pass through the entire training dataset 
def gradient_descent(X, y, learning_rate, epochs):
    #epochs 
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    for epoch in range(epochs):
        dw, db = compute_gradients(X, y, weights, bias)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        if epoch % 100 == 0:
            cost = compute_cost(X, y, weights, bias)
            print(f"Epoch {epoch}, Cost: {cost}")
    return weights, bias

X_train_np = X_train.values.astype(np.float64)
y_train_np = y_train.values.astype(np.float64)
learning_rate = 0.03
epochs = 10000


weights, bias = gradient_descent(X_train_np, y_train_np, learning_rate, epochs)
print("Weights:", weights)
print("Bias:", bias)

y_pred_manual = X_test.values.dot(weights) + bias
mse_manual = mean_squared_error(y_test, y_pred_manual)
print("Manual Linear Regression MSE:", mse_manual)  
print("Manual Linear Regression RMSE:", np.sqrt(mse_manual))
print("Manual R2:", r2_score(y_test, y_pred_manual))

print("Linear Regression MSE:", mse)
print("Linear Regression RMSE:", np.sqrt(mse))
print("R2:", r2_score(y_test, y_pred_lr))


#Plotting different survival
plt.scatter(y_test, y_pred_manual, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Survival")
plt.ylabel("Predicted Survival")
plt.title("Manual Linear Regression Predicted vs True")
plt.savefig("predicted_vs_true.png")

feature_importance = pd.Series(weights, index=X_train.columns).sort_values(ascending=False)
print(feature_importance.head(10))

#to write a csv
#lung_cancer_data.to_csv("lung_cancer_data_output.csv", index=False) 

