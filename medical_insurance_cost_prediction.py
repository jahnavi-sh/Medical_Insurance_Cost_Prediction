#understanding the problem statement - 
#Insurance is a policy that helps to cover up all loss or decrease loss in terms of expenses incurred by various risks. 
#A number of variables affect how much insurance costs. Using machine learning algorithms will help provide a computational 
#intelligence approach for predicting insurance cost.  
#Our job is to automate this process of predicting the healthcare insurance cost calculated from the features provided in 
#the dataset. provides a computational intelligence approach for predicting healthcare insurance costs

#workflow for the project -  
#1. load insurance cost data 
#2. data analysis and axploration 
#3. data preprocessing 
#4. train test split 
#5. model used - linear regression model 
#6. model evaluation  

#import libraries 
#linear algebra
import numpy as np 

#data preprocessing and analysis 
import pandas as pd 

#data visualisation 
import matplotlib.pyplot as plt
import seaborn as sns 

#model training and evaluation 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#load data 
insurance_dataset = pd.read_csv(r'insurance_cost.csv')

#view data 
#view first 5 rows of dataset 
insurance_dataset.head()
#the dataset contains the following columns - 
#1. age - age of the patient 
#2. sex - female or male 
#3. bmi - body mass proportion. bmi is the weight of person in kg divided by square of the height of person in cm 
#4. children - number of children the patient has 
#5. smoker - whether the person is smoker or not. yes or no 
#6. region - region of residence of patient 
#7. charges - amount of money charged 

#view total number of rows and columns 
insurance_dataset.shape
#it has 1338 rows (1338 data points) and 7 columns (7 features as mentioned above)

#statistical measures of the data 
insurance_dataset.describe()

#get more information about the dataframe 
insurance_dataset.info()

#view missing values  
insurance_dataset.isnull().sum()
#there are no missing values 

#data visualisation
#distribution of age value 
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()

#gender column 
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('sex distribution')

insurance_dataset['sex'].value_counts()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['sex'])
plt.title('Sex Distribution')
plt.show()

#normal bmi range - 18.5 to 24.9

#children range 
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()

insurance_dataset['children'].value_counts()

#smoker column 
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('Smoker')
plt.show()

insurance_dataset['smoker'].value_counts()

#countplot for region column  
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('Region')
plt.show()

insurance_dataset['region'].value_counts()

#distribution of charges value 
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()

#encoding categorical features 
#sex column 
insurance_dataset.replace({'sex':{'male':0, 'female':1}}, inplace=True)
#smoker column 
insurance_dataset.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)
# region column
insurance_dataset.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)

#split features and target 
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

#train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#train model 
#linear regression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#evaluate model 
#training data
training_data_prediction = regressor.predict(X_train)
#r squared value 
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print ("r squared error", r2_train)
#the r squared error for training data is 0.75

#test data 
test_data_prediction = regressor.predict(X_test)
#r squared value 
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print ("r squared error", r2_test)
#the r squared error for test data is 0.74