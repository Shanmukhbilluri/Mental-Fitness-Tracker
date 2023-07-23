IMPORT NECESSARY PACKAGES
import warnings
warnings.filterwarnings('ignore')
#import all libraries
import pandas as pd   #data processing ,CSV I/O
import numpy as np    #linear algebra
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, BayesianRidge
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# from xgboost import XGBRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.neural_network import MLPRegressor
import seaborn as sns #seaborn in python data visulization library basesd on matplotlib
import matplotlib.pyplot as plt #matplotlib is a low level graph plotting library in python that serves as a visulization utility
import plotly.express as px #allows you to create interactive plots with very little code
ACCESSING CSV FILE
#prevalence-by-mental-and-substance-use-disorder.csv
df1 = pd.read_csv('prevalence-by-mental-and-substance-use-disorder.csv')
#mental-and-substance-use-as-share-of-disease.csv
df2 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')
df1.head()
df2.head()
MERGING TWO DATASETS
data = pd.merge(df1, df2)
data.head(10)
#filling missing values in dataset
data.isnull().sum()
#drop the column
data.drop('Code', axis=1, inplace=True)
VIEW THE DATA
data.head(10)
#size =row*column ,shape=tuple of array dimensions(row,col)
data.size,data.shape
#column set
data.set_axis(['Country','Year','Schizophrenia', 'Bipolar_disorder', 'Eating_disorder','Anxiety','drug_usage','depression','alcohol','mental_fitness'], axis='columns', inplace=True)
data.head(10) #our target or dependent if mental_fitness
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True,cmap='Greens')  #heatmap is defined as graphical representation of data using colors for visual representation of matrix
plt.plot()
sns.jointplot(data,x="Schizophrenia",y="mental_fitness",kind="reg",color="m")
plt.show()
sns.jointplot(data,x='Bipolar_disorder',y='mental_fitness',kind='reg',color='blue')
plt.show()
sns.pairplot(data,corner=True)  #paiwise relation ships in a dataset
plt.show()
mean = data['mental_fitness'].mean()
mean
fig = px.pie(data, values='mental_fitness', names='Year')
fig.show()
fig=px.bar(data.head(10),x='Year',y='mental_fitness',color='Year',template='ggplot2')
fig.show()
fig = px.line(data, x="Year", y="mental_fitness", color='Country',markers=True,color_discrete_sequence=['red','blue'],template='plotly_dark')
fig.show()
df=data.copy()
df.head()
INFORMATION ABOUT THE DATA
df.info()
#transform non-numeric labels to numeric labeles
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object': #transform non-numerical labels (as long as they are hashable and comparable) to numeric labels
        df[i]=l.fit_transform(df[i])
X = df.drop('mental_fitness',axis=1)
y = df['mental_fitness']
from sklearn.model_selection import train_test_split   #used to split the data into training data and testing data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)
#random_state simply set seeds to the random generator,so that your train test splits are always deterministic,if you don't set seed it will different each time
#tainning(6840,10)
#6840*80/100=5472
#6840*20/100=1368
print("xtrain: ", xtrain.shape)
print("xtest: ", xtest.shape)
print("ytrain: ", ytrain.shape)
print("ytest: ", ytest.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(xtrain,ytrain)   #fit trainng data

# model evaluation for training set
ytrain_pred = lr.predict(xtrain)
#the mean square error is the average of the square of the difference between observed and predicted value of a variable
mse = mean_squared_error(ytrain, ytrain_pred)   #observed value and predicted value
#root mean square error measures the average difference between values predicted by model and actua values
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
#the coefficent of determination or R2,is a measure that priovides information about the goodness of fit of a model.In the context of regression it is a statistical measure oif
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain, ytrain)

# model evaluation for training set
ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain, ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain, ytrain_pred)))
r2 = r2_score(ytrain, ytrain_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")
#linear regression model evaluation for testing set
ytest_pred = lr.predict(xtest)  # (unseen data)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print("linear regression model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
# random forest model evaluation for testing set
ytest_pred = rf.predict(xtest)   # (unseen data)
mse = mean_squared_error(ytest, ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest, ytest_pred)))
r2 = r2_score(ytest, ytest_pred)

print(" random forest model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))# Mental-Fitness-Tracker
