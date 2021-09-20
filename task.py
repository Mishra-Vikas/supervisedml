
#Importing useful libraries
import numpy as num
import pandas as pd
import matplotlib.pyplot as plt

#importing datas
dataset = pd.read_csv("C:\\Users\\PNDMUME470\\internship\\dataset.txt")

#reading the top datas
dataset.head()



#Plotting the scores
dataset.plot(x='Hours', y='Scores', style='^')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


#extracting features and labels of the data set
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#spliting data into train and test 
from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state= 42)

x_train

#Creating and Training the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train , y_train)

#predicting using the model
yp= model.predict(x_test)

# Plotting the regression line
line = model.coef_*x+model.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line);
plt.show()



# Import library to check accuracy
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(y_test , yp)
print(score)

# test with hours = 9.25
hr = num.array([9.25]).reshape(1, -1)
p= model.predict(hr)
print("Student studies for 9.25 hrs/day")
print(" Score of student = {}".format(p[0]))
