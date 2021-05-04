import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
#print(data_X)
#print(data_Y)
#Transform the given matrix to a column vector, by taking only the third column features data
diabetes_X = diabetes_X[:, np.newaxis, 2]
#print(data_X)

#Split the data into training and testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#Split the label into training and testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

#Create a Linear Regression object
linreg_object = linear_model.LinearRegression()

#Train the model using the training sets of data
linreg_object.fit(diabetes_X_train,diabetes_y_train)

#Make predictions using the training sets
diabetes_y_predicted = linreg_object.predict(diabetes_X_test)

#The coefficients to measure the mean square error and r2 determination
print('Coefficients : \n',linreg.coef_)
print('Mean Square Error : %.2f \n' % mean_squared_error(diabetes_y_test,diabetes_y_predicted))
print('Coefficient of determination : %.2f \n' % r2_score(diabetes_y_test,diabetes_y_predicted))

#Plot Outputs
fig = plt.figure(figsize=(8,8))

line1 = plt.scatter(diabetes_X_test,diabetes_y_test,color='black',label="Test features versus Test Labels")
line2 = plt.plot(diabetes_X_test,diabetes_y_predicted,color='blue',linewidth=3,label="Test features vs Predicted Labels")
plt.xlabel('Diabetes X Axis')
plt.ylabel('Diabetes Y Axis')
plt.xticks()
plt.yticks()
#plt.legend((line1,line2),('Test features versus Test Labels','Test features vs Predicted Labels'))
plt.legend()

plt.show()