#a)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


np.random.seed(2)
x = np.random.uniform(0, 10, 200)
y = 2 * x**2 - 5 * x + 3 + np.random.normal(0, 10, 200)


x = x.reshape(-1, 1)

# b) Split the data: Divide the dataset into a training set and a test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#c) Plot the data
plt.scatter(x, y, color='orange')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.show()

# b)Train the model: fit the model to the training data
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# c)Evaluated the trained model by the R2 and the MSE
y_train_pred = linear_model.predict(x_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

# d)Evaluate the model: Assess the performance of the trained model on the test data by the R2 and the MSE
y_test_pred = linear_model.predict(x_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Print the evaluation results for the linear model
print(f"Linear Model - Training R2: {train_r2}, Training MSE: {train_mse}")
print(f"Linear Model - Test R2: {test_r2}, Test MSE: {test_mse}")

# Refine the model using Polynomial Regression (Quadratic Model)

#Transform the features to include quadratic terms
poly_features = PolynomialFeatures(degree=2)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)

#Train a new model using the quadratic features
poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)

#  Evaluate the quadratic model on training data
y_train_poly_pred = poly_model.predict(x_train_poly)
train_r2_poly = r2_score(y_train, y_train_poly_pred)
train_mse_poly = mean_squared_error(y_train, y_train_poly_pred)

#  Evaluate the quadratic model on test data
y_test_poly_pred = poly_model.predict(x_test_poly)
test_r2_poly = r2_score(y_test, y_test_poly_pred)
test_mse_poly = mean_squared_error(y_test, y_test_poly_pred)

# Print the evaluation results for the quadratic model
print(f"Quadratic Model - Training R2: {train_r2_poly}, Training MSE: {train_mse_poly}")
print(f"Quadratic Model - Test R2: {test_r2_poly}, Test MSE: {test_mse_poly}")

# Plot the quadratic fit on top of the data
x_range = np.linspace(0, 10, 300).reshape(-1, 1)
x_range_poly = poly_features.transform(x_range)
y_range_pred = poly_model.predict(x_range_poly)

plt.scatter(x, y, color='orange', label='Data')
plt.plot(x_range, y_range_pred, color='blue', label='Quadratic Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Fit to Dataset')
plt.legend()
plt.show()