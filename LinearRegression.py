# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Define the Mean Squared Error (MSE) function
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


# Define the Gradient Descent function for linear regression
def gradient_descent(X, y, alpha, epochs):
    theta_in = np.zeros(X.shape[1])  # Initialize theta (coefficients) with zeros
    hist_cost = []  # Initialize cost history
    for i in range(epochs):
        y_pred = np.dot(X, theta_in)  # Compute the predicted values
        error = y - y_pred  # Compute the error
        gradient = -np.dot(X.T, error) / len(y)  # Compute the gradient
        theta_in -= alpha * gradient  # Update the coefficients
        cost = mse(y, y_pred)  # Compute the cost using MSE
        hist_cost.append(cost)  # Add cost to history
    return theta_in, hist_cost  # Return final coefficients and cost history


# Main function
if __name__ == '__main__':
    data = pd.read_csv('CustomerService.csv')  # Load the data

    X = data[['Complaint_Type', 'Complaint_Details']]  # Set features
    y = data['Time_to_Resolve']  # Set target

    X = pd.get_dummies(X)  # Convert categorical variables to dummy variables
    X['intercept'] = 1  # Add an intercept column to the features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into train and test sets

    alpha = 0.01  # Set learning rate
    epochs = 10000  # Set number of iterations

    theta, cost_history = gradient_descent(X_train, y_train, alpha, epochs)  # Perform gradient descent on the training data
    y_predicted = np.dot(X_test, theta)  # Predict values on the test set
    mse_test = mse(y_test, y_predicted)  # Compute MSE on the test set
    rsq_test = 1 - (np.sum((y_test - y_predicted) * 2) / np.sum((y_test - np.mean(y_test)) * 2))  # Compute R-squared value on the test set

    print(f"Mean squared error on the test set: {mse_test}")
    print(f"R-squared value on the test set: {rsq_test}")
    print(f"Accuracy on the test set: {rsq_test}")
    plt.plot(cost_history)  # Plot the cost history
    plt.xlabel('Epochs')  # Label x-axis as 'Epochs'
    plt.ylabel('Cost')  # Label y-axis as 'Cost'
    plt.show()  # Display the plot
