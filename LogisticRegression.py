# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('ModifiedHeartDisease.csv')

# Extract features (X) and labels (y) from the dataset
X = data.iloc[:, :-1].values  # All columns except the last one
y = data.iloc[:, -1].values   # Only the last column

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Fit to training data and transform it
X_test = scaler.transform(X_test)       # Transform the test data

# Add a column of ones for the bias term
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Define the sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the logistic regression function using gradient descent
def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    theta_log = np.zeros(X.shape[1]) # Initialize the weights
    cost_hist = []                    # Initialize the cost history
    for i in range(num_iterations):
        # Calculate predictions
        z = np.dot(X, theta_log)
        y_pred = sigmoid(z)
        # Compute the gradient
        gradient = np.dot(X.T, (y_pred - y)) / y.size
        # Update the weights
        theta_log -= learning_rate * gradient
        # Compute the cost using binary cross-entropy loss
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        # Record the cost
        cost_hist.append(cost)
    return theta_log, cost_hist # Return the trained weights and cost history

# Train the logistic regression model
theta, cost_history = logistic_regression(X_train, y_train)

# Plot the cost history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Print the trained weights
print('Trained weights:', theta)

# Predict the labels for the test set
y_predicted = sigmoid(np.dot(X_test, theta))
y_predicted_class = np.round(y_predicted) # Round the probabilities to get class labels
accuracy = np.mean(y_predicted_class == y_test) # Calculate the accuracy
print('Test set accuracy:', accuracy)
