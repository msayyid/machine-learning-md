import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score


line = "##################################################################"


# Load the bike dataset
DATA = pd.read_csv("week_2/DATA.csv")

print(DATA.head())
print(DATA.columns)
print(line)


# Separate input features and target
X = DATA.drop(["cnt"], axis=1)
Y = DATA["cnt"]


# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


print(line)


# Train a Lasso regression model
model = Lasso(alpha=0.1)

model.fit(X_train, Y_train)


# Predict values for the test set
predictions = model.predict(X_test)


# Evaluate the model
r2 = r2_score(Y_test, predictions)

print("R2 score:", r2)


print(line)
print(line)
print(line)


# Create sample data for polynomial regression
np.random.seed(2)

pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds


# Use 80% of the data for training and 20% for testing
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]


# Plot the training data
plt.scatter(trainX, trainY)
plt.title("Training Data")
plt.xlabel("Page Speed")
plt.ylabel("Purchase Amount")
plt.show()


# Plot the test data
plt.scatter(testX, testY)
plt.title("Test Data")
plt.xlabel("Page Speed")
plt.ylabel("Purchase Amount")
plt.show()


# Test polynomial models with different degrees
print("R2 scores for different polynomial degrees:")

best_degree = None
best_r2 = float("-inf")
best_model = None

for degree in range(1, 9):

    model = np.poly1d(
        np.polyfit(trainX, trainY, degree)
    )

    predictions = model(testX)

    r2 = r2_score(testY, predictions)

    print(f"Degree {degree}: R2 = {r2}")

    # Keep the model with the highest R2 score
    if r2 > best_r2:
        best_r2 = r2
        best_degree = degree
        best_model = model


print()
print("Best degree:", best_degree)
print("Best R2 score:", best_r2)


# Plot the best model with the training data
xp = np.linspace(0, 7, 100)

plt.scatter(trainX, trainY)
plt.plot(xp, best_model(xp))

plt.xlim(0, 7)
plt.ylim(0, 200)

plt.title(f"Best Polynomial Model - Degree {best_degree}")
plt.xlabel("Page Speed")
plt.ylabel("Purchase Amount")

plt.show()


# Plot the best model with the test data
plt.scatter(testX, testY)
plt.plot(xp, best_model(xp))

plt.xlim(0, 7)
plt.ylim(0, 200)

plt.title(f"Best Polynomial Model on Test Data - Degree {best_degree}")
plt.xlabel("Page Speed")
plt.ylabel("Purchase Amount")

plt.show()