import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


line = "##################################################################"
line = "-------------------------------------------------------------------------------------------------------" \
"\n-------------------------------------------------------------------------------------------------------"

# Load the dataset from Lab 3
DATA = pd.read_csv("week_3/PastHires.csv")

print(DATA.head())
print()
print(DATA.shape)

print(line)


# Convert text values into numbers
DATA["Employed?"] = DATA["Employed?"].map({
    "N": 0,
    "Y": 1
})

DATA["Top-tier school"] = DATA["Top-tier school"].map({
    "N": 0,
    "Y": 1
})

DATA["Interned"] = DATA["Interned"].map({
    "N": 0,
    "Y": 1
})

DATA["Hired"] = DATA["Hired"].map({
    "N": 0,
    "Y": 1
})

DATA["Level of Education"] = DATA["Level of Education"].map({
    "BS": 0,
    "MS": 1,
    "PhD": 2
})


# X contains the input features
# Y contains the value we want to predict
X = DATA.drop(["Hired"], axis=1)
Y = DATA["Hired"]


# Split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Test data:", X_test.shape)

print(line)


# Create the three classifiers from Lab 3
decision_tree = DecisionTreeClassifier(random_state=42)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

naive_bayes = GaussianNB()


# Train each model using the training data
decision_tree.fit(X_train, Y_train)
random_forest.fit(X_train, Y_train)
naive_bayes.fit(X_train, Y_train)


# Use the trained models to make predictions
dt_predictions = decision_tree.predict(X_test)
rf_predictions = random_forest.predict(X_test)
nb_predictions = naive_bayes.predict(X_test)


print(line)


# Decision Tree evaluation

print("DECISION TREE")

print("Accuracy:")
print(accuracy_score(Y_test, dt_predictions))

print("Precision:")
print(precision_score(Y_test, dt_predictions, zero_division=0))

print("Recall:")
print(recall_score(Y_test, dt_predictions, zero_division=0))

print("F1 score:")
print(f1_score(Y_test, dt_predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, dt_predictions))

print(line)


# Random Forest evaluation

print("RANDOM FOREST")

print("Accuracy:")
print(accuracy_score(Y_test, rf_predictions))

print("Precision:")
print(precision_score(Y_test, rf_predictions, zero_division=0))

print("Recall:")
print(recall_score(Y_test, rf_predictions, zero_division=0))

print("F1 score:")
print(f1_score(Y_test, rf_predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, rf_predictions))

print(line)


# Naive Bayes evaluation

print("NAIVE BAYES")

print("Accuracy:")
print(accuracy_score(Y_test, nb_predictions))

print("Precision:")
print(precision_score(Y_test, nb_predictions, zero_division=0))

print("Recall:")
print(recall_score(Y_test, nb_predictions, zero_division=0))

print("F1 score:")
print(f1_score(Y_test, nb_predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(Y_test, nb_predictions))

print(line)


# Plot the confusion matrix for the Decision Tree

ConfusionMatrixDisplay.from_predictions(
    Y_test,
    dt_predictions
)

plt.title("Decision Tree Confusion Matrix")
plt.show()


# ROC curve and AUC

# Get the probability that each test sample belongs to class 1
dt_probabilities = decision_tree.predict_proba(X_test)[:, 1]


# Calculate the points needed for the ROC curve
fpr, tpr, thresholds = roc_curve(
    Y_test,
    dt_probabilities
)


# Calculate the area under the ROC curve
roc_auc = auc(fpr, tpr)

print("Decision Tree AUC:")
print(roc_auc)


# Plot the ROC curve
plt.plot(
    fpr,
    tpr,
    label="Decision Tree"
)

# This line represents random guessing
plt.plot(
    [0, 1],
    [0, 1]
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree ROC Curve")
plt.legend()

plt.show()

print(line)


# Feature importance

# Get how important each feature was to the Decision Tree
importances = decision_tree.feature_importances_


# Put the feature names and their importance into a table
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})


# Sort the table from most important to least important
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("Feature Importance:")
print(feature_importance)

print()


# Plot the feature importance
plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Decision Tree Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print(line)