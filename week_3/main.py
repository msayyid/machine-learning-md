import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


line = "##################################################################"


# Load the dataset
DATA = pd.read_csv("week_3/PastHires.csv")

print(DATA.head())
print()
print(DATA.columns)
print()
print(DATA.shape)

print(line)


# Convert categorical values into numbers
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


print(DATA.head())

print(line)


# Separate the input features from the target
X = DATA.drop(["Hired"], axis=1)
Y = DATA["Hired"]


# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


print("Training data:", X_train.shape)
print("Test data:", X_test.shape)

print(line)


# ---------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(X_train, Y_train)

dt_predictions = decision_tree.predict(X_test)

dt_accuracy = accuracy_score(
    Y_test,
    dt_predictions
)

print("Decision Tree")
print("Accuracy:", dt_accuracy)

print()


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, Y_train)

rf_predictions = random_forest.predict(X_test)

rf_accuracy = accuracy_score(
    Y_test,
    rf_predictions
)

print("Random Forest")
print("Accuracy:", rf_accuracy)

print()


# ---------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------

naive_bayes = GaussianNB()

naive_bayes.fit(X_train, Y_train)

nb_predictions = naive_bayes.predict(X_test)

nb_accuracy = accuracy_score(
    Y_test,
    nb_predictions
)

print("Naive Bayes")
print("Accuracy:", nb_accuracy)

print(line)


# Show a more detailed evaluation for the Decision Tree
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(Y_test, dt_predictions))

print()

print("Decision Tree Classification Report:")
# print(classification_report(Y_test, dt_predictions))
print(classification_report(Y_test, dt_predictions, zero_division=0))

print(line)


# ---------------------------------------------------------
# Create our own test sample
# ---------------------------------------------------------

# Example applicant:
# 5 years experience
# currently employed
# 2 previous employers
# master's degree
# went to a top-tier school
# completed an internship

sample = pd.DataFrame(
    [[5, 1, 2, 1, 1, 1]],
    columns=[
        "Years Experience",
        "Employed?",
        "Previous employers",
        "Level of Education",
        "Top-tier school",
        "Interned"
    ]
)


print("Test sample:")
print(sample)

print()


# Predict whether the applicant would be hired
dt_result = decision_tree.predict(sample)
rf_result = random_forest.predict(sample)
nb_result = naive_bayes.predict(sample)


print("Decision Tree prediction:", dt_result[0])
print("Random Forest prediction:", rf_result[0])
print("Naive Bayes prediction:", nb_result[0])

print()


# Check whether all three models agree
if dt_result[0] == rf_result[0] == nb_result[0]:
    print("All three classifiers gave the same result.")
else:
    print("The classifiers gave different results.")

print(line)