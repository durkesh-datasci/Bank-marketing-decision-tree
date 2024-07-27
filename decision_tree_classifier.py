# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#I have done this in google colab, please give your file the correctly
# Load the data with the correct delimiter
data = pd.read_csv('/content/bank-additional-full.csv', delimiter=';')

# Inspect the column names
print(data.columns)

# Check if 'y' exists, else find the correct target column name
target_column = 'y'
if target_column not in data.columns:
    print(f"Column '{target_column}' not found in the DataFrame. Please check the column names.")
else:
    # Preprocess the data
    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Define features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Visualize the decision tree
    plt.figure(figsize=(20,10))
    plot_tree(clf, feature_names=X.columns, class_names=label_encoders[target_column].classes_, filled=True)
    plt.show()