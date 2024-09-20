import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


titanic_df = pd.read_csv("train.csv")
titanic_df.head(10)

# Drop the specified columns: 'PassengerID', 'Name', 'Ticket', and 'Cabin'
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

titanic_df.head(10)

# Select features: 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
X = titanic_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Convert categorical variables ('Sex' and 'Embarked') to numerical values using one-hot encoding
X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Create the target variable 'Y'
Y = titanic_df['Survived']

# Show the first 10 rows of X and Y to verify
print(X.head(10))
print(Y.head(10))

# List of numerical columns to be normalized
numerical_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply the scaler only to the numerical columns
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the XGBoost Classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model on the training data
xgb_model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = xgb_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy of the XGBoost model: {accuracy * 100:.2f}%")

# Load the test dataset (this file does not contain 'Survived')
test_df = pd.read_csv("test.csv")

# Store the PassengerId column separately (needed for submission)
passenger_ids = test_df['PassengerId']

# Drop irrelevant columns: 'PassengerId', 'Name', 'Ticket', 'Cabin'
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values in 'Age' or 'Fare' (common in Titanic dataset)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)

# Select the same features: 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
X_test_final = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Convert categorical variables ('Sex' and 'Embarked') to numerical values using one-hot encoding
X_test_final = pd.get_dummies(X_test_final, columns=['Sex', 'Embarked'], drop_first=True)

# Apply the same normalization to the numerical columns (using the same scaler from training)
X_test_final[numerical_columns] = scaler.transform(X_test_final[numerical_columns])

# Make predictions on the test dataset using the trained model
test_predictions = xgb_model.predict(X_test_final)

# Create a DataFrame for the submission with 'PassengerId' and 'Survived' predictions
submission_df = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': test_predictions
})

# Save the DataFrame to a CSV file in the required format
submission_df.to_csv('submission.csv', index=False)

print("submission.csv file created successfully!")