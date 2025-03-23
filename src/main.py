import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # For handling missing values more robustly

# 1. Load the data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Data Cleaning and Feature Engineering

# Impute missing Age values based on Pclass and Sex
train_df['Age'] = train_df['Age'].fillna(train_df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
test_df['Age'] = test_df['Age'].fillna(test_df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))

# If there are still any missing Age values (e.g., a Pclass/Sex combination with no ages), fill with the overall median
train_df.loc[:,'Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df.loc[:,'Age'] = test_df['Age'].fillna(test_df['Age'].median())

# Impute missing Embarked values with the mode
train_df.loc[:,'Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Impute missing Fare values with the median
test_df.loc[:,'Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Convert categorical features to numerical
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Create dummy variables for Embarked
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# Feature Engineering - FamilySize and IsAlone
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)
test_df['IsAlone'] = test_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)

# Feature Engineering - FarePerPerson
train_df['FarePerPerson'] = train_df['Fare'] / train_df['FamilySize']
test_df['FarePerPerson'] = test_df['Fare'] / test_df['FamilySize']

# Handle potential division by zero
train_df['FarePerPerson'] = train_df['FarePerPerson'].replace([float('inf'), float('nan')], 0)
test_df['FarePerPerson'] = test_df['FarePerPerson'].replace([float('inf'), float('nan')], 0)

# Ensure Pclass columns exist in the test set
for col in ['Pclass_2', 'Pclass_3']:
    if col not in train_df.columns:
        train_df[col] = 0
# Ensure Pclass columns exist in the test set
    if col not in test_df.columns:
        test_df[col] = 0

# Drop unnecessary columns (PassengerId is dropped later)
train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# 3. Select Features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'Embarked_Q', 'Embarked_S', 'Pclass_2', 'Pclass_3']

# 4. Prepare Data for Model Training
# Separate features (X) and target (y) in the training set
X = train_df[features]
y = train_df['Survived']

# Use the same features in the test set
X_test = test_df[features]

# Drop the PassengerId column after feature selection
train_passenger_ids = train_df['PassengerId']
test_passenger_ids = test_df['PassengerId']

train_df = train_df.drop(['PassengerId'], axis=1)
test_df = test_df.drop(['PassengerId'], axis=1)

# 5. Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 6. Model Training and Evaluation
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose the model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred_val = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {accuracy}")

# 7. Make Predictions on the Test Set
predictions = model.predict(X_test)

# 8. Create the Submission File
submission = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")