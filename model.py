import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()

# Train-test split with a different random state
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=1)

# Initialize and train model
model = RandomForestClassifier(n_estimators=90, random_state=1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with a different random seed: {accuracy}")

# Perform cross-validation
cross_val_scores = cross_val_score(model, data.data, data.target, cv=5)

# Print average cross-validation accuracy
print(f"Average cross-validation accuracy: {cross_val_scores.mean()}")
