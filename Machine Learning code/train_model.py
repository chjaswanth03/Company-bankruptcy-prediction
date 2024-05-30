import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv(r'C:\Users\JASWANTH CH\OneDrive\Desktop\@SEM 6\Summer Intership\Projects 2024 (03-04-2024)\Company bankruptcy prediction\Machine Learning code\bankruptcy.csv')

# Assuming the target column is named 'Bankrupt?'
# and all other columns are features
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Save the trained model
joblib.dump(model, 'model.pkl')
print('Model saved as model.pkl')
