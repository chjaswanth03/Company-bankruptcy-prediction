# Company Bankruptcy Prediction

This project aims to predict the likelihood of a company going bankrupt based on its financial data using a machine learning model. The application provides a user-friendly web interface where users can input financial metrics and receive a prediction about the company's bankruptcy status.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This bankruptcy prediction project utilizes financial indicators to predict whether a company is likely to go bankrupt. The prediction model is built using machine learning techniques and is deployed via a Flask web application.

## Features

- Input financial data through a web form.
- Predict the likelihood of a company going bankrupt.
- Display prediction results on the web interface.

## Data

The dataset used for training the model includes various financial features such as Return on Assets (ROA), Operating Gross Margin, Cash Flow Rate, etc.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/company-bankruptcy-prediction.git
    cd company-bankruptcy-prediction
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure the model is trained and saved as `model.pkl` in the project directory. (Refer to the Model Training section for training the model.)

2. Start the Flask application:
    ```sh
    python app.py
    ```

3. Open your web browser and navigate to `http://127.0.0.1:5000` to access the Bankruptcy Prediction web application.

## Model Training

The model is trained using financial data. Here's an example of how to train the model:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess your data
data = pd.read_csv('financial_data.csv')
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'model.pkl')
