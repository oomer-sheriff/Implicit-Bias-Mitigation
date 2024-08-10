# DataAnalyzer Project Documentation

## Overview
The `DataAnalyzer` class is designed to streamline the process of loading, preprocessing, visualizing, and analyzing data, particularly in the context of classification problems. The class also includes methods for bias mitigation and implicit bias detection, making it a comprehensive tool for machine learning projects.

## Features
- **Data Loading**: Load CSV data into a pandas DataFrame.
- **Preprocessing**: Handle missing values, encode categorical variables, and prepare data for modeling.
- **Data Visualization**: Generate histograms to visualize the distribution of features against the target variable.
- **Bias Mitigation**: Resample data to address class imbalance and train a RandomForestClassifier.
- **Model Evaluation**: Evaluate the model's performance on resampled data using accuracy score and classification report.
- **Implicit Bias Detection**: Analyze and visualize potential biases in the features against the target variable.

## Prerequisites
Before using the `DataAnalyzer` class, ensure you have the following Python libraries installed:
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:
```bash
pip install pandas matplotlib seaborn scikit-learn
```

## How to Use

### 1. Import Required Libraries and Initialize the `DataAnalyzer` Class
Start by importing the necessary libraries and initializing the `DataAnalyzer` class with your dataset path and target column name.
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from DataAnalyzer import DataAnalyzer

# Example Initialization
data_analyzer = DataAnalyzer(data_path="Employee.csv", target_column="LeaveOrNot")
```

### 2. Load the Data
Use the `load_data()` method to load your dataset into a pandas DataFrame.
```python
data_analyzer.load_data()
```
This will print the first few rows of the dataset to the console.

### 3. Preprocess the Data
The `preprocessing()` method handles missing values, encodes categorical variables, and separates features from the target variable.
```python
data_analyzer.preprocessing()
```
This will print the dataset's head before and after preprocessing to the console.

### 4. Visualize the Data
The `visualize_data()` method generates histograms to visualize the distribution of each feature against the target variable.
```python
data_analyzer.visualize_data()
```
This will display the histograms for each feature.

### 5. Perform Bias Mitigation
The `bias_mitigation()` method resamples the dataset to mitigate class imbalance and trains a RandomForestClassifier model.
```python
data_analyzer.bias_mitigation()
```
This will train the model and print a success message if training is successful.

### 6. Evaluate the Resampled Model
Use the `evaluate_resampled_model()` method to evaluate the trained model on the test data.
```python
accuracy = data_analyzer.evaluate_resampled_model()
print(f"Accuracy: {accuracy}")
```
This will print the accuracy and classification report of the resampled model.

### 7. Check for Implicit Bias
The `check_implicit_bias()` method analyzes and visualizes potential biases in the dataset.
```python
data_analyzer.check_implicit_bias()
```
This will display histograms and print statistical parity differences for each feature against the target variable.

## Example Usage
Here's a complete example of how to use the `DataAnalyzer` class:
```python
# Initialize the DataAnalyzer with your dataset
data_analyzer = DataAnalyzer(data_path="Employee.csv", target_column="LeaveOrNot")

# Load the dataset
data_analyzer.load_data()

# Preprocess the data
data_analyzer.preprocessing()

# Visualize the data
data_analyzer.visualize_data()

# Perform bias mitigation and train the model
data_analyzer.bias_mitigation()

# Evaluate the model
accuracy = data_analyzer.evaluate_resampled_model()
print(f"Accuracy: {accuracy}")

# Check for implicit bias
data_analyzer.check_implicit_bias()
```

## Conclusion
The `DataAnalyzer` class is a versatile tool for data analysis, preprocessing, visualization, and bias mitigation in machine learning projects. By following the steps outlined in this documentation, you can efficiently analyze your datasets and build more robust models with a focus on fairness and performance.