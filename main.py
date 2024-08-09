from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from fairlearn.reductions import DemographicParity,EqualizedOdds
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
# Load a dataset (e.g., adult census income dataset)
from datanalyzer import DataAnalyzer

from test3 import get_model_accuracy
from test4 import get_model_accuracy2 as gma

accuracy = get_model_accuracy("DS01.csv", "IMDB_Rating")
accuracy2=gma("Employee.csv", "LeaveOrNot")

data_analyzer =DataAnalyzer(data_path="DS01.csv", target_column="IMDB_Rating")
data_analyzer.load_data()
data_analyzer.preprocessing()
data_analyzer.visualize_data()
data_analyzer.bias_mitigation()
accuracy = data_analyzer.evaluate_resampled_model()
print(f"Accuracy: {accuracy}")
data_analyzer.check_implicit_bias()

