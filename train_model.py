import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import joblib

# Load the dataset
df = pd.read_csv("mine_safety_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())