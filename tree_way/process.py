import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import shuffle
import joblib


#class to take a csv and encode categories / normalize numbers / split into test and train data
class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()

    #load csv, assumes first three columns numerical and next three categorical, with the output variable as the final column (price)
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.target_column = self.df.columns[-1]
        self.numerical_features = self.df.columns[:3].tolist()  # first 3 columns
        self.categorical_features = self.df.columns[3:-1].tolist()  # middle 3 columns 

    #split the data so normalization isn't affected by test data
    def split_data(self, test_size=0.2, random_state=42):
        self.df = shuffle(self.df, random_state=42).reset_index(drop=True)
        X = self.df.iloc[:, :-1].values  # Everything except last column
        y = self.df.iloc[:, -1].values.reshape(-1, 1)  # Last column (price)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    #encoding categories + normalizing
    def encode_features(self):
        # encoding categories
        self.encoder.fit(self.X_train[:, 3:])
        X_train_encoded = self.encoder.transform(self.X_train[:, 3:])
        X_test_encoded = self.encoder.transform(self.X_test[:, 3:])

        # normalizing numbers
        self.scaler.fit(self.X_train[:, :3])
        X_train_scaled = self.scaler.transform(self.X_train[:, :3])
        X_test_scaled = self.scaler.transform(self.X_test[:, :3])

        #concatenate
        self.X_train_processed = np.hstack([X_train_scaled, X_train_encoded])
        self.X_test_processed = np.hstack([X_test_scaled, X_test_encoded])

    def get_processed_data(self):
        """Returns processed train/test sets ready for PyTorch"""
        return self.X_train_processed, self.X_test_processed, self.y_train, self.y_test


processor = DataProcessor("../data.csv")
processor.load_data()
processor.split_data()
processor.encode_features()

# get and save processed data
X_train, X_test, y_train, y_test = processor.get_processed_data()

joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")
joblib.dump(processor.encoder, "encoder.pkl")  
