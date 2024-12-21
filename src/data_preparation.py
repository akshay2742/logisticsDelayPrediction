import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self):
        self.le_dict = {}
        
    def load_and_clean_data(self, filepath):
    # Load data from Excel file
        df = pd.read_excel(filepath)  # Changed from read_csv to read_excel
        
        # Handle missing values
        df['Weather_Conditions'].fillna(df['Weather_Conditions'].mode()[0], inplace=True)
        df['Traffic_Conditions'].fillna(df['Traffic_Conditions'].mode()[0], inplace=True)
        
        # Encode categorical variables
        categorical_cols = ['Origin', 'Destination', 'Vehicle_Type', 
                        'Weather_Conditions', 'Traffic_Conditions']
        
        for col in categorical_cols:
            self.le_dict[col] = LabelEncoder()
            df[col] = self.le_dict[col].fit_transform(df[col])
            
        return df

    def prepare_features(self, df):
        # Select features for model
        feature_cols = ['Origin', 'Destination', 'Vehicle_Type', 'Distance',
                       'Weather_Conditions', 'Traffic_Conditions']
        
        X = df[feature_cols]
        y = df['Delayed'].map({'Yes': 1, 'No': 0})
        
        return train_test_split(X, y, test_size=0.2, random_state=42)