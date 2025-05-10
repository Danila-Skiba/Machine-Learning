import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

from components.utils import load_default_data

from sklearn.model_selection import train_test_split

def handle_missing_values(df, numerical_strategy='mean'):
    df_processed = df.copy()
    numerical_cols = df_processed.select_dtypes(include=['number']).columns
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
  
    if numerical_cols.all():
        num_imputer = SimpleImputer(strategy=numerical_strategy)
        df_processed[numerical_cols] = num_imputer.fit_transform(df_processed[numerical_cols])
    
    if categorical_cols.all():
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])
    
    return df_processed

def encoding_data(data, encod_method):
    df_processed = data.copy()
    valid_categorical = [col for col in df_processed.columns if df_processed[col].dtype in ["object", "category"]]

    if not valid_categorical:
        print("Предупреждение: Ни один из указанных категориальных столбцов не найден в DataFrame.")
        return None

    if encod_method == "one-hot":
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        try:
            df_encoded = pd.DataFrame(encoder.fit_transform(df_processed[valid_categorical]),
            columns=encoder.get_feature_names_out(valid_categorical),
            index=df_processed.index)    
            df_non_category = df_processed.drop(valid_categorical, axis=1)
            df_processed = pd.concat([df_non_category, df_encoded], axis=1)
        except Exception as e:
                print(f"Ошибка при One-Hot кодировании: {e}")
    elif encod_method == "label":
        for col in valid_categorical:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
    return df_processed

def get_test_samples():
    df = load_default_data("RGR/Streamlit_project/src/result_mumbai.csv")
    if 'Unnamed: 0' in df:
        df = df.drop(columns=['Unnamed: 0'])
    Y = df['price']
    X = df.drop(['price'], axis =1)
    _, X_test, _, Y_test = train_test_split(X,Y,test_size=0.2)
    return X_test, Y_test