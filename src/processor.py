import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


class DataProcessor:
    def __init__(self, n_components=3):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.imputer = SimpleImputer(strategy='mean')

    def clean_dataframe(self, df):
        df_cleaned = df.copy()
        for col in df_cleaned.select_dtypes(include=['object']).columns:
            if col not in ['Country', 'Region']:
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(',', '.')
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        return df_cleaned

    def process_data(self, df):
        df_cleaned = self.clean_dataframe(df)
        numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])

        imputed_data = self.imputer.fit_transform(numeric_df)
        scaled_data = self.scaler.fit_transform(imputed_data)
        pca_data = self.pca.fit_transform(scaled_data)

        pca_df = pd.DataFrame(
            data=pca_data,
            columns=[f'PC{i + 1}' for i in range(pca_data.shape[1])]
        )

        return pca_df, scaled_data

    def get_feature_names(self, df):
        # Returns the list of numeric columns used for training
        return df.select_dtypes(include=['float64', 'int64']).columns.tolist()
