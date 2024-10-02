import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    train_data_path, test_data_path = DataIngestion().initiate_data_ingestion()


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self):
        logging.info("Entered the data transformation method or component")
        try:
            train_set = pd.read_csv(self.transformation_config.train_data_path)
            test_set = pd.read_csv(self.transformation_config.test_data_path)

            X_train = train_set.drop(columns=["math_score"], axis=1)
            y_train = train_set["math_score"]

            X_test = test_set.drop(columns=["math_score"], axis=1)
            y_test = test_set["math_score"]

            X_train = self.initiate_preprocessing(
                X_train.select_dtypes(exclude="object").columns,
                X_train.select_dtypes(include="object").columns,
            ).fit_transform(X_train)
            X_test = self.initiate_preprocessing(
                X_test.select_dtypes(exclude="object").columns,
                X_test.select_dtypes(include="object").columns,
            ).transform(X_test)

            return (X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_preprocessing(self, num_features, cat_features):
        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder()

        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, cat_features),
                ("StandardScaler", numeric_transformer, num_features),
            ]
        )

        return preprocessor


if __name__ == "__main__":
    obj = DataTransformation()
    X_train, y_train, X_test, y_test = obj.initiate_data_transformation()
    print(X_train.head())
