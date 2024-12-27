import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    prepressor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is used to transform data
        '''

        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical Columns: {}".format(numerical_columns))
            logging.info("Categorical Columns: {}".format(categorical_columns))

            prepocessor = ColumnTransformer(
                transformers = [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return prepocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        '''
        This function is used to initiate data transformation
        '''

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train Data Shape: {}".format(train_df.shape))
            logging.info("Test Data Shape: {}".format(test_df.shape))

            logging.info("Obtaining Data Preprocessing Object")
        
            prepocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Data Preprocessing on Train and Test Data")

            input_feature_train_arr = prepocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = prepocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]   

            logging.info("Saving Data Preprocessing Object")

            save_object(
                file_path = self.data_transformation_config.prepressor_obj_file_path,
                obj = prepocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.prepressor_obj_file_path
            )        

        except Exception as e:
            raise CustomException(e, sys)