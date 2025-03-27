import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import (
    DataTranformationArtifact,
    DataValidationArtifact
)
from networksecurity.entity.config_entity import DataTranformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_atifact:DataValidationArtifact,
                 data_tranformation_config:DataTranformationConfig):
        
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_atifact
            self.data_tranformation_config:DataTranformationConfig=data_tranformation_config

        except Exception as e:
            raise NetworkSecurityException(e,sys)





    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def get_data_transformer_object(cls)->Pipeline:
        """
        It initialises a KNNInputer object with the parameers specified in the tranining_pipeline.py file
        as returns a pipeline object with the KNNImputer object as the first step.

        args:
          clas:DataTransformation

        returns:
          A pipeline object
        
        """

        logging.info(
            "entered get_data_transformer_object function of datatransformation class"
        )

        try:
            imputer:KNNImputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialized KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            processor:Pipeline=Pipeline([("imputer",imputer)])
            return processor
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_transformation(Self)->DataTranformationArtifact:
        logging.info("entered initiate_data-_tansformation function of datatransformation class ")
        try:
            logging.info("starting data transformation")
            train_df=DataTransformation.read_data(Self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(Self.data_validation_artifact.valid_test_file_path)
            ## training dataframe
            target_feature_train_df=train_df[TARGET_COLUMN]
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=target_feature_train_df.replace(-1,0)
            ## test dataframe
            target_feature_test_df=test_df[TARGET_COLUMN]
            input_feature_test_df=test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df=target_feature_test_df.replace(-1,0)

            preprocessor=Self.get_data_transformer_object()
            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature=preprocessor_object.transform(input_feature_test_df)

            train_arr=np.c_[transformed_input_train_feature,np.array(target_feature_train_df)]
            test_arr=np.c_[transformed_input_test_feature,np.array(target_feature_test_df)]

            #save numpy array data
            save_numpy_array_data(Self.data_tranformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(Self.data_tranformation_config.transformed_test_file_path,array=test_arr)
            save_object(Self.data_tranformation_config.transformed_object_file_path,preprocessor_object)

            ## preparing artifacts

            data_transformation_artifact=DataTranformationArtifact(
                transformed_object_file_path=Self.data_tranformation_config.transformed_object_file_path,
                transformed_train_file_path=Self.data_tranformation_config.transformed_train_file_path,
                transformed_test_file_path=Self.data_tranformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)