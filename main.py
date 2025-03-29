from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTranformationConfig,ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
import sys
if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")
        print(dataingestionartifact)
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)
        logging.info("Initiate the data Transformation")
        data_transformation_config=DataTranformationConfig(trainingpipelineconfig)
        data_transormation=DataTransformation(data_validation_artifact,data_transformation_config)
        data_tranformation_artifact=data_transormation.initiate_data_transformation()
        print (data_tranformation_artifact)
        logging.info("Data Transformation Completed")
        logging.info("Initiate the model training")
        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_tranformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()
        logging.info("model training Completed")
    except Exception as e:
        raise NetworkSecurityException(e,sys)