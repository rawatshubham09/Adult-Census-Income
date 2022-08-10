from censusIncome.entity.config_entity import DataIngestionConfig
import sys
import os
from censusIncome.exception import CensusException
from censusIncome.logger import logging
from censusIncome.entity.artifact_entity import DataIngestionArtifact
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>' * 20}Data Ingestion log started.{'<<' * 20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CensusException(e, sys)

    def download_census_data(self, ) -> None:
        try:
            # extraction remote url to download dataset
            download_url = self.data_ingestion_config.dataset_download_url

            # folder location to download file
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir

            os.makedirs(tgz_download_dir, exist_ok=True)

            census_file_name = os.path.basename(download_url)

            tgz_file_path = os.path.join(tgz_download_dir, census_file_name)

            logging.info(f"Downloading file from :[{download_url}] into :[{tgz_file_path}]")

            urllib.request.urlretrieve(download_url, tgz_file_path)

            logging.info(f"File :[{tgz_file_path}] has been downloaded successfully.")

        except Exception as e:
            raise CensusException(e, sys) from e

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            tgz_data_dir = self.data_ingestion_config.tgz_download_dir

            file_name = os.listdir(tgz_data_dir)[0]

            census_file_path = os.path.join(tgz_data_dir, file_name)

            logging.info(f"Reading csv file: [{census_file_path}]")

            census_data_frame = pd.read_csv(census_file_path)

            census_data_frame["salary"] = census_data_frame["salary"].apply(lambda x: 1 if x == "<=50K" else 0)

            logging.info(f"Splitting data into train and test")

            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index, test_index in split.split(census_data_frame, census_data_frame["salary"]):
                strat_train_set = census_data_frame.loc[train_index]
                strat_test_set = census_data_frame.loc[test_index]

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                           file_name)

            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                          file_name)

            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)

            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message=f"Data ingestion completed successfully."
                                                            )
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise CensusException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:

            tgz_file_path = self.download_census_data()
            print(tgz_file_path)

            # self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return self.split_data_as_train_test()

        except Exception as e:
            raise CensusException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 20}Data Ingestion log completed.{'<<' * 20} \n\n")
