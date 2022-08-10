import os
import sys

from censusIncome.exception import CensusException
from censusIncome.util.util import load_object

import pandas as pd


class CensusData:

    def __init__(self,
                 age: int,
                 work_class: str,
                 fnlwgt: int,
                 education: str,
                 education_num: int,
                 marital_status: str,
                 occupation: str,
                 relationship: str,
                 race: str,
                 sex: str,
                 capital_gain: int,
                 capital_loss: int,
                 country: str,
                 hours_per_week: int
                 ):
        try:
            self.age = age,
            self.work_class = work_class,
            self.fnlwgt = fnlwgt,
            self.education = education,
            self.education_num = education_num,
            self.marital_status = marital_status,
            self.occupation = occupation,
            self.relationship = relationship,
            self.race = race,
            self.sex = sex,
            self.capital_gain = capital_gain,
            self.capital_loss = capital_loss,
            self.hours_per_week = hours_per_week,
            self.country = country,

        except Exception as e:
            raise CensusException(e, sys) from e

    def get_census_input_data_frame(self):

        try:
            census_input_dict = self.get_census_data_as_dict()
            return pd.DataFrame(census_input_dict)
        except Exception as e:
            raise CensusException(e, sys) from e

    def get_census_data_as_dict(self):
        try:
            input_data = {
                'age': [self.age],
                'work_class': [self.work_class],
                'fnlwgt': [self.fnlwgt],
                'education': [self.education],
                'education_num': [self.education_num],
                'marital_status': [self.marital_status],
                'occupation': [self.occupation],
                'relationship': [self.relationship],
                'race': [self.race],
                'sex': [self.sex],
                'capital_gain': [self.capital_gain],
                'capital_loss': [self.capital_loss],
                'hours_per_week': [self.hours_per_week],
                'country': [self.country]
            }
            print(input_data)
            return input_data
        except Exception as e:
            raise CensusException(e, sys)


class CensusPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise CensusException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise CensusException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            median_census_value = model.predict(X)
            return median_census_value
        except Exception as e:
            raise CensusException(e, sys) from e