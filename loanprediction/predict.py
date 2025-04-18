import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from loanprediction import __version__ as _version
from loanprediction.config.core import config
from loanprediction.processing.data_manager import load_pipeline
from loanprediction.processing.data_manager import pre_pipeline_preparation
from loanprediction.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
loanprediction_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config_.features)
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        predictions = loanprediction_pipe.predict(validated_data)
        result = "Default" if predictions[0] == 1 else "No Default"
        print("result predicted ", result)
        results = {"predictions": result, "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    data_in = {'Age': [56], 'Income': [85994], 'LoanAmount': [50587], 'CreditScore': [520], 'MonthsEmployed': [80],
               'NumCreditLines': [4], 'InterestRate': [15.23], 'LoanTerm': [16], 'DTIRatio': [17.5], 'Education': ['PhD'], 'EmploymentType': ['Full-time'],
               'MaritalStatus': ['Married'], 'HasMortgage': ['No'], 'HasDependents': ['No'], 'LoanPurpose': ['Education'], 'HasCoSigner': ['No']}

    make_prediction(input_data = data_in)