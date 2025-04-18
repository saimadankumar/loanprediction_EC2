import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from loanprediction import __version__ as model_version
from loanprediction.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {
            "Age": 56, # datetime.datetime.strptime("2012-11-05", "%Y-%m-%d"),  
            "Income": 85994, 
            "LoanAmount": 50587,
            "CreditScore": 520, 
            "MonthsEmployed": 80,
            "NumCreditLines": 4,
            "InterestRate": 15.23,
            "LoanTerm": 16,
            "DTIRatio": 17.5,
            "Education": "PhD",	
            "EmploymentType": "Full-time",
            "MaritalStatus": "Married",
            "HasMortgage": "No",
            "HasDependents": "No",
            "LoanPurpose": "Education",
            "HasCoSigner": "No"
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Bike rental count prediction with the bikeshare_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results
