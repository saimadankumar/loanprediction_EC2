from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[str]


class DataInputSchema(BaseModel):
    Age: Optional[int]
    Income: Optional[int]
    LoanAmount: Optional[int]
    CreditScore: Optional[int]
    MonthsEmployed: Optional[int]
    NumCreditLines: Optional[int]
    InterestRate: Optional[float]
    LoanTerm: Optional[int]
    DTIRatio: Optional[float]
    Education: Optional[str]
    EmploymentType: Optional[str]
    MaritalStatus: Optional[str]
    HasMortgage: Optional[str]
    HasDependents: Optional[str]
    LoanPurpose: Optional[str]
    HasCoSigner: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
