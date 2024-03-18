from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from titanic_model.processing.validation import DataInputSchema

import numpy as np


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                "PassengerId": [80], # datetime.datetime.strptime("2012-11-05", "%Y-%m-%d"),  
                "Pclass": [2], 
                "Name": ["Caldwell, Master. Alden Gates"],
                "Sex": ['male'], 
                "Age": [83],
                "SibSp": [0],
                "Parch": [2],
                "Ticket": ['248738'],
                "Cabin": [np.nan],
                "Embarked": ['S'],	
                "Fare": [30],
                    }
                ]
            }
        }
#data_in={'PassengerId':[80],'Pclass':[2],'Name':["Caldwell, Master. Alden Gates"],'Sex':['male'],'Age':[0.83],
#                'SibSp':[0],'Parch':[2],'Ticket':['248738'],'Cabin':[np.nan,],'Embarked':['S'],'Fare':[29]}