
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from titanic_model.config.core import config
from titanic_model.processing.features import embarkImputer, Mapper, age_col_tfr


import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from titanic_model.config.core import config
from titanic_model.processing.data_manager import load_pipeline

"""
# test Age imputation
def test_age_imputation(sample_input_data):
    # Given
    pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    titanic_pipe = load_pipeline(file_name=pipeline_file_name)
    
    # Assuming 'sample_input_data' is a DataFrame with missing 'Age' values
    assert pd.isnull(sample_input_data.loc[some_index, 'Age'])

    # When
    processed_data = titanic_pipe.transform(sample_input_data)

    # Then
    assert not pd.isnull(processed_data.loc[some_index, 'Age'])
"""
"""
def test_weekday_variable_imputer(sample_input_data):
    # Given
    imputer = WeekdayImputer(variable = config.model_config.weekday_var, date_var = config.model_config.date_var)
    assert np.isnan(sample_input_data[0].loc[7046, 'weekday'])

    # When
    subject = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[7046, 'weekday'] == 'Wed'


def test_weathersit_variable_imputer(sample_input_data):
    # Given
    imputer = WeathersitImputer(variable = config.model_config.weathersit_var)
    assert np.isnan(sample_input_data[0].loc[7046, 'weathersit'])

    # When
    subject = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[7046, 'weathersit'] == 'Clear'


def test_season_variable_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.model_config.season_var, 
                    mappings = config.model_config.season_mappings)
    assert sample_input_data[0].loc[8688, 'season'] == 'summer'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[8688, 'season'] == 2


def test_windspeed_variable_outlierhandler(sample_input_data):
    # Given
    encoder = OutlierHandler(variable = config.model_config.windspeed_var)
    q1, q3 = np.percentile(sample_input_data[0]['windspeed'], q=[25, 75])
    iqr = q3 - q1
    assert sample_input_data[0].loc[5813, 'windspeed'] > q3 + (1.5 * iqr)

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[5813, 'windspeed'] <= q3 + (1.5 * iqr)


def test_weekday_variable_encoder(sample_input_data):
    # Given
    encoder = WeekdayOneHotEncoder(variable = config.model_config.weekday_var)
    assert sample_input_data[0].loc[8688, 'weekday'] == 'Sun'

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[8688, 'weekday_Sun'] == 1.0

*/

"""