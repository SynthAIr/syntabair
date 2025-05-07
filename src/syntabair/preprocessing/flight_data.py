
import pandas as pd
import numpy as np
import calendar
# from sdv.metadata import Metadata

def preprocess_flight_data(original_data):
    """
    Preprocess the original flight data for training generative models.
    
    Args:
        original_data (pd.DataFrame): Original flight data
        
    Returns:
        pd.DataFrame: Preprocessed data with temporal features extracted
    
    For an autoregressive model like GPT-2, the order of columns in the preprocessing step is crucial because the model predicts the next token based on the previous tokens. To maximize the model's ability to learn meaningful patterns, you should order the columns in a way that reflects the natural dependencies or logical flow of the data. Here's a suggested order for your flight data:

    1. **Categorical identifiers first**: Start with columns that uniquely identify or categorize the data, such as carrier codes and airport codes.
    - `IATA_CARRIER_CODE`
    - `DEPARTURE_IATA_AIRPORT_CODE`
    - `ARRIVAL_IATA_AIRPORT_CODE`
    - `AIRCRAFT_TYPE_IATA`

    2. **Temporal features**: Include features related to time in a logical sequence, such as scheduled departure time, followed by derived temporal features.
    - `SCHEDULED_MONTH`
    - `SCHEDULED_DAY`
    - `SCHEDULED_HOUR`
    - `SCHEDULED_MINUTE`

    3. **Duration features**: Add features related to flight durations, as they depend on the temporal features.
    - `SCHEDULED_DURATION_MIN`
    - `ACTUAL_DURATION_MIN`

    4. **Delay features**: Include delay-related features last, as they are often the result of earlier features.
    - `DEPARTURE_DELAY_MIN`

    This order ensures that the model processes the data in a way that respects the natural flow of information, from identifiers to temporal features, and finally to derived outcomes like delays.
    """

    # Convert date columns to datetime
    date_columns = [
        'DEPARTURE_ACTUAL_OUTGATE_UTC',
        'ARRIVAL_ACTUAL_INGATE_UTC',
        'SCHEDULED_DEPARTURE_UTC',
        'SCHEDULED_ARRIVAL_UTC'
    ]
    for col in date_columns:
        original_data[col] = pd.to_datetime(original_data[col])
    
    # Select base features
    preprocessed_df = original_data[[
        'IATA_CARRIER_CODE', 
        'DEPARTURE_IATA_AIRPORT_CODE', 
        'ARRIVAL_IATA_AIRPORT_CODE', 
        'AIRCRAFT_TYPE_IATA',
        'DEPARTURE_DELAY_MIN',
        'TURNAROUND_MIN'
    ]].copy()

    # Extract temporal features from scheduled departure time
    preprocessed_df['SCHEDULED_MONTH'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.month    # 1–12
    preprocessed_df['SCHEDULED_DAY'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.day        # 1–31
    preprocessed_df['SCHEDULED_HOUR'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.hour      # 0–23  
    preprocessed_df['SCHEDULED_MINUTE'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.minute  # 0–59

    # Calculate durations
    preprocessed_df['SCHEDULED_DURATION_MIN'] = (
        original_data['SCHEDULED_ARRIVAL_UTC'] - 
        original_data['SCHEDULED_DEPARTURE_UTC']
    ).dt.total_seconds() / 60
    
    preprocessed_df['ACTUAL_DURATION_MIN'] = (
        original_data['ARRIVAL_ACTUAL_INGATE_UTC'] - 
        original_data['DEPARTURE_ACTUAL_OUTGATE_UTC']
    ).dt.total_seconds() / 60

    # Reorder columns before returning
    column_order = [
        'IATA_CARRIER_CODE',
        'DEPARTURE_IATA_AIRPORT_CODE',
        'ARRIVAL_IATA_AIRPORT_CODE',
        'AIRCRAFT_TYPE_IATA',
        'SCHEDULED_MONTH',
        'SCHEDULED_DAY',
        'SCHEDULED_HOUR',
        'SCHEDULED_MINUTE',
        'SCHEDULED_DURATION_MIN',
        'ACTUAL_DURATION_MIN',
        'DEPARTURE_DELAY_MIN',
        'TURNAROUND_MIN'
    ]
    preprocessed_df = preprocessed_df[column_order]
    
    
    return preprocessed_df

def preprocess_flight_data_for_prediction(original_data):
    """
    Preprocess flight data specifically for prediction tasks.
    
    Unlike the generative model preprocessing, this function:
    1. Calculates and includes arrival delay
    2. Organizes features for predictive modeling rather than autoregressive generation
    3. Creates additional derived features useful for prediction tasks
    
    Args:
        original_data (pd.DataFrame): Original flight data
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction tasks
    """
    # Convert date columns to datetime
    date_columns = [
        'DEPARTURE_ACTUAL_OUTGATE_UTC',
        'ARRIVAL_ACTUAL_INGATE_UTC',
        'SCHEDULED_DEPARTURE_UTC',
        'SCHEDULED_ARRIVAL_UTC'
    ]
    for col in date_columns:
        original_data[col] = pd.to_datetime(original_data[col])
    
    # Create a new dataframe for prediction features
    preprocessed_df = original_data[[
        'IATA_CARRIER_CODE', 
        'DEPARTURE_IATA_AIRPORT_CODE', 
        'ARRIVAL_IATA_AIRPORT_CODE', 
        'AIRCRAFT_TYPE_IATA',
        'DEPARTURE_DELAY_MIN',
        'TURNAROUND_MIN'
    ]].copy()

    # Extract temporal features from scheduled departure time
    preprocessed_df['SCHEDULED_MONTH'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.month    # 1–12
    preprocessed_df['SCHEDULED_DAY'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.day        # 1–31
    preprocessed_df['SCHEDULED_HOUR'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.hour      # 0–23  
    preprocessed_df['SCHEDULED_MINUTE'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.minute  # 0–59

    # Calculate durations
    preprocessed_df['SCHEDULED_DURATION_MIN'] = (
        original_data['SCHEDULED_ARRIVAL_UTC'] - 
        original_data['SCHEDULED_DEPARTURE_UTC']
    ).dt.total_seconds() / 60
    
    preprocessed_df['ACTUAL_DURATION_MIN'] = (
        original_data['ARRIVAL_ACTUAL_INGATE_UTC'] - 
        original_data['DEPARTURE_ACTUAL_OUTGATE_UTC']
    ).dt.total_seconds() / 60
    
    # Calculate arrival delay
    preprocessed_df['ARRIVAL_DELAY_MIN'] = (
        original_data['ARRIVAL_ACTUAL_INGATE_UTC'] - 
        original_data['SCHEDULED_ARRIVAL_UTC']
    ).dt.total_seconds() / 60
    
    # Add duration difference (useful for tactical prediction)
    preprocessed_df['DURATION_DIFF_MIN'] = preprocessed_df['ACTUAL_DURATION_MIN'] - preprocessed_df['SCHEDULED_DURATION_MIN']
    
    # Include day of week (potentially useful for prediction)
    preprocessed_df['DAY_OF_WEEK'] = original_data['SCHEDULED_DEPARTURE_UTC'].dt.dayofweek  # 0-6, Monday=0
    
    # Feature organization for prediction models
    column_order = [
        # Categorical features
        'IATA_CARRIER_CODE',
        'DEPARTURE_IATA_AIRPORT_CODE',
        'ARRIVAL_IATA_AIRPORT_CODE',
        'AIRCRAFT_TYPE_IATA',
        
        # Time features
        'SCHEDULED_MONTH',
        'SCHEDULED_DAY',
        'SCHEDULED_HOUR',
        'SCHEDULED_MINUTE',
        'DAY_OF_WEEK',
        
        # Duration features
        'SCHEDULED_DURATION_MIN',
        'ACTUAL_DURATION_MIN',
        'DURATION_DIFF_MIN',
        
        # Target variables for different prediction tasks
        'DEPARTURE_DELAY_MIN',
        'ARRIVAL_DELAY_MIN',
        'TURNAROUND_MIN'
    ]
    

    preprocessed_df = preprocessed_df[column_order]
    
    return preprocessed_df

def reconstruct_original_format(synthetic_df, default_year=2019):
    """
    Convert synthetic data back to the original format by reconstructing datetime columns
    and calculating arrival delay. It also corrects invalid datetime components by clipping
    them to valid ranges.
    
    Args:
        synthetic_df (pd.DataFrame): Synthetic data generated by a model
        default_year (int): Default year to use for reconstructed dates
        
    Returns:
        pd.DataFrame: Data in original flight record format
    """
    # Correct SCHEDULED_MONTH: valid range 1 to 12
    synthetic_df['SCHEDULED_MONTH'] = synthetic_df['SCHEDULED_MONTH'].clip(lower=1, upper=12)
    
    # Correct SCHEDULED_HOUR: valid range 0 to 23
    synthetic_df['SCHEDULED_HOUR'] = synthetic_df['SCHEDULED_HOUR'].clip(lower=0, upper=23)
    
    # Correct SCHEDULED_MINUTE: valid range 0 to 59
    synthetic_df['SCHEDULED_MINUTE'] = synthetic_df['SCHEDULED_MINUTE'].clip(lower=0, upper=59)
    
    # Correct SCHEDULED_DAY: depends on month, so apply row-wise correction
    def correct_day(row):
        # Get the maximum valid day for the month in the given year
        max_day = calendar.monthrange(default_year, int(row['SCHEDULED_MONTH']))[1]
        # Clip the day to be between 1 and max_day
        return np.clip(row['SCHEDULED_DAY'], 1, max_day)
    
    synthetic_df['SCHEDULED_DAY'] = synthetic_df.apply(correct_day, axis=1)
    
    # Reconstruct scheduled departure datetime
    synthetic_df['SCHEDULED_DEPARTURE_UTC'] = pd.to_datetime({
        'year': default_year,
        'month': synthetic_df['SCHEDULED_MONTH'].round().astype('int16'),
        'day': synthetic_df['SCHEDULED_DAY'].round().astype('int16'),
        'hour': synthetic_df['SCHEDULED_HOUR'].round().astype('int16'),
        'minute': synthetic_df['SCHEDULED_MINUTE'].round().astype('int16')
    })
    # synthetic_df['SCHEDULED_DEPARTURE_UTC'] = pd.to_datetime({
    #     'year': default_year,
    #     'month': synthetic_df['SCHEDULED_MONTH'],
    #     'day': synthetic_df['SCHEDULED_DAY'],
    #     'hour': synthetic_df['SCHEDULED_HOUR'],
    #     'minute': synthetic_df['SCHEDULED_MINUTE']
    # })
    
    # Compute scheduled arrival time
    synthetic_df['SCHEDULED_ARRIVAL_UTC'] = (
        synthetic_df['SCHEDULED_DEPARTURE_UTC'] +
        pd.to_timedelta(synthetic_df['SCHEDULED_DURATION_MIN'], unit='m')
    )
    
    # Compute actual departure time using departure delay
    synthetic_df['DEPARTURE_ACTUAL_OUTGATE_UTC'] = (
        synthetic_df['SCHEDULED_DEPARTURE_UTC'] +
        pd.to_timedelta(synthetic_df['DEPARTURE_DELAY_MIN'], unit='m')
    )
    
    # Compute actual arrival time
    synthetic_df['ARRIVAL_ACTUAL_INGATE_UTC'] = (
        synthetic_df['DEPARTURE_ACTUAL_OUTGATE_UTC'] +
        pd.to_timedelta(synthetic_df['ACTUAL_DURATION_MIN'], unit='m')
    )
    
    # Compute arrival delay in minutes
    synthetic_df['ARRIVAL_DELAY_MIN'] = (
        synthetic_df['ARRIVAL_ACTUAL_INGATE_UTC'] - 
        synthetic_df['SCHEDULED_ARRIVAL_UTC']
    ).dt.total_seconds() / 60

    # floor to seconds
    dt_cols = [
        'SCHEDULED_DEPARTURE_UTC',
        'SCHEDULED_ARRIVAL_UTC',
        'DEPARTURE_ACTUAL_OUTGATE_UTC',
        'ARRIVAL_ACTUAL_INGATE_UTC'
    ]
    for col in dt_cols:
        synthetic_df[col] = synthetic_df[col].dt.floor('s')
    
    # Convert timedelta columns to minutes only
    round_cols = ['DEPARTURE_DELAY_MIN', 'ARRIVAL_DELAY_MIN', 'TURNAROUND_MIN']
    synthetic_df[round_cols] = synthetic_df[round_cols].round(1).astype(int).astype(float)

    # Select and reorder columns to match original format
    reconstructed_df = synthetic_df[[
        'IATA_CARRIER_CODE',
        'DEPARTURE_IATA_AIRPORT_CODE',
        'DEPARTURE_ACTUAL_OUTGATE_UTC',
        'ARRIVAL_IATA_AIRPORT_CODE',
        'ARRIVAL_ACTUAL_INGATE_UTC',
        'AIRCRAFT_TYPE_IATA',
        'SCHEDULED_DEPARTURE_UTC',
        'SCHEDULED_ARRIVAL_UTC',
        'DEPARTURE_DELAY_MIN',
        'ARRIVAL_DELAY_MIN',
        'TURNAROUND_MIN'
    ]]
    
    return reconstructed_df

