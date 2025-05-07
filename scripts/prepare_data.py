import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def read_and_filter_data(data_path, filename, flight_state="InGate"):
    """
    Reads the CSV file and filters rows based on the given flight state.
    """
    df = pd.read_csv(data_path + filename, low_memory=False)
    df = df[df['FLIGHT_STATE'] == flight_state]
    return df

def select_and_clean_columns(df):
    """
    Selects the relevant columns and drops rows with missing values.
    """
    selected_columns = [
        'IATA_CARRIER_CODE',
        'DEPARTURE_IATA_AIRPORT_CODE',
        'AIRCRAFT_TYPE_IATA',
        'ARRIVAL_IATA_AIRPORT_CODE',
        'AIRCRAFT_REGISTRATION_NUMBER',
        'SCHEDULED_DEPARTURE_TIME_LOCAL',
        'DEPARTURE_ACTUAL_OUTGATE_TIMELINESS',
        'DEPARTURE_ACTUAL_OUTGATE_VARIATION',
        'DEPARTURE_ACTUAL_OUTGATE_UTC',
        'DEPARTURE_ACTUAL_OUTGATE_LOCAL',
        'SCHEDULED_ARRIVAL_TIME_LOCAL',
        'ARRIVAL_ACTUAL_INGATE_TIMELINESS',
        'ARRIVAL_ACTUAL_INGATE_VARIATION',
        'ARRIVAL_ACTUAL_INGATE_UTC',
        'ARRIVAL_ACTUAL_INGATE_LOCAL',
    ]
    df = df[selected_columns]
    df = df.dropna().reset_index(drop=True)
    return df

def convert_time_columns(df, time_cols):
    """
    Converts specified columns in the DataFrame to datetime.
    """
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
    return df

def calculate_time_offsets(df):
    """
    Computes the departure and arrival offsets as the difference between local and UTC times.
    """
    df['DEPARTURE_OFFSET'] = df['DEPARTURE_ACTUAL_OUTGATE_LOCAL'] - df['DEPARTURE_ACTUAL_OUTGATE_UTC']
    df['ARRIVAL_OFFSET'] = df['ARRIVAL_ACTUAL_INGATE_LOCAL'] - df['ARRIVAL_ACTUAL_INGATE_UTC']
    return df

def calculate_scheduled_utc(df):
    """
    Calculates the scheduled departure and arrival times in UTC using the offsets.
    """
    df['SCHEDULED_DEPARTURE_UTC'] = df['SCHEDULED_DEPARTURE_TIME_LOCAL'] - df['DEPARTURE_OFFSET']
    df['SCHEDULED_ARRIVAL_UTC'] = df['SCHEDULED_ARRIVAL_TIME_LOCAL'] - df['ARRIVAL_OFFSET']
    return df

def calculate_delays(df):
    """
    Calculates departure and arrival delays in minutes.
    Also converts the actual variation columns to minutes.
    """
    df['DEPARTURE_DELAY_MIN'] = (
        df['DEPARTURE_ACTUAL_OUTGATE_UTC'] - df['SCHEDULED_DEPARTURE_UTC']
    ).dt.total_seconds() / 60.0

    df['ARRIVAL_DELAY_MIN'] = (
        df['ARRIVAL_ACTUAL_INGATE_UTC'] - df['SCHEDULED_ARRIVAL_UTC']
    ).dt.total_seconds() / 60.0

    # Convert variations to minutes
    df['DEPARTURE_ACTUAL_OUTGATE_VARIATION_MIN'] = pd.to_timedelta(
        df['DEPARTURE_ACTUAL_OUTGATE_VARIATION']
    ).dt.total_seconds() / 60.0
    df['ARRIVAL_ACTUAL_INGATE_VARIATION_MIN'] = pd.to_timedelta(
        df['ARRIVAL_ACTUAL_INGATE_VARIATION']
    ).dt.total_seconds() / 60.0

    return df

def filter_by_delay_variation(df):
    """
    Filters rows where the calculated delay matches the actual variation.
    """
    df = df[df['DEPARTURE_ACTUAL_OUTGATE_VARIATION_MIN'] == df['DEPARTURE_DELAY_MIN']]
    df = df[df['ARRIVAL_ACTUAL_INGATE_VARIATION_MIN'] == df['ARRIVAL_DELAY_MIN']]
    return df

def drop_unused_columns(df):
    """
    Drops columns that are not needed for further analysis.
    """
    columns_to_drop = [
        'SCHEDULED_DEPARTURE_TIME_LOCAL',
        'DEPARTURE_ACTUAL_OUTGATE_TIMELINESS',
        'DEPARTURE_ACTUAL_OUTGATE_VARIATION',
        'DEPARTURE_ACTUAL_OUTGATE_LOCAL',
        'DEPARTURE_OFFSET',
        'DEPARTURE_ACTUAL_OUTGATE_VARIATION_MIN',
        'SCHEDULED_ARRIVAL_TIME_LOCAL',
        'ARRIVAL_ACTUAL_INGATE_TIMELINESS',
        'ARRIVAL_ACTUAL_INGATE_VARIATION',
        'ARRIVAL_ACTUAL_INGATE_LOCAL',
        'ARRIVAL_OFFSET',
        'ARRIVAL_ACTUAL_INGATE_VARIATION_MIN'
    ]
    df.drop(columns=columns_to_drop, inplace=True)
    return df

# drop duplicates
def drop_duplicates(df):
    """
    Drops duplicate rows from the DataFrame.
    """
    df.drop_duplicates(inplace=True)
    return df

def calculate_turnaround_time(df):
    """
    Calculates the turnaround time for each flight.
    """
    # df = df.sort_values(
    #     by=["AIRCRAFT_REGISTRATION_NUMBER", "DEPARTURE_ACTUAL_OUTGATE_UTC"]
    # ).reset_index(drop=True)

    df = df.reset_index(drop=False).rename(columns={"index":"IDX"})

    # Let's rename a few columns to standard short names for clarity:
    #  - REG for aircraft registration
    #  - APT for airport
    df = df.rename(
        columns={
            "AIRCRAFT_REGISTRATION_NUMBER": "REG",
            "ARRIVAL_IATA_AIRPORT_CODE": "ARR_APT",
            "DEPARTURE_IATA_AIRPORT_CODE": "DEP_APT",
            "ARRIVAL_ACTUAL_INGATE_UTC": "ARR_TIME",
            "DEPARTURE_ACTUAL_OUTGATE_UTC": "DEP_TIME",
        }
    )

    # Suppose your threshold
    THRESHOLD_HOURS = 6
    threshold_timedelta = pd.Timedelta(hours=THRESHOLD_HOURS)

    # --- Step 1. Split into arrivals and departures sub-dataframes ---
    arrivals = df[["IDX","REG","ARR_APT","ARR_TIME"]].copy()
    arrivals.rename(columns={"IDX":"IDX_ARR","ARR_APT":"APT"}, inplace=True)

    departures = df[["IDX","REG","DEP_APT","DEP_TIME"]].copy()
    departures.rename(columns={"IDX":"IDX_DEP","DEP_APT":"APT"}, inplace=True)

    # --- Step 2. Self-merge on (REG, APT) ---
    merged = pd.merge(
        arrivals,
        departures,
        on=["REG","APT"],      # same aircraft & same airport
        how="inner"
    )
    # Now we have all pairs of flights that share the same tail & airport.

    # --- Step 3. Filter for flights where the departure is strictly after the arrival ---
    merged = merged[ merged["ARR_TIME"] < merged["DEP_TIME"] ].copy()

    # --- Step 4. Filter out flights where DEP_TIME is > 24 hours after ARR_TIME (or any threshold)
    merged = merged[ (merged["DEP_TIME"] - merged["ARR_TIME"]) <= threshold_timedelta ].copy()

    # --- Step 5. Among the possible matches, keep only the earliest “valid” departure for each arrival
    merged = merged.sort_values(["IDX_ARR","DEP_TIME"])
    merged = merged.groupby("IDX_ARR", as_index=False).first()

    # --- Step 6. Compute turnaround in minutes ---
    merged["TURNAROUND_MIN"] = (
        merged["DEP_TIME"] - merged["ARR_TIME"]
    ).dt.total_seconds() / 60.0

    # --- Step 7. Merge back to original df ---
    df = pd.merge(
        df,
        merged[["IDX_ARR","IDX_DEP","TURNAROUND_MIN"]],
        left_on="IDX", 
        right_on="IDX_ARR",
        how="left"
    )

    # drop rown with missing turnaround time
    df.dropna(subset=["TURNAROUND_MIN"], inplace=True)

    # --- Step 8. Drop intermediate columns ---
    df.drop(columns=["IDX_ARR","IDX_DEP"], inplace=True
    )

    # --- Step 9. drop IDX column and rename columns to original names ---
    df.drop(columns=["IDX"], inplace=True)
    df.rename(columns={
                "REG":"AIRCRAFT_REGISTRATION_NUMBER",
                "ARR_APT":"ARRIVAL_IATA_AIRPORT_CODE",
                "DEP_APT":"DEPARTURE_IATA_AIRPORT_CODE",
                "ARR_TIME":"ARRIVAL_ACTUAL_INGATE_UTC",
                "DEP_TIME":"DEPARTURE_ACTUAL_OUTGATE_UTC",
            }, inplace=True)

    return df

def extract_delay_turnaround(input_path):
    """
    Extracts the delay/turnaround time for flights in the given dataset.
    """
    df = read_and_filter_data(input_path)
    df = select_and_clean_columns(df)

    time_cols = [
        'SCHEDULED_DEPARTURE_TIME_LOCAL',
        'DEPARTURE_ACTUAL_OUTGATE_UTC',
        'DEPARTURE_ACTUAL_OUTGATE_LOCAL',
        'SCHEDULED_ARRIVAL_TIME_LOCAL',
        'ARRIVAL_ACTUAL_INGATE_UTC',
        'ARRIVAL_ACTUAL_INGATE_LOCAL'
    ]
    df = convert_time_columns(df, time_cols)
    df = calculate_time_offsets(df)
    df = calculate_scheduled_utc(df)
    df = calculate_delays(df)
    df = filter_by_delay_variation(df)
    df = drop_unused_columns(df)
    df = drop_duplicates(df)

    column_order = [
        'AIRCRAFT_REGISTRATION_NUMBER',
        'IATA_CARRIER_CODE',
        'DEPARTURE_IATA_AIRPORT_CODE',
        'ARRIVAL_IATA_AIRPORT_CODE',
        'AIRCRAFT_TYPE_IATA',
        'SCHEDULED_DEPARTURE_UTC',
        'DEPARTURE_ACTUAL_OUTGATE_UTC',
        'DEPARTURE_DELAY_MIN',
        'SCHEDULED_ARRIVAL_UTC',
        'ARRIVAL_ACTUAL_INGATE_UTC',
        'ARRIVAL_DELAY_MIN'
    ]
    df = df[column_order]
    df = df.reset_index(drop=True)
    df = calculate_turnaround_time(df)
    return df



def prepare_data(input_path, output_train_path, output_test_path, test_size=0.2, random_state=42):
    """
    Prepares the flight data for analysis by extracting delay and turnaround information,
    and splits the data into train and test sets.
    """
    # Extract and process the data
    df = extract_delay_turnaround(input_path)
    
    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Save the train and test sets to their respective files
    train_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    
    print(f"Data preparation complete.")
    print(f"Train data saved to {output_train_path}")
    print(f"Test data saved to {output_test_path}")
    return train_df, test_df

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Prepare flight data for analysis and split into train/test sets.")
    parser.add_argument("--input_path", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_train_path", required=True, help="Path to save the train CSV file.")
    parser.add_argument("--output_test_path", required=True, help="Path to save the test CSV file.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the data to include in the test split (default: 0.2).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    args = parser.parse_args()

    # Call the prepare_data function with the provided arguments
    prepare_data(args.input_path, args.output_train_path, args.output_test_path, args.test_size, args.random_state)

    # Example usage:
    # python prepare_data.py --input_path data/flights.csv --output_train_path data/real/train.csv --output_test_path data/real/test.csv --test_size 0.2 --random_state 42

