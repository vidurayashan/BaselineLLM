import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from joblib import dump
from dynaconf import settings
import pyodbc
import logging
from functools import lru_cache
from joblib import Memory
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy import stats
from pycaret.regression import *
from openai import OpenAI

memory = Memory(location='./cachedir', verbose=0)

def get_db_connection():
    server = settings.get('DB_SERVER', '')
    database = settings.get('DB_NAME', '')
    username = settings.get('DB_USERNAME', '')
    password = settings.get('DB_PASSWORD', '')

    connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database \
        + ';UID=' + username + ';PWD=' + password
    return pyodbc.connect(connection_string)

METER_DAILY_CONSUMPTION_SQL = \
"""WITH DistinctReadingDates AS (
    SELECT DISTINCT
        ReadingDateTime,
        CONVERT(DATE,[ReadingDateTime]) as date,
        E AS EnergyConsumption,
        KVA AS MeterReading
    FROM dbo.ImportShell
    WHERE NetworkId = %s
        AND ReadingDateTime >= '%s'
        AND KVA IS NOT NULL
)

SELECT
    date,
    SUM(EnergyConsumption) AS metered_consumption,
    MAX(MeterReading) AS peak_demand
FROM DistinctReadingDates
GROUP BY date
ORDER BY date;
"""

METER_DAILY_CONSUMPTION_SQL_NEW_MAIN = \
"""
WITH DistinctReadingDates AS (
    SELECT DISTINCT
        ReadingDateTime,
        CONVERT(DATE,[ReadingDateTime]) as date,
        E AS EnergyConsumption,
        KVA AS MeterReading
    FROM [Leap].[dbo].[ImportShell]
    WHERE NMI=STR(6001733781) OR NetworkId = %s
        AND ReadingDateTime >= '%s'
        AND KVA IS NOT NULL
)

SELECT
    date,
    SUM(EnergyConsumption) AS metered_consumption,
    MAX(MeterReading) AS peak_demand
FROM DistinctReadingDates
GROUP BY date
ORDER BY date;
"""

METER_SOLAR_GENERATION = \
"""SELECT
    Timestamp AS ReadingDateTime,
    SolarPowerReading
FROM
    dbo.SolarPowerReadings_kVA
WHERE
    NetworkID = %s
ORDER BY
    Timestamp ASC
"""

METER_DETAILS = \
"""SELECT *
    FROM [Leap].[dbo].[DimNetwork]
    WHERE IsActive = 1
    ORDER BY NetworkID
"""

@memory.cache
def get_meter_info():
    connection = get_db_connection()
    df_meter_info = pd.read_sql(METER_DETAILS, connection)
    connection.close()
    return df_meter_info

@memory.cache
def get_meter_daily_consumption(nmi_id, start_date):
    connection = get_db_connection()
    if nmi_id == 1:
        df_nmi = pd.read_sql(METER_DAILY_CONSUMPTION_SQL_NEW_MAIN % (str(nmi_id), str(start_date)), connection)
    else:
        df_nmi = pd.read_sql(METER_DAILY_CONSUMPTION_SQL % (str(nmi_id), str(start_date)), connection)
    connection.close()
    df_nmi['DateKey']          = pd.to_datetime(df_nmi['date']).dt.strftime('%Y%m%d').astype(int)
    df_nmi.drop(columns=["date"], inplace=True)
    df_nmi = df_nmi[df_nmi['DateKey'] >= 20180101].reset_index(drop=True)
    df_nmi = df_nmi[['DateKey','metered_consumption', 'peak_demand']]
    df_nmi_cleaned = remove_outliers_combined(df_nmi, consumption_col='metered_consumption')
    return df_nmi_cleaned

@memory.cache
def get_meter_solar_generation(nmi_id):
    connection = get_db_connection()
    df_solar = pd.read_sql(METER_SOLAR_GENERATION % (str(nmi_id)), connection)
    connection.close()
    return df_solar

def preprocess_solar_data(df_solar):
    df_solar['date_time'] = pd.to_datetime(df_solar['ReadingDateTime'])
    df_solar_agg = df_solar.groupby(df_solar['date_time'].dt.date).agg({
        'SolarPowerReading': 'sum'
    }).reset_index().sort_values(by=['date_time'])

    # Rename the index column to be more descriptive
    df_solar_agg = df_solar_agg.rename(columns={'date_time': 'DateKey'})
    df_solar_agg['DateKey'] = pd.to_datetime(df_solar_agg['DateKey']).dt.strftime('%Y%m%d').astype(int)
    return df_solar_agg

def merge_meter_n_solar_data(df_nmi, df_solar):
    # Create daily aggregation with all fields
    
    df_solar_agg = preprocess_solar_data(df_solar)

    df_merged = df_nmi.merge(
        df_solar_agg,
        on='DateKey',
        how='left'
    )

    df_merged['SolarPowerReading'] = df_merged['SolarPowerReading'].fillna(0)
    df_merged['consumption'] = df_merged['metered_consumption'] #+ df_merged['SolarPowerReading']
    df_merged['DatePlot'] = pd.to_datetime(df_merged["DateKey"], format="%Y%m%d")

    return df_merged

@memory.cache
def get_time_data(start_date='20180101', end_date='20250101'):
    connection = get_db_connection()
    date = pd.read_sql("SELECT * FROM dbo.DimDate WHERE DateKey >= %s AND DateKey <= %s" % (start_date, end_date),
                       connection)
    time = pd.read_sql("SELECT * FROM dbo.DimTime", connection)
    connection.close()
    return date, time

@memory.cache
def get_temperature_data(nmi_id, start_date='20180101', end_date='20250101'):
    connection = get_db_connection()
    formatted_start_date = datetime.strptime(str(start_date), "%Y%m%d").strftime("%Y-%m-%d")
    formatted_end_date   = datetime.strptime(str(end_date), "%Y%m%d").strftime("%Y-%m-%d")

    df_meter_info = get_meter_info()    
    campus_key = df_meter_info[df_meter_info['NetworkID'] == nmi_id]['CampusKey'].values[0]

    # Create a unique filename based on parameters
    cache_filename = f"cachedir/temperature_data_campus{campus_key}_{start_date}_{end_date}.csv"

    # Check if cached file exists with the same parameters
    if os.path.exists(cache_filename):
        # Read from cache file if it exists
        reading_temperature = pd.read_csv(cache_filename)
    else:
        # If not cached, query the database
        reading_temperature = pd.read_sql("""SELECT[DateKey]
                    ,[TimeKey]
                    ,[ApparentTemperature]
                    ,[AirTemperature]
                    ,[DewPointTemperature]
                    ,[RelativeHumidity]
                FROM [Leap].[dbo].[vwClimate] WHERE CampusKey= %s AND DateKey >= %s AND DateKey <= %s"""
                                            % (campus_key, start_date, end_date), connection)
        # Save to cache file for future use
        reading_temperature.to_csv(cache_filename, index=False)
    
    connection.close()

    date, time = get_time_data(start_date, end_date)
    reading_temperature['Timestamp'] = pd.to_datetime(
        reading_temperature.DateKey.astype(
            str) + " " + reading_temperature['TimeKey'].astype(str).str.zfill(6),
        format="ISO8601")

    temp_range = pd.date_range(reading_temperature['Timestamp'].dt.date.min(axis=0),
                                reading_temperature['Timestamp'].max(axis=0),
                                freq='15min',
                                name="Timestamp").to_frame().reset_index(drop=True)
    temp_out = pd.merge(temp_range, reading_temperature, how="left", left_on="Timestamp", right_on="Timestamp").fillna(
        method='ffill')
    temp_out.TimeKey = temp_out.Timestamp.dt.strftime("%H%M%S").astype(int)
    temp_out.DateKey = temp_out.Timestamp.dt.strftime("%Y%m%d").astype(int)
    reading_temperature = temp_out.drop(columns=["Timestamp"])

    date_data = date.drop_duplicates(subset='DateKey', keep='first')
    date_data = date_data.drop(columns=['Date', 'DaySuffix', 'WeekDayName', 'HolidayText', 'DayOfYear',
                                        'ISOWeekOfYear', 'MonthName', 'QuarterName', 'MMYYYY', 'MonthYear',
                                        'FirstDayOfMonth', 'LastDayOfMonth', 'FirstDayOfQuarter', 'LastDayOfQuarter',
                                        'FirstDayOfYear', 'LastDayOfYear', 'FirstDayOfNextMonth', 'FirstDayOfNextYear',
                                        'IsSemester', 'IsExamPeriod', 'CalendarSignificance',
                                        'HasCalendarSignificance'])
    date_data["IsWeekend"] = date_data["IsWeekend"].astype(int)
    date_data["IsHoliday"] = date_data["IsHoliday"].astype(int)

    # preprocess the time data
    time_data = time.drop_duplicates(subset='TimeKey', keep='first')
    time_data = time_data.drop(
        columns=['Hour24ShortString', 'Hour24FullString', 'Hour24MinString', 'Hour12', 'Hour12ShortString',
                    'Hour12MinString',
                    'Hour12FullString', 'AmPmString', 'MinuteCode', 'MinuteShortString', 'MinuteFullString24',
                    'MinuteFullString12', 'HalfHourShortString', 'HalfHourCode',
                    'HalfHourFullString12', 'SecondShortString', 'Second', 'FullTimeString12',
                    'FullTime'])



    reading_temperature.reset_index(drop=True, inplace=True)
    reading_temperature["DateKey"] = pd.to_datetime(reading_temperature["DateKey"], format="%Y%m%d")

    df_temperature_daily = reading_temperature.groupby(reading_temperature['DateKey'].dt.date).agg({
        'ApparentTemperature': 'mean',
        'AirTemperature': 'mean',
        'DewPointTemperature': 'mean',
        'RelativeHumidity': 'mean'
    }).reset_index()

    df_temperature_daily['DateKey'] = pd.to_datetime(df_temperature_daily['DateKey']).dt.strftime('%Y%m%d').astype(int)

    return date_data, df_temperature_daily

def get_temperature_daily_data(nmi_id, start_date, end_date):
    date_data, df_temperature_daily = get_temperature_data(nmi_id, start_date, end_date)
    df_temperature_daily_data = pd.merge(df_temperature_daily, date_data, on='DateKey', how='left')
    return df_temperature_daily_data

def merge_nmi_n_temperature_data(df_nmi, df_solar, nmi_id, start_date, end_date):
    df_temperature_daily_data = get_temperature_daily_data(nmi_id, start_date, end_date)
    df_merged = merge_meter_n_solar_data(df_nmi, df_solar)
    df_nmi_daily_data = pd.merge(df_merged, df_temperature_daily_data, on='DateKey', how='left')

    return df_nmi_daily_data

def get_train_test_data(nmi_id, start_date, end_date):
    df_nmi = get_meter_daily_consumption(nmi_id, start_date)
    df_solar = get_meter_solar_generation(nmi_id)
    df_nmi_daily_data = merge_nmi_n_temperature_data(df_nmi, df_solar, nmi_id, start_date, end_date)
    df_nmi_X = df_nmi_daily_data.drop(columns=["consumption", "metered_consumption", "peak_demand", "SolarPowerReading"])
    df_nmi_Y = df_nmi_daily_data[['DateKey', 'consumption', 'DatePlot', 'metered_consumption']]

    return df_nmi_X, df_nmi_Y

def train_test_split(df_nmi_X, df_nmi_Y, test_dates):

    if "test_start_date" not in test_dates:
        test_dates["test_start_date"] = 20220101
    if "test_end_date" not in test_dates:
        test_dates["test_end_date"] = 20221231
    if "remove_start" not in test_dates:
        test_dates["remove_start"] = 20170101
    if "remove_end" not in test_dates:
        test_dates["remove_end"] = 20171231
        
    test_start_date = test_dates["test_start_date"]
    test_end_date   = test_dates["test_end_date"]
    remove_start_date = test_dates["remove_start"]
    remove_end_date   = test_dates["remove_end"]

    # df_nmi_test_X  = df_nmi_X[(df_nmi_X["DateKey"] > test_start_date)  & (df_nmi_X["DateKey"] <= test_end_date)]
    # df_nmi_train_X = df_nmi_X[(df_nmi_X["DateKey"] <= train_start_date) | (df_nmi_X["DateKey"] > train_end_date)]
    # df_nmi_train_X = df_nmi_train_X[ df_nmi_train_X["DateKey"] <= train_end_date ]

    # df_nmi_test_y = df_nmi_Y[(df_nmi_Y["DateKey"] > test_start_date) & (df_nmi_Y["DateKey"] <= test_end_date)]
    # df_nmi_train_y = df_nmi_Y[(df_nmi_Y["DateKey"] <= train_start_date) | (df_nmi_Y["DateKey"] > train_end_date)]
    # df_nmi_train_y = df_nmi_train_y[ df_nmi_train_y["DateKey"] <= train_end_date ]

    remove_mask = (df_nmi_X["DateKey"] >= remove_start_date) & (df_nmi_X["DateKey"] <= remove_end_date)
    df_nmi_X = df_nmi_X[~remove_mask]
    df_nmi_Y = df_nmi_Y[~remove_mask]

    test_mask = (df_nmi_X["DateKey"] >  test_start_date) & (df_nmi_X["DateKey"] <= test_end_date)
    

    df_nmi_test_X  = df_nmi_X[test_mask]
    df_nmi_train_X = df_nmi_X[~test_mask]

    df_nmi_test_y = df_nmi_Y[test_mask]
    df_nmi_train_y = df_nmi_Y[~test_mask]

    return df_nmi_test_X, df_nmi_train_X, df_nmi_test_y, df_nmi_train_y

def get_model():
    best_params = {
        'colsample_bytree': 0.6,
        'gamma': 1,
        'learning_rate': 0.05,
        'max_depth': 3,
        'n_estimators': 1000,
        'reg_alpha': 1,
        'reg_lambda': 10,
        'subsample': 0.7
    }

    model = XGBRegressor(**best_params)
    return model

def prep_train_test_data(df_nmi_train_X, df_nmi_test_X, df_nmi_train_y, df_nmi_test_y):
    df_nmi_train_X.drop(columns=['DateKey', 'DatePlot'], inplace=True)
    df_nmi_test_X.drop(columns=['DateKey', 'DatePlot'], inplace=True)

    df_nmi_train_y.drop(columns=['DatePlot', 'DateKey', 'metered_consumption'], inplace=True)
    df_nmi_test_y.drop(columns=['DatePlot', 'DateKey', 'metered_consumption'], inplace=True)

    return df_nmi_train_X, df_nmi_test_X, df_nmi_train_y, df_nmi_test_y

def train_model_autoML(df_nmi_train_X, df_nmi_train_y):
    df_train = df_nmi_train_X.copy()
    df_train['target'] = df_nmi_train_y  # Replace 'target' with your actual target column name if needed

    # 2. Setup PyCaret
    regression_setup = setup(
        data=df_train,
        target='target',
        session_id=42,
        fold_strategy='kfold',
        fold=5,
        use_gpu=True,  # Optional: enable if using GPU
        verbose=False
    )

    xgb_model = create_model('xgboost')
    tuned_xgb = tune_model(xgb_model, optimize='MAPE')
    return tuned_xgb

def train_model(df_nmi_train_X, df_nmi_train_y, nmi_id, autoML=True):
    import json

    # Define cache file path for this NMI's hyperparameters
    cache_file = f'./cachedir/xgb_params_{nmi_id}.json'
    
    # Check if cached parameters exist
    if os.path.exists(cache_file):
        # Load cached parameters and use them
        with open(cache_file, 'r') as f:
            best_params = json.load(f)
        
        print("Model loaded from cache")
        model = XGBRegressor(**best_params)
        model.fit(df_nmi_train_X, df_nmi_train_y)
    else:
        # First iteration - use autoML to find best parameters
        if autoML:
            model = train_model_autoML(df_nmi_train_X, df_nmi_train_y)
            
            # Extract hyperparameters from the tuned model
            best_params = model.get_params()
            
            # Save parameters to cache file
            os.makedirs('./cachedir', exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(best_params, f, indent=2)
        else:
            model = get_model()
            model.fit(df_nmi_train_X, df_nmi_train_y)

    y_pred_train = model.predict(df_nmi_train_X)
    mse_train = mean_squared_error(df_nmi_train_y, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    mape_train = mean_absolute_percentage_error(df_nmi_train_y, y_pred_train)
    return model, rmse_train, mape_train

def predict_model(model, df_nmi_test_X, df_nmi_test_y):
    df_nmi_test_y_pred = model.predict(df_nmi_test_X)
    mse_test = mean_squared_error(df_nmi_test_y, df_nmi_test_y_pred)
    rmse_test = np.sqrt(mse_test)
    mape_test = mean_absolute_percentage_error(df_nmi_test_y, df_nmi_test_y_pred)
    return df_nmi_test_y_pred, rmse_test, mape_test

def evaluate_model(df_nmi_test_y, df_nmi_test_y_pred):
    rmse = np.sqrt(mean_squared_error(df_nmi_test_y, df_nmi_test_y_pred))
    mape = mean_absolute_percentage_error(df_nmi_test_y, df_nmi_test_y_pred)
    return rmse, mape


def get_temperature_models(nmi_id, start_date, end_date):
    temperature_models = {}
    best_params = {
        'colsample_bytree': 0.6,
        'gamma': 1,
        'learning_rate': 0.05,
        'max_depth': 3,
        'n_estimators': 1000,
        'reg_alpha': 1,
        'reg_lambda': 10,
        'subsample': 0.7
    }

    temperature_models['AirTemperature']      = XGBRegressor(**best_params)
    temperature_models['ApparentTemperature'] = XGBRegressor(**best_params)
    temperature_models['DewPointTemperature'] = XGBRegressor(**best_params)
    temperature_models['RelativeHumidity']    = XGBRegressor(**best_params)

    df_temperature_daily_data = get_temperature_daily_data(nmi_id, start_date, end_date)

    df_temperature_daily_data_X = df_temperature_daily_data.drop(columns=['DateKey', 
                                                                      'ApparentTemperature', 'AirTemperature', 
                                                                      'DewPointTemperature', 'RelativeHumidity'])


    temperature_models['AirTemperature'].fit(df_temperature_daily_data_X, df_temperature_daily_data['AirTemperature'])
    temperature_models['ApparentTemperature'].fit(df_temperature_daily_data_X, df_temperature_daily_data['ApparentTemperature'])
    temperature_models['DewPointTemperature'].fit(df_temperature_daily_data_X, df_temperature_daily_data['DewPointTemperature'])
    temperature_models['RelativeHumidity'].fit(df_temperature_daily_data_X, df_temperature_daily_data['RelativeHumidity'])

    return temperature_models

def get_future_dates():
    # Create date range
    start_date = pd.to_datetime('20250101', format='%Y%m%d')
    end_date   = pd.to_datetime('20261231', format='%Y%m%d')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create DataFrame with date as index initially
    future_dates = pd.DataFrame(index=date_range)

    # Add all date components
    future_dates['DateKey'] = future_dates.index.strftime('%Y%m%d').astype(int)
    future_dates['DatePlot'] = pd.to_datetime(future_dates["DateKey"], format="%Y%m%d")
    future_dates['Day'] = future_dates.index.day
    future_dates['Weekday'] = future_dates.index.weekday + 1
    future_dates['IsWeekend'] = (future_dates.index.weekday >= 5).astype(int)
    future_dates['Month'] = future_dates.index.month
    future_dates['Quarter'] = future_dates.index.quarter
    future_dates['Year'] = future_dates.index.year

    # Calculate WeekOfYear properly
    future_dates['WeekOfYear'] = future_dates.index.isocalendar().week

    # Calculate DOWInMonth
    future_dates['DOWInMonth'] = future_dates.groupby(['Year', 'Month', 'Weekday']).cumcount() + 1

    # Calculate WeekOfMonth
    future_dates['WeekOfMonth'] = future_dates.groupby(['Year', 'Month']).cumcount() // 7 + 1

    # Initialize IsHoliday as 0
    future_dates['IsHoliday'] = 0

    # Reorder columns to match your format
    future_dates = future_dates[['DateKey', 'Day', 'Weekday', 'IsWeekend', 'IsHoliday', 
                            'DOWInMonth', 'WeekOfMonth', 'WeekOfYear', 'Month', 
                            'Quarter', 'Year', 'DatePlot']]

    future_dates.reset_index(inplace=True, drop='index')
    df_future_dates = future_dates.drop(columns=['DateKey'])

    return df_future_dates, future_dates


def predict_weather(nmi_id, start_date, end_date):
    df_future_dates, _ = get_future_dates()
    temperature_models = get_temperature_models(nmi_id, start_date, end_date)
    df_future_temperature = df_future_dates.copy()
    df_future_dates.drop(columns=['DatePlot'], inplace=True)
    for temp_type, model in temperature_models.items():
        y_pred = model.predict(df_future_dates)
        df_future_temperature[temp_type] = y_pred
    
    df_future_temperature = df_future_temperature[['ApparentTemperature', 'AirTemperature',
                                                    'DewPointTemperature', 'RelativeHumidity', 
                                                    'Day', 'Weekday', 'IsWeekend', 'IsHoliday', 
                                                    'DOWInMonth', 'WeekOfMonth', 'WeekOfYear', 
                                                    'Month', 'Quarter', 'Year', 'DatePlot']]
    return df_future_temperature

# def get_future_X(nmi_id, start_date, end_date):
#     df_future_temperature = predict_weather(nmi_id, start_date, end_date)
#     df_future_dates, future_dates = get_future_dates()
#     return df_future_temperature, future_dates
    
def forecast_nmi_consumption(model, nmi_id, start_date, end_date, df_future, codes=[]):
    df_future_temperature = predict_weather(nmi_id, start_date, end_date)
    df_future_dates, future_dates = get_future_dates()

    # df_LLM = df_future_temperature.copy()
    # globals()['df_LLM'] = df_LLM
    # if len(codes) > 0:
    #     for code in codes:
    #         exec(code, globals())
    
    df_future_backup = df_future.copy()
    df_future_backup.drop(columns=['DatePlot', 'DateKey'], inplace=True)

    y_pred = model.predict(df_future_backup)
    return df_future, y_pred


def solar_farm_data(nmi_id, start_date, end_date):
    df_solar_andrew = pd.read_excel("gridpowerkWh.xlsx")
    df_solar_andrew['DateKey'] = pd.to_datetime(df_solar_andrew['timestamp']).dt.strftime('%Y%m%d').astype(int)
    df_solar_andrew['DateKey'] = pd.to_datetime(df_solar_andrew['DateKey'], format="%Y%m%d")

    df_andrew_nmi_daily = df_solar_andrew.groupby(df_solar_andrew['DateKey'].dt.date).agg({
        'Grid Power kWh': 'sum'
    }).reset_index()

    df_andrew_nmi_daily['Day'] = pd.to_datetime(df_andrew_nmi_daily['DateKey'], format='%Y%m%d').dt.day
    df_andrew_nmi_daily['Month'] = pd.to_datetime(df_andrew_nmi_daily['DateKey'], format='%Y%m%d').dt.month

    df_future_temperature = predict_weather(nmi_id, start_date, end_date)
    df_merged_future = df_andrew_nmi_daily.merge(
        df_future_temperature,
        on=['Day', 'Month'],
        how='left'
    ).drop(columns=['DateKey']).sort_values(by=['Year', 'Month', 'Day']).reset_index(drop=True)

    df_solar = get_meter_solar_generation(nmi_id)
    df_solar_agg = preprocess_solar_data(df_solar)

    df_solar_agg['DateKey'] = pd.to_datetime(df_solar_agg['DateKey'], format="%Y%m%d")
    df_solar_agg['Day'] = df_solar_agg['DateKey'].dt.day
    df_solar_agg['Month'] = df_solar_agg['DateKey'].dt.month

    # Drop DateKey and aggregate SolarPowerReading by Month and Day
    df_solar_agg_yearly = df_solar_agg.drop('DateKey', axis=1)\
        .groupby(['Month', 'Day'])['SolarPowerReading']\
        .mean()\
        .reset_index()

    # Merge df_merged_future with df_solar_agg_yearly, keeping all records from df_merged_future
    df_merged_future = df_merged_future.merge(
        df_solar_agg_yearly,
        on=['Month', 'Day'],
        how='left'
    ).reset_index(drop=True)

    return df_merged_future

def remove_outliers_combined(df, consumption_col='consumption', z_score_threshold=3):
    """
    Remove outliers from a dataframe using two methods:
    1. Global z-score based outlier removal
    2. Yearly local outlier removal
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data. Must have a DateKey or DatePlot column for date information
        and the consumption column
    consumption_col : str, default='consumption'
        The name of the column containing consumption data
    z_score_threshold : float, default=2
        The z-score threshold to identify global outliers
        
    Returns:
    --------
    pandas.DataFrame
        A dataframe with outliers removed using both methods.
        If input had DateKey, it will be returned as integer in YYYYMMDD format.
        If input had DatePlot, both DatePlot (datetime) and DateKey (int YYYYMMDD) will be included.
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_clean = df.copy()
    
    # Determine which date column to use
    has_datekey = 'DateKey' in df_clean.columns
    has_dateplot = 'DatePlot' in df_clean.columns
    
    # Convert or create DateKey in YYYYMMDD format
    if has_datekey:
        # If DateKey is already datetime, convert to YYYYMMDD integer
        if pd.api.types.is_datetime64_any_dtype(df_clean['DateKey']):
            df_clean['DateKey'] = df_clean['DateKey'].dt.strftime('%Y%m%d').astype(int)
        # If DateKey is string, convert to YYYYMMDD integer
        elif df_clean['DateKey'].dtype == 'object':
            df_clean['DateKey'] = pd.to_datetime(df_clean['DateKey']).dt.strftime('%Y%m%d').astype(int)
        # If DateKey is already integer, ensure it's in YYYYMMDD format
        elif df_clean['DateKey'].dtype == 'int64':
            # It's already in the correct format, no need to modify
            pass
    elif has_dateplot:
        # Create DateKey from DatePlot
        df_clean['DateKey'] = df_clean['DatePlot'].dt.strftime('%Y%m%d').astype(int)
    else:
        raise ValueError("DataFrame must have either 'DateKey' or 'DatePlot' column")

    # Step 1: Global z-score based outlier removal
    df_clean = df_clean[np.abs(stats.zscore(df_clean[consumption_col])) < z_score_threshold]
    
    # Step 2: Yearly local outlier removal
    # Extract years from DateKey
    years = (df_clean['DateKey'] // 10000).unique()
    
    # Create a list to store cleaned yearly dataframes
    cleaned_yearly_dfs = []
    
    # Process each year separately
    for year in years:
        # Filter data for the current year
        year_df = df_clean[df_clean['DateKey'] // 10000 == year].copy()
        
        # Calculate yearly z-scores
        yearly_z_scores = np.abs(stats.zscore(year_df[consumption_col]))
        
        # Keep only non-outliers for this year
        year_df = year_df[yearly_z_scores < z_score_threshold]
        
        # Add to list of cleaned dataframes
        cleaned_yearly_dfs.append(year_df)
    
    # Combine all cleaned yearly dataframes
    df_final = pd.concat(cleaned_yearly_dfs, axis=0)
    
    # Sort by DateKey
    df_final = df_final.sort_values(by='DateKey')
    
    # Reset index
    df_final.reset_index(drop=True, inplace=True)
    
    return df_final

client = OpenAI(
    # This is the default and can be omitted
    api_key=settings.get('OPENAI_API_KEY', '')
)

def get_response(msg, temperature=0, top_p = 0, model = "gpt-4o-2024-08-06"):
    messages = [{"role": "user", "content": msg}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    return response.choices[0].message.content

def integrate_domain_knowledge(nmi_id, start_date, end_date, df_nmi_train_X, df_nmi_test_X, domain_facts=[]):

    df_future_temperature = predict_weather(nmi_id, start_date, end_date)

    df_nmi_train_X['set'] = 'train'
    df_nmi_test_X['set'] = 'test'
    df_future_temperature['set'] = 'future'

    df_LLM = pd.concat([df_nmi_train_X, df_nmi_test_X, df_future_temperature])
    globals()['df_LLM'] = df_LLM

    codes = []

    if len(domain_facts) > 0:
        # Load the prompt template from file
        with open('metadata/code_generation.prompt', 'r') as f:
            prompt_template = f.read()
        
        for fact in domain_facts:
            prompt = prompt_template.format(fact=fact)

            raw_resp = get_response(prompt)
            code = get_response("Show me the Python syntax of the following. Remove starting ```python and ending ```:\n\n" + raw_resp)
            print(f"Fact: {fact}")
            print(f"Code: {code}")
            exec(code, globals())
            codes.append(code)
            print(f"df_LLM.columns: {df_LLM.columns}")
            print("===============================================")
    

    df_nmi_train_X = df_LLM[df_LLM['set'] == 'train'].drop(columns='set')
    df_nmi_test_X = df_LLM[df_LLM['set'] == 'test'].drop(columns='set')
    df_future_temperature = df_LLM[df_LLM['set'] == 'future'].drop(columns='set')

    return df_nmi_train_X, df_nmi_test_X, df_future_temperature, df_LLM, codes