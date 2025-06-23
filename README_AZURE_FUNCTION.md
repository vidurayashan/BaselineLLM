# Azure Function Integration for NMI Consumption Forecast Dashboard

## Overview

The NMI Consumption Forecast Dashboard has been updated to use an Azure Function for model training instead of local processing. This change offloads the computationally intensive model training to the cloud, improving performance and scalability.

## Changes Made

### 1. Removed Local Dependencies
- Removed import of `train_and_pred.py` library
- Eliminated local model training and data processing functions
- Added `requests` library for HTTP communication with Azure Function

### 2. Updated Data Flow
- The `load_data()` function now sends requests to Azure Function
- Model training, prediction, and forecasting are handled remotely
- Results are returned as JSON and converted to DataFrames for plotting

### 3. Configuration Files
- Created `metadata/meter_info.json` to replace database calls for meter information
- Updated `requirements.txt` to include `requests` library

## Configuration

### Azure Function URL

You need to configure the Azure Function URL in your Streamlit secrets. Create a `.streamlit/secrets.toml` file with:

```toml
AZURE_FUNCTION_URL = "https://your-function-app.azurewebsites.net/api/nmitrain"
```

For local testing, you can use:
```toml
AZURE_FUNCTION_URL = "http://localhost:7071/api/nmitrain"
```

### Expected Azure Function Response Format

The Azure Function should return a JSON response with the following structure:

```json
{
  "model_performance": {
    "train_rmse": 123.45,
    "train_mape": 0.1234,
    "test_rmse": 234.56,
    "test_mape": 0.2345
  },
  "training_data": {
    "dates": ["2018-01-01", "2018-01-02", ...],
    "actual_values": [100.0, 110.0, ...]
  },
  "testing_data": {
    "dates": ["2022-01-01", "2022-01-02", ...],
    "actual_values": [120.0, 130.0, ...],
    "predicted_values": [125.0, 135.0, ...]
  },
  "forecast_data": {
    "dates": ["2025-01-01", "2025-01-02", ...],
    "predicted_values": [140.0, 150.0, ...]
  }
}
```

### Request Format

The app sends the following JSON payload to the Azure Function:

```json
{
  "nmi_id": "1",
  "start_date": "20180101",
  "end_date": "20250101",
  "test_dates": {
    "test_start_date": 20220101,
    "test_end_date": 20221231
  },
  "domain_facts": [
    "Data before 2020 mid has a different pattern than after 2020 mid",
    "Campus semester usually run from March to end of May, then end of July to end of October"
  ]
}
```

## Testing

Use the provided test script to verify Azure Function connectivity:

```bash
python test_azure_function.py
```

## Benefits

1. **Scalability**: Model training is offloaded to cloud infrastructure
2. **Performance**: Faster response times for the Streamlit app
3. **Resource Management**: Reduced local computational requirements
4. **Maintenance**: Centralized model training logic in Azure Function

## Migration Notes

- The app no longer requires the `train_and_pred.py` file
- Database connections are no longer needed in the Streamlit app
- All model training logic should be implemented in the Azure Function
- The Azure Function should handle all the data processing, feature engineering, and model training that was previously done locally

## Error Handling

The app includes error handling for:
- Network connectivity issues
- Invalid responses from Azure Function
- Missing configuration

Error messages are displayed to users when issues occur. 