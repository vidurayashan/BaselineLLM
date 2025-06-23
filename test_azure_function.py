import requests
import json
import pandas as pd
from datetime import datetime

def test_azure_function():
    """Test the Azure function integration"""
    
    # Test data
    test_data = {
        "nmi_id": "1",  # Replace with your actual NMI ID
        "start_date": "20180101",
        "end_date": "20250101",
        "test_dates": {
            "test_start_date": "20220101",
            "test_end_date": "20221231"
        },
        "domain_facts": [
            "Data before 2020 mid has a different pattern than after 2020 mid",
            "Campus semester usually run from March to end of May, then end of July to end of October"
        ]
    }

    # Local function URL (when running locally)
    # local_url = "http://localhost:7072/api/nmitrain"
    # local_url = "http://localhost:7072/api/nmibaselinetrainllm"

    # local_url = "http://localhost:7072/api/nmitrain"
    local_url = "https://dynamicnmibaselines.azurewebsites.net/api/nmitrain"
    

    # local_url = "https://dynamicnmibaselines.azurewebsites.net/api/nmitrain"
    # Azure function URL (when deployed)
    # azure_url = "https://your-function-app.azurewebsites.net/api/train-nmi-model"

    try:
        # Make the request
        print("Sending request to function...")
        response = requests.post(local_url, json=test_data)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Print performance metrics
        print("\nModel Performance Metrics:")
        print(f"Train RMSE: {result['model_performance']['train_rmse']:.2f}")
        print(f"Train MAPE: {result['model_performance']['train_mape']:.2%}")
        print(f"Test RMSE: {result['model_performance']['test_rmse']:.2f}")
        print(f"Test MAPE: {result['model_performance']['test_mape']:.2%}")
        
        # Create DataFrames for plotting
        train_df = pd.DataFrame({
            'date': pd.to_datetime(result['training_data']['dates']),
            'actual': result['training_data']['actual_values']
        })
        
        test_df = pd.DataFrame({
            'date': pd.to_datetime(result['testing_data']['dates']),
            'actual': result['testing_data']['actual_values'],
            'predicted': result['testing_data']['predicted_values']
        })
        
        forecast_df = pd.DataFrame({
            'date': pd.to_datetime(result['forecast_data']['dates']),
            'predicted': result['forecast_data']['predicted_values']
        })
        
        print(f"\nData Summary:")
        print(f"Training data points: {len(train_df)}")
        print(f"Testing data points: {len(test_df)}")
        print(f"Forecast data points: {len(forecast_df)}")
        
        print(f"\nForecast date range: {forecast_df['date'].min()} to {forecast_df['date'].max()}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to Azure function: {str(e)}")
        return False
    except Exception as e:
        print(f"Error processing response from Azure function: {str(e)}")
        return False

if __name__ == "__main__":
    test_azure_function() 