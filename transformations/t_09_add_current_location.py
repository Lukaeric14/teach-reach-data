"""
Transformation to add the current location (city and country) for each teacher.
"""
import pandas as pd
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add current_location_city and current_location_country columns to the DataFrame.
    
    Args:
        df: The output DataFrame from previous transformations
        input_df: The original input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added location columns
    """
    # Add country and city columns directly from input (country first)
    df['current_location_country'] = input_df['country']
    df['current_location_city'] = input_df['city']
    
    # Clean up any missing values
    df['current_location_country'] = df['current_location_country'].fillna('Unknown')
    df['current_location_city'] = df['current_location_city'].fillna('Unknown')
    
    return df
