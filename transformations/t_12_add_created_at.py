"""
Transformation to add a created_at timestamp to each record.
"""
import pandas as pd
from datetime import datetime
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a created_at column with the current timestamp to the DataFrame.
    
    Args:
        df: The output DataFrame from previous transformations
        input_df: The original input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added created_at column
    """
    # Add current timestamp in ISO 8601 format (UTC)
    current_time = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    df['created_at'] = current_time
    
    return df
