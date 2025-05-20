import pandas as pd
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a headline column to the dataframe from the input data.
    
    Args:
        df (pd.DataFrame): The current transformed dataframe
        input_df (pd.DataFrame): The original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with headline column added
    """
    # Make a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Add the headline column from input data
    if 'headline' in input_df.columns:
        result_df['headline'] = input_df['headline'].fillna('')
    else:
        result_df['headline'] = ''
    
    return result_df
