"""
Transformation 17: Add a source_id column based on the id column from the input file
"""
import pandas as pd
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'source_id' column based on the 'id' column from the input file.
    
    Args:
        df (pd.DataFrame): Transformed dataframe with previous transformations applied
        input_df (pd.DataFrame): Original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with an additional 'source_id' column
    """
    print("Adding source_id column...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Check if the id column exists in the input dataframe
    id_col = 'id'
    
    if id_col in input_df.columns:
        # Copy the id values to the new source_id column
        df['source_id'] = input_df[id_col].astype(str).copy()
        
        # Count non-null source_ids for logging
        non_null_ids = df['source_id'].notna().sum()
        print(f"Added source_id column with {non_null_ids} non-null IDs")
    else:
        print(f"Warning: Column '{id_col}' not found in the input dataframe. 'source_id' will be set to None")
        df['source_id'] = None
    
    return df
