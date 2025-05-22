"""
Transformation 14: Extract current school from employment history
"""
import pandas as pd
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the current school from the first employment history entry.
    
    Args:
        df (pd.DataFrame): Transformed dataframe with previous transformations applied
        input_df (pd.DataFrame): Original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with an additional 'current_school' column
    """
    print("Extracting current school from employment history...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Initialize the new column with None
    df['current_school'] = None
    
    # Check if the employment history column exists in the input dataframe
    employment_col = 'employment_history/0/organization_name'
    
    if employment_col in input_df.columns:
        # Copy the current organization from the first employment history entry
        df['current_school'] = input_df[employment_col].copy()
        
        # Handle any missing values
        df['current_school'] = df['current_school'].fillna('Not specified')
    else:
        print(f"Warning: Column '{employment_col}' not found in the input dataframe. Cannot extract current school.")
        df['current_school'] = 'Not specified'
    
    return df
