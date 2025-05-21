"""
Transformation to add LinkedIn profile URL for each teacher.
"""
import pandas as pd
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add linkedin_profile_url column to the DataFrame.
    
    Args:
        df: The output DataFrame from previous transformations
        input_df: The original input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added linkedin_profile_url column
    """
    # Add LinkedIn URL from input
    df['linkedin_profile_url'] = input_df['linkedin_url'] if 'linkedin_url' in input_df.columns else ''
    
    # Clean up any missing or invalid values
    df['linkedin_profile_url'] = df['linkedin_profile_url'].fillna('')
    
    # Ensure the URL is a string and strip any whitespace
    df['linkedin_profile_url'] = df['linkedin_profile_url'].astype(str).str.strip()
    
    return df
