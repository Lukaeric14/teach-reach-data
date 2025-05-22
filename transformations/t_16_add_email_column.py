"""
Transformation 16: Add an Email column based on the email column
"""
import pandas as pd
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'Email' column based on the 'email' column.
    
    Args:
        df (pd.DataFrame): Transformed dataframe with previous transformations applied
        input_df (pd.DataFrame): Original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with an additional 'Email' column
    """
    print("Adding Email column...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Check if the email column exists in the input dataframe
    email_col = 'email'
    
    if email_col in input_df.columns:
        # Copy the email values to the new Email column
        df['Email'] = input_df[email_col].copy()
        
        # Count non-null emails for logging
        non_null_emails = df['Email'].notna().sum()
        print(f"Added Email column with {non_null_emails} non-null email addresses")
    else:
        print(f"Warning: Column '{email_col}' not found in the input dataframe. 'Email' will be set to None")
        df['Email'] = None
    
    return df
