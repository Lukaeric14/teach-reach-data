"""
Transformation 13: Infer nationality from teacher names
"""
import pandas as pd
from typing import Dict, Any
from utils.openai_utils import infer_nationality_from_name

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infers the most likely nationality for each teacher based on their name.
    
    Args:
        df (pd.DataFrame): Transformed dataframe with previous transformations applied
        input_df (pd.DataFrame): Original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with an additional 'inferred_nationality' column
    """
    print("Starting nationality inference...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Initialize the new column
    df['inferred_nationality'] = "Not specified"
    
    # Get the name column (assuming 'name' exists, adjust if needed)
    name_column = 'name'  # Change this if the name column has a different name
    
    if name_column not in df.columns:
        print(f"Warning: Column '{name_column}' not found in the dataframe. Cannot infer nationalities.")
        return df
    
    # Process each row to infer nationality
    for idx, row in df.iterrows():
        name = row.get(name_column, '')
        if pd.notna(name) and name.strip():
            df.at[idx, 'inferred_nationality'] = infer_nationality_from_name(str(name))
    
    print("Nationality inference completed.")
    return df
