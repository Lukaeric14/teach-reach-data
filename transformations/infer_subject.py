import pandas as pd
from typing import Dict, Any
import json
from utils.openai_utils import infer_teacher_subject

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infers the subject each teacher teaches using OpenAI's API and adds it as a new column.
    
    Args:
        df (pd.DataFrame): The current transformed dataframe
        input_df (pd.DataFrame): The original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with inferred subject column added
    """
    # Make a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Initialize the subject column with empty strings
    result_df['subject'] = ''
    
    # Iterate through each teacher and infer their subject
    for idx, row in input_df.iterrows():
        # Convert the row to a dictionary and remove any NaN values
        teacher_data = row.dropna().to_dict()
        
        # Infer the subject using OpenAI
        subject = infer_teacher_subject(teacher_data)
        
        # Add the inferred subject to the result dataframe
        result_df.at[idx, 'subject'] = subject
    
    return result_df
