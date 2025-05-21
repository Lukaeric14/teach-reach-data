import pandas as pd
from typing import Dict, Any
from utils.openai_utils import generate_teacher_bio

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an AI-generated bio column to the dataframe.
    The bio is a clean, anonymized summary of the teacher's experience.
    
    Args:
        df (pd.DataFrame): The current transformed dataframe
        input_df (pd.DataFrame): The original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with bio column added
    """
    # Make a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Initialize the bio column
    result_df['bio'] = ''
    
    # Generate bio for each teacher
    for idx, row in input_df.iterrows():
        # Convert the row to a dictionary and remove any NaN values
        teacher_data = {k: v for k, v in row.dropna().items() if v}
        
        # Generate the bio using OpenAI
        bio = generate_teacher_bio(teacher_data)
        
        # Add the bio to the result dataframe
        result_df.at[idx, 'bio'] = bio
    
    return result_df
