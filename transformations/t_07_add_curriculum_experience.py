import pandas as pd
from typing import Dict, Any
from utils.openai_utils import infer_curriculum_experience

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a column with the inferred curriculum experience for each teacher.
    
    Args:
        df (pd.DataFrame): The current transformed dataframe
        input_df (pd.DataFrame): The original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with curriculum_experience column added
    """
    # Make a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Initialize the curriculum_experience column
    result_df['curriculum_experience'] = ''
    
    # Generate curriculum experience for each teacher
    for idx, row in input_df.iterrows():
        # Convert the row to a dictionary and remove any NaN values
        teacher_data = {k: v for k, v in row.dropna().items() if v}
        
        try:
            # Infer the curriculum experience
            curriculum = infer_curriculum_experience(teacher_data)
            result_df.at[idx, 'curriculum_experience'] = curriculum
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            result_df.at[idx, 'curriculum_experience'] = 'Not specified'
    
    return result_df
