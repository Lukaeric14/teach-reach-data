import pandas as pd
from typing import Dict, Any
import time
from utils.openai_utils import extract_teaching_experience

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total years of teaching experience for each teacher using AI.
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Add the new column if it doesn't exist
    if 'years_of_teaching_experience' not in result_df.columns:
        result_df['years_of_teaching_experience'] = 0
    
    # Process each row
    for idx, row in input_df.iterrows():
        try:
            if idx > 0 and idx % 5 == 0:  # Add a small delay every 5 records to avoid rate limiting
                time.sleep(1)
                
            # Convert the row to a dictionary for processing
            teacher_data = row.to_dict()
            
            # Extract experience using AI
            experience_years = extract_teaching_experience(teacher_data)
            
            # Update the result dataframe
            result_df.at[idx, 'years_of_teaching_experience'] = experience_years
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            result_df.at[idx, 'years_of_teaching_experience'] = 0
    
    return result_df
