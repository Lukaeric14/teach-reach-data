"""
Transformation to add preferred grade level for each teacher using AI inference.
"""
import pandas as pd
from typing import Dict, Any
import time
import random
from utils.openai_utils import infer_preferred_grade_level

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add preferred_grade_level column to the DataFrame using AI inference.
    
    Args:
        df: The output DataFrame from previous transformations
        input_df: The original input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with added preferred_grade_level column
    """
    print("\nAnalyzing preferred grade levels...")
    
    # Create a list to store the grade levels
    grade_levels = []
    
    # Process each row to infer the grade level
    for _, row in df.iterrows():
        # Create a dictionary with the relevant teacher information
        teacher_info = {
            'bio': row.get('bio', ''),
            'subject': row.get('subject', ''),
            'headline': row.get('headline', ''),
            'years_of_teaching_experience': row.get('years_of_teaching_experience', 0),
            'nationality': row.get('nationality', '')  # Add nationality for context
        }
        
        # Also include the full row in case we need to access other fields
        teacher_info.update({k: v for k, v in row.items() if k not in teacher_info})
        
        # Get the AI-inferred grade level
        grade_level = infer_preferred_grade_level(teacher_info)
        grade_levels.append(grade_level)
        
        # Print progress
        print(f"Teacher: {row.get('name', 'Unknown')} - Grade Level: {grade_level}")
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.5)
    
    # Add the new column to the DataFrame
    df['preferred_grade_level'] = grade_levels
    
    return df
