"""
Transformation to calculate profile completion percentage for teachers.
"""
import pandas as pd
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add profile completion percentage for each teacher.
    
    Args:
        df: The transformed DataFrame with teacher information
        input_df: The original input DataFrame (unused in this transformation)
        
    Returns:
        DataFrame with added profile_completion_percentage column
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # List of required fields to check
    required_fields = [
        'name', 'headline', 'linkedin_profile_url', 'Email', 'subject',
        'bio', 'nationality', 'preferred_grade_level', 'curriculum_experience',
        'teaching_experience_years', 'current_school', 'school_website',
        'current_location_country', 'current_location_city'
    ]
    
    # Initialize profile completion column with 0
    result_df['profile_completion_percentage'] = 0
    
    for idx, row in result_df.iterrows():
        # Start with maximum possible score (50%)
        completion = 50
        
        # Check each required field
        for field in required_fields:
            value = row.get(field, '')
            # Check if field is missing, empty, or contains placeholder values
            if pd.isna(value) or not str(value).strip() or str(value).lower() in ['not specified', 'unknown', '']:
                completion = max(0, completion - 5)  # Subtract 5% for each missing/invalid field
        
        # Ensure the value is between 0 and 50
        result_df.at[idx, 'profile_completion_percentage'] = max(0, min(50, completion))
    
    return result_df
