"""
Transformation to calculate profile completion percentage for teachers.
"""
import json
import pandas as pd
from typing import Dict, Any

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and add profile completion percentage for each teacher.
    
    Args:
        df: The transformed DataFrame with teacher information
        input_df: The original input DataFrame (unused in this transformation)
        
    Returns:
        DataFrame with added profile_completion_percentage and missing_fields columns
    """
    print("Starting profile completion calculation...")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # List of required fields to check with their display names for debugging
    required_fields = [
        ('name', 'Name'),
        ('headline', 'Headline'),
        ('linkedin_profile_url', 'LinkedIn URL'),
        ('Email', 'Email'),
        ('subject', 'Subject'),
        ('bio', 'Bio'),
        ('nationality', 'Nationality'),
        ('preferred_grade_level', 'Preferred Grade Level'),
        ('curriculum_experience', 'Curriculum Experience'),
        ('teaching_experience_years', 'Teaching Experience (Years)'),
        ('current_school', 'Current School'),
        ('school_website', 'School Website'),
        ('current_location_country', 'Country'),
        ('current_location_city', 'City')
    ]
    
    # Field-specific validation rules
    field_validations = {
        'subject': {
            'generic_values': ['education', 'general', 'not specified', 'unknown', '']
        },
        'nationality': {
            'generic_values': ['not specified', 'unknown', '']
        },
        'preferred_grade_level': {
            'valid_values': [
                'Early Childhood (Ages 0-5)',
                'Elementary (Ages 6-10, Grades 1-5)',
                'Middle School (Ages 11-13, Grades 6-8)',
                'High School (Ages 14-18, Grades 9-12)',
                'University/College',
                'Adult Education'
            ]
        },
        'teaching_experience_years': {
            'min': 0,
            'max': 50
        }
    }
    
    # Initialize profile completion column with 0 if it doesn't exist
    if 'profile_completion_percentage' not in result_df.columns:
        result_df['profile_completion_percentage'] = 0
    
    # Initialize missing fields column
    result_df['missing_fields'] = ''
    
    total_teachers = len(result_df)
    print(f"Calculating profile completion for {total_teachers} teachers...")
    
    for idx, row in result_df.iterrows():
        # Start with maximum possible score (50%)
        completion = 50
        missing_fields = []
        
        # Check each required field
        for field, display_name in required_fields:
            value = str(row.get(field, '')).strip()
            field_missing = False
            
            # Check if field is missing or empty
            if pd.isna(value) or not value or value.lower() in ['not specified', 'unknown', 'n/a', 'none']:
                missing_fields.append(f"{field}_missing")
                field_missing = True
            
            # Apply field-specific validations if not missing
            if not field_missing and field in field_validations:
                validation = field_validations[field]
                
                # Check for generic/non-specific values
                if 'generic_values' in validation and value.lower() in validation['generic_values']:
                    missing_fields.append(f"{field}_not_specific")
                
                # Check for valid values in enum fields
                elif 'valid_values' in validation and value not in validation['valid_values']:
                    missing_fields.append(f"{field}_invalid")
                
                # Check numeric ranges
                elif 'min' in validation and 'max' in validation:
                    try:
                        num_value = float(value)
                        if num_value < validation['min'] or num_value > validation['max']:
                            missing_fields.append(f"{field}_out_of_range")
                    except (ValueError, TypeError):
                        missing_fields.append(f"{field}_invalid")
            
            # If any issues were found with this field, deduct points
            if field_missing or f"{field}_" in ' '.join(missing_fields):
                completion = max(0, completion - 5)
        
        # Ensure the value is between 0 and 50
        result_df.at[idx, 'profile_completion_percentage'] = max(0, min(50, completion))
        
        # Store missing fields as a JSON string
        result_df.at[idx, 'missing_fields'] = json.dumps(missing_fields) if missing_fields else ''
        
        # Print debug info for the first few records
        if idx < 3:  # Show details for first 3 records
            print(f"\nTeacher {idx + 1}:")
            print(f"- Completion: {result_df.at[idx, 'profile_completion_percentage']}%")
            if missing_fields:
                print(f"- Missing/Invalid fields: {', '.join(missing_fields)}")
    
    # Print summary
    avg_completion = result_df['profile_completion_percentage'].mean()
    print(f"\nProfile completion calculation complete!")
    print(f"Average completion: {avg_completion:.1f}%")
    
    return result_df
