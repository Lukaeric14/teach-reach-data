import pandas as pd
import os
from typing import Dict, Optional

def load_school_curriculum_mapping() -> Dict[str, str]:
    """
    Load the mapping of school names to their curriculum from the DubaiPrivateSchoolsOpenData.csv file.
    
    Returns:
        Dict[str, str]: Dictionary mapping school names to their curriculum
    """
    try:
        # Load the CSV file
        df = pd.read_csv('DubaiPrivateSchoolsOpenData.csv')
        
        # Create a mapping of school names to curriculum
        # Convert both keys and values to lowercase for case-insensitive matching
        mapping = {}
        for _, row in df.iterrows():
            school_name = row['﻿School name'].strip().lower()
            curriculum = row['Curriculum'].strip()
            
            # Special case for GEMS schools
            if 'gems' in school_name and 'british' not in school_name.lower():
                curriculum = 'British'
            # Special case for SABIS schools
            elif 'sabis' in school_name.lower():
                curriculum = 'IB'
                
            mapping[school_name] = curriculum
            
            # Also add variations of the school name for better matching
            if 'school' in school_name:
                mapping[school_name.replace(' school', '').strip()] = curriculum
            if 'academy' in school_name:
                mapping[school_name.replace(' academy', '').strip()] = curriculum
                
        return mapping
        
    except Exception as e:
        print(f"Error loading school curriculum mapping: {e}")
        return {}

def get_curriculum_for_school(school_name: str, mapping: Dict[str, str]) -> Optional[str]:
    """
    Get the curriculum for a given school name using the provided mapping.
    
    Args:
        school_name (str): The name of the school
        mapping (Dict[str, str]): The school to curriculum mapping
        
    Returns:
        Optional[str]: The curriculum if found, None otherwise
    """
    if not school_name or not isinstance(school_name, str):
        return None
        
    # Clean the school name
    school_name = school_name.lower().strip()
    
    # Check for exact match first
    if school_name in mapping:
        return mapping[school_name]
    
    # Check for partial matches
    for key, value in mapping.items():
        if key in school_name or school_name in key:
            return value
    
    # Special cases
    if 'gems' in school_name:
        return 'British'
    if 'sabis' in school_name:
        return 'IB'
    if 'raffles' in school_name:
        return 'IB'
    if 'american school' in school_name:
        return 'American'
    if 'british school' in school_name:
        return 'British'
    
    return None
