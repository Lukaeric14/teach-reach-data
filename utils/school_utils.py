import pandas as pd
import os
import re
from typing import Dict, List, Optional, Tuple

def load_dubai_schools() -> Dict[str, str]:
    """
    Loads the Dubai private schools data and returns a dictionary mapping school names to their curricula.
    
    Returns:
        Dict[str, str]: Dictionary with school names as keys and their curricula as values
    """
    try:
        # Load the CSV file
        file_path = os.path.join(os.path.dirname(__file__), '..', 'DubaiPrivateSchoolsOpenData.csv')
        df = pd.read_csv(file_path)
        
        # Create a dictionary of school names to curricula
        # Convert school names to lowercase for case-insensitive matching
        return {
            str(school_name).lower().strip(): str(curriculum).strip()
            for school_name, curriculum in zip(df['School name'], df['Curriculum'])
            if pd.notna(school_name) and pd.notna(curriculum)
        }
    except Exception as e:
        print(f"Error loading Dubai schools data: {str(e)}")
        return {}

def clean_school_name(name: str) -> str:
    """Clean and standardize school names for better matching."""
    if not name or not isinstance(name, str):
        return ""
    
    # Remove common suffixes and special characters
    name = re.sub(r'[^\w\s]', ' ', name.lower())
    
    # Remove common words that might cause false matches
    common_terms = ['school', 'academy', 'college', 'international', 'private', 'public', 'high', 'elementary', 
                   'primary', 'secondary', 'the', 'and', 'of', 'for', 'in', 'at', 'on', 'a', 'an', 'to']
    words = [word for word in name.split() if word not in common_terms and len(word) > 2]
    
    return ' '.join(words).strip()

def get_curriculum_from_school(text: str, schools_data: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Tries to find a matching school in the provided text using the Dubai schools data.
    
    Args:
        text (str): The text to search for school names
        schools_data (Dict[str, str]): Dictionary of school names to curricula
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (school_name, curriculum) if found, (None, None) otherwise
    """
    if not text or not isinstance(text, str):
        return None, None
        
    # Clean the input text
    text = text.lower().strip()
    
    # First, try to find exact matches or close matches
    for school_name, curriculum in schools_data.items():
        # Clean the school name from our database
        clean_db_name = clean_school_name(school_name)
        clean_text = clean_school_name(text)
        
        # Skip very short names
        if len(clean_db_name) < 3 or len(clean_text) < 3:
            continue
            
        # Check for exact match in cleaned names
        if clean_db_name == clean_text:
            return school_name, curriculum
            
        # Check if either is contained in the other
        if clean_db_name in clean_text or clean_text in clean_db_name:
            return school_name, curriculum
            
        # Split into words and check for partial matches
        db_words = set(clean_db_name.split())
        text_words = set(clean_text.split())
        
        # If we have at least 2 matching words, it's likely a match
        if len(db_words.intersection(text_words)) >= 2:
            return school_name, curriculum
    
    # If no match found, try to find any school name in the text
    for school_name, curriculum in schools_data.items():
        clean_db_name = clean_school_name(school_name)
        
        # Skip very short names
        if len(clean_db_name) < 3:
            continue
            
        # Check if any word from the school name appears in the text
        db_words = set(clean_db_name.split())
        text_words = set(clean_school_name(text).split())
        
        # If we have at least 2 matching words, it's likely a match
        if len(db_words.intersection(text_words)) >= 2:
            return school_name, curriculum
    
    return None, None
