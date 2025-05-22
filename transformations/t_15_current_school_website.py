"""
Transformation 15: Extract current school website/domain from employment history
"""
import pandas as pd
from typing import Dict, Any
import re

def extract_domain(url):
    """Extract domain from a URL"""
    if pd.isna(url) or url == 'Not specified':
        return 'Not specified'
    
    # Remove protocol and www.
    domain = url.lower().replace('https://', '').replace('http://', '').replace('www.', '')
    
    # Remove path and query parameters
    domain = domain.split('/')[0]
    
    # Remove port number if present
    domain = domain.split(':')[0]
    
    return domain

def transform(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the current school website/domain from the first employment history entry.
    
    Args:
        df (pd.DataFrame): Transformed dataframe with previous transformations applied
        input_df (pd.DataFrame): Original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with an additional 'current_school_website' column
    """
    print("Extracting current school website/domain from employment history...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Initialize the new column with 'Not specified'
    df['current_school_website'] = 'Not specified'
    
    # Check if the website column exists in the input dataframe
    website_cols = [
        'organization_website_url',  # Primary website URL column in the data
        'organization/website_url',   # Alternative website URL column
    ]
    
    # Try to find a website URL in the available columns
    website_url = None
    
    for col in website_cols:
        if col in input_df.columns and not input_df[col].isna().all():
            # Get the first non-null value
            website_url = input_df[col].dropna().iloc[0] if not input_df[col].dropna().empty else None
            if website_url and website_url != 'Not specified' and pd.notna(website_url):
                print(f"Found website URL in column '{col}': {website_url}")
                df['current_school_website'] = input_df[col].apply(extract_domain)
                return df
    
    # If we get here, no website URL was found in the expected columns
    print("Warning: No valid website URL found in the expected columns. 'current_school_website' will be set to 'Not specified'")
    df['current_school_website'] = 'Not specified'
    
    return df
