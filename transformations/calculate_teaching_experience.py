import pandas as pd
from typing import Dict, Any
import time
import re
from datetime import datetime
from typing import Tuple, Optional

# Import the AI-powered experience extraction function
from utils.openai_utils import extract_teaching_experience

def parse_date_range(date_str: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse a date range string into start and end dates.
    Handles various formats like 'Jan 2010 - Present', '2015 - 2020', etc.
    """
    if not date_str or not isinstance(date_str, str):
        return None, None
        
    date_str = date_str.strip()
    current_year = datetime.now().year
    
    # Common date formats to try
    date_formats = [
        '%b %Y',  # Jan 2020
        '%B %Y',  # January 2020
        '%Y',     # 2020
        '%m/%Y',  # 01/2020
        '%Y-%m',  # 2020-01
    ]
    
    # Split into start and end dates
    parts = re.split(r'\s*-\s*|\s*to\s*|\s*–\s*', date_str, 1)
    if len(parts) != 2:
        return None, None
        
    start_part, end_part = parts
    
    def parse_date(date_str: str) -> Optional[datetime]:
        date_str = date_str.strip().lower()
        
        # Handle "Present" or "Current"
        if date_str in ['present', 'current', 'now']:
            return datetime.now()
            
        # Try different date formats
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        # Try to extract year if other formats fail
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            try:
                return datetime(int(year_match.group(0)), 1, 1)
            except (ValueError, IndexError):
                pass
                
        return None
    
    start_date = parse_date(start_part)
    end_date = parse_date(end_part)
    
    return start_date, end_date

def extract_experience_years(experience_text: str) -> float:
    """
    Extract total years of experience from experience text.
    Improved to handle more patterns and extract from bio text.
    """
    if not experience_text or not isinstance(experience_text, str):
        return 0.0
    
    # Convert to lowercase once
    text = experience_text.lower()
    
    # Look for experience patterns in the text
    patterns = [
        # Patterns like "5 years", "10+ years", "1.5 years"
        r'(?:\b(?:over|more than|about|approximately|~)?\s*)?(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?\b)',
        # Patterns like "5+ years of experience"
        r'(\d+)\s*\+\s*years?\s*(?:of)?\s*(?:experience|exp|teaching)',
        # Patterns like "a decade" or "decades"
        r'(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+decades?',
        # Patterns like "since 2010" or "from 2010 to 2015"
        r'(?:since|from|in)\s+(?:the year\s+)?(19\d{2}|20[0-2]\d)(?:\s*(?:to|until|-)\s*(?:present|now|current|20[0-2]\d|19\d{2}))?',
    ]
    
    # Check for explicit experience mentions first
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                # Handle word numbers for decades
                word_to_num = {
                    'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                
                # Get the matched group
                value = match.group(1) if len(match.groups()) >= 1 else None
                if not value:
                    continue
                    
                # Convert word numbers to digits
                if value in word_to_num:
                    years = word_to_num[value]
                    # If it's a decade, multiply by 10
                    if 'decade' in match.group(0):
                        years *= 10
                    return float(years)
                
                # Handle year ranges (e.g., since 2010)
                if value.isdigit() and len(value) == 4 and 1900 <= int(value) <= 2100:
                    start_year = int(value)
                    current_year = datetime.now().year
                    return float(current_year - start_year)
                
                # Regular number of years
                years = float(value)
                return years
                
            except (ValueError, IndexError):
                continue
    
    # If no explicit mention, look for experience indicators in bio
    experience_indicators = [
        r'(?:over|more than|about|approximately|~)?\s*\d+\s*(?:\+)?\s*(?:years?|yrs?\b)',
        r'\d+\+\s*(?:years?|yrs?\b)',
        r'(?:nearly|almost|over|more than|about|approximately|~)?\s*\d+\s*(?:years?|yrs?\b)',
        r'since\s+\d{4}',
        r'from\s+\d{4}\s+to\s+\d{4}',
        r'\d+\s*[-–]\s*\d+\s*(?:years?|yrs?\b)',
    ]
    
    for pattern in experience_indicators:
        if re.search(pattern, text):
            # Extract all numbers that might be years of experience
            numbers = re.findall(r'\b\d+\b', text)
            if numbers:
                try:
                    # Take the largest number that makes sense as years of experience
                    max_years = max(int(n) for n in numbers if 1 <= int(n) <= 60)
                    return float(max_years)
                except (ValueError, TypeError):
                    continue
    
    return 0.0

def calculate_experience(experience_items: list) -> float:
    """
    Calculate total years of experience from a list of experience items.
    Each item can be a string or a dict with 'date_range' or 'duration' fields.
    """
    total_years = 0.0
    
    for item in experience_items:
        if not item:
            continue
            
        # Handle string items (assume it's a job description)
        if isinstance(item, str):
            # First try to extract years directly from text
            years = extract_experience_years(item)
            if years > 0:
                total_years += years
                continue
                
            # Then try to parse date ranges
            start_date, end_date = parse_date_range(item)
            if start_date and end_date:
                delta = end_date - start_date
                total_years += delta.days / 365.25
                
        # Handle dict items (structured data)
        elif isinstance(item, dict):
            # Try to get date range
            if 'date_range' in item:
                start_date, end_date = parse_date_range(item['date_range'])
                if start_date and end_date:
                    delta = end_date - start_date
                    total_years += delta.days / 365.25
                    continue
                    
            # Try to get duration
            if 'duration' in item and isinstance(item['duration'], str):
                years = extract_experience_years(item['duration'])
                if years > 0:
                    total_years += years
    
    return round(total_years, 1)  # Round to 1 decimal place

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
            # Convert the row to a dictionary for processing
            teacher_data = row.to_dict()
            
            # Extract experience using AI
            experience_years = extract_teacher_experience(teacher_data)
            
            # Update the result dataframe
            result_df.at[idx, 'years_of_teaching_experience'] = experience_years
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            result_df.at[idx, 'years_of_teaching_experience'] = 0
    
    return result_df
