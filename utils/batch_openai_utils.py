import os
import json
import time
import re
from openai import OpenAI
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import sys
import os

# Add project root to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configurations
from config.openai_config import get_model_config

# Import school curriculum mapping
from utils.school_curriculum_mapping import load_school_curriculum_mapping, get_curriculum_for_school

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load school curriculum mapping
SCHOOL_CURRICULUM_MAPPING = load_school_curriculum_mapping()

def batch_teacher_profile(teacher_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get subject, bio, nationality, and preferred grade level in a single API call.
    
    Args:
        teacher_data (dict): Dictionary containing teacher information
    
    Returns:
        dict: Dictionary with subject, bio, nationality, and preferred_grade_level
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Convert dict to formatted string if needed
    teacher_info = json.dumps(teacher_data, indent=2) if isinstance(teacher_data, dict) else str(teacher_data)
    
    prompt = f"""Based on the following teacher information, provide:
    
    1. Subject: What subject do they most likely teach?
    2. Bio: A professional, anonymized 2-3 sentence bio. Remove all personally identifiable information.
    3. Nationality: Most likely nationality based on their name (use demonym form, e.g., "Egyptian" not "Egypt")
    4. Preferred Grade Level: Elementary, Middle School, High School, or Early Childhood
    5. Is Currently Teaching: TRUE if they are currently a teacher (employed or unemployed), FALSE if they hold another position (coordinator, HR, etc.)
    
    Teacher Information:
    {teacher_info}
    
    Format your response as JSON:
    {{
        "subject": "Subject name",
        "bio": "Professional bio that is anonymized and 2-3 sentences long",
        "nationality": "Nationality",
        "preferred_grade_level": "Grade level",
        "is_currently_teacher": boolean
    }}
    """
    
    try:
        # Get model configuration
        config = get_model_config("teacher_profile")
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are an expert in education who creates structured data about teachers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            response_format=config.get("response_format", None)
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"Error processing teacher profile: {str(e)}")
        return {
            "subject": "Unknown", 
            "bio": "Professional educator with teaching experience.", 
            "nationality": "Not specified",
            "preferred_grade_level": "Not specified",
            "is_currently_teacher": True
        }

def batch_curriculum_and_school(teacher_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get curriculum experience, teaching experience years, current school, 
    and school website in a single API call.
    
    Args:
        teacher_data (dict): Dictionary containing teacher information
    
    Returns:
        dict: Dictionary with curriculum_experience, teaching_experience_years, 
              current_school, and school_website
    """
    # Initialize default result
    result = {
        "curriculum_experience": "Not specified",
        "teaching_experience_years": 0,
        "current_school": "",
        "school_website": ""
    }
    
    # Try to get current school from teacher data
    current_school = ""
    if isinstance(teacher_data, dict):
        current_school = teacher_data.get('current_school', '')
        if not current_school and 'headline' in teacher_data:
            # Try to extract school from headline
            headline = teacher_data['headline']
            if isinstance(headline, str):
                # Look for patterns like "at School Name" or "Teacher at School"
                match = re.search(r'(?:at|@|from|,|\bat\b)\s*([A-Z][A-Za-z0-9\s\-&\']+(?:School|Academy|College|University|Institute|Nursery|Kindergarten|GEMS|SABIS|RAK Academy|Dubai College))', headline, re.IGNORECASE)
                if match:
                    current_school = match.group(1).strip()
    
    # Use the school curriculum mapping if we have a school
    if current_school:
        curriculum = get_curriculum_for_school(current_school, SCHOOL_CURRICULUM_MAPPING)
        if curriculum:
            result["curriculum_experience"] = curriculum
            result["current_school"] = current_school
            return result
    
    # Fall back to OpenAI if we couldn't determine from the mapping
    if not os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key not found. Using default values.")
        return result
    
    # Convert dict to formatted string if needed
    teacher_info = json.dumps(teacher_data, indent=2) if isinstance(teacher_data, dict) else str(teacher_data)
    
    prompt = f"""Based on the following teacher information, provide:
    
    1. Curriculum Experience: What curriculum are they most experienced with? Must be one of:
       - British (for UK/England/Scotland curriculum)
       - American (for US curriculum, including Common Core)
       - IB (for International Baccalaureate)
       - Indian (for CBSE, ICSE, or other Indian boards)
       - UAE (for UAE national curriculum)
       - French (for French curriculum)
       - Australian (for Australian curriculum)
       - Not specified (only if you cannot determine)
    
    2. Teaching Experience Years: Total years of teaching experience (integer)
    
    3. Current School: Their current school (if mentioned)
    
    4. School Website: URL of their current school (if available, otherwise leave empty)
    
    Important Notes:
    - GEMS schools typically follow the British curriculum
    - SABIS schools typically follow the IB curriculum
    - American schools typically follow the American curriculum
    
    Teacher Information:
    {teacher_info}
    
    Format your response as JSON:
    {{
        "curriculum_experience": "Curriculum name from the list above",
        "teaching_experience_years": integer years,
        "current_school": "School name or empty if not mentioned",
        "school_website": "Website URL or empty if not available"
    }}"""
    
    try:
        # Get model configuration
        config = get_model_config("curriculum_school")
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are an expert in international education who extracts structured data about teachers' experience."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            response_format=config.get("response_format", None)
        )
        
        # Parse the JSON response
        api_result = json.loads(response.choices[0].message.content)
        
        # Update our result with API response
        if 'curriculum_experience' in api_result:
            result['curriculum_experience'] = api_result['curriculum_experience']
        if 'teaching_experience_years' in api_result:
            try:
                result['teaching_experience_years'] = int(api_result['teaching_experience_years'])
            except (ValueError, TypeError):
                result['teaching_experience_years'] = 0
        if 'current_school' in api_result and api_result['current_school']:
            result['current_school'] = api_result['current_school']
        if 'school_website' in api_result and api_result['school_website']:
            result['school_website'] = api_result['school_website']
                
    except Exception as e:
        print(f"Error processing curriculum and school with OpenAI: {str(e)}")
    
    return result

def process_teachers_batch(teachers_data: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
    """
    Process a batch of teachers with appropriate rate limiting
    
    Args:
        teachers_data: List of dictionaries containing teacher information
        batch_size: Number of teachers to process in each sub-batch
    
    Returns:
        List of processed teacher data
    """
    results = []
    
    # Process in smaller sub-batches to avoid rate limits
    for i in range(0, len(teachers_data), batch_size):
        sub_batch = teachers_data[i:i+batch_size]
        print(f"  Processing sub-batch {i//batch_size + 1} ({len(sub_batch)} teachers)")
        
        batch_results = []
        for teacher_data in sub_batch:
            # Get the first set of transformations
            profile_data = batch_teacher_profile(teacher_data)
            
            # Wait a moment to avoid rate limiting
            time.sleep(0.5)
            
            # Get the second set of transformations
            curriculum_data = batch_curriculum_and_location(teacher_data)
            
            # Combine results
            combined_data = {**profile_data, **curriculum_data}
            batch_results.append(combined_data)
            
            # Small delay between teachers in the same sub-batch
            time.sleep(0.2)
        
        results.extend(batch_results)
        
        # Add a longer delay between sub-batches
        if i + batch_size < len(teachers_data):
            print("  Waiting between sub-batches...")
            time.sleep(2)
    
    return results
