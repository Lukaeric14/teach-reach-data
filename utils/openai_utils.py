import os
import sys
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, List
import time

# Add project root to path to allow absolute imports
sys.path.append(str(Path(__file__).parent.parent))

# Import configurations
from config.openai_config import get_model_config

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def enrich_teacher_profile(teacher_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive function to enrich a teacher profile with all required fields using a single API call.
    
    Args:
        teacher_data (dict): Dictionary containing teacher information
    
    Returns:
        dict: Dictionary with all enriched fields including:
              - subject: The main subject taught
              - bio: Professional anonymized bio
              - nationality: Inferred nationality
              - preferred_grade_level: Preferred teaching grade level
              - is_currently_teacher: Whether they are currently a teacher
              - curriculum_experience: Curriculum experience
              - teaching_experience_years: Years of teaching experience
              - current_school: Current or most recent school
              - school_website: School website if available
              - current_location_country: Current country location
              - current_location_city: Current city location
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Convert dict to formatted string if needed
    teacher_info = json.dumps(teacher_data, indent=2) if isinstance(teacher_data, dict) else str(teacher_data)
    
    # Extract employment history for better analysis
    employment_history = []
    i = 0
    while isinstance(teacher_data, dict) and True:
        org_key = f'employment_history/{i}/organization_name'
        if org_key not in teacher_data:
            break
            
        # Get employment entry
        entry = {
            'organization': str(teacher_data.get(org_key, '')).strip(),
            'title': str(teacher_data.get(f'employment_history/{i}/title', '')).strip(),
            'current': bool(teacher_data.get(f'employment_history/{i}/current', False)),
            'start_date': str(teacher_data.get(f'employment_history/{i}/start_date', '')).strip(),
            'end_date': str(teacher_data.get(f'employment_history/{i}/end_date', '')).strip()
        }
        
        # Only add if we have an organization name
        if entry['organization'] and entry['organization'].lower() not in ['none', 'n/a', 'not specified']:
            employment_history.append(entry)
        
        i += 1
        
    # Sort by current status (current first) and then by start_date (most recent first)
    if employment_history:
        employment_history.sort(
            key=lambda x: (
                not x['current'],  # Current jobs first
                x['start_date'] if x['start_date'] else '1900-01-01'  # Then by start date
            ),
            reverse=True  # Most recent first
        )
        
    # Add employment history to the prompt for better context
    employment_summary = "\n\nEmployment History:\n"
    for job in employment_history[:5]:  # Limit to top 5 jobs
        employment_summary += f"- {job['organization']}: {job['title']} ({'Current' if job['current'] else job['start_date'] + ' to ' + (job['end_date'] if job['end_date'] else 'Present')})\n"
    
    prompt = f"""Based on the following comprehensive teacher information, provide a detailed profile enrichment with ALL of the following fields.
    
    Teacher Information:
    {teacher_info}
    {employment_summary if employment_history else ''}
    
    NOTE: All of these fields are REQUIRED and must be provided with high accuracy!
    
    1. Subject: Be very specific about the subject they teach. For example:
       - Instead of "English", use "English Literature" or "English as a Second Language (ESL)"
       - Instead of "Math", use "Mathematics", "Calculus", or "Statistics"
       - Instead of "Science", use "Physics", "Chemistry", "Biology", etc.
       - For primary/elementary teachers that don't have a specific subject, use "Primary Education" or "Elementary Education"
       - Only use "Education" as a last resort if when no specific subject can be determined.
    
    2. Bio: A professional, anonymized 2-3 sentence bio. Remove all personally identifiable information.
    
    3. Nationality: Most likely nationality based on their name, work history, and location (use demonym form, e.g., "Egyptian" not "Egypt"). If uncertain, provide best estimate.
    
    4. Preferred Grade Level: Choose one of these exact values:
       - Early Childhood (Ages 0-5)
       - Elementary (Ages 6-10, Grades 1-5)
       - Middle School (Ages 11-13, Grades 6-8)
       - High School (Ages 14-18, Grades 9-12)
       - University/College
       - Adult Education
    
    5. Is Currently Teaching: 
       - Set to TRUE ONLY if their current/most recent role is a teaching position (e.g., Teacher, Instructor, Professor, Lecturer, etc.)
       - Set to FALSE if they are in non-teaching roles like: Administrator, Principal, Director, Counselor, Coordinator, HR, Recruiter, etc.
       - If uncertain, default to FALSE
       
    6. Curriculum Experience: Based on their schools, nationality and experience, choose from:
       - British
       - American
       - IB (International Baccalaureate)
       - Indian
       - UAE
       - Australian
       - Cambridge
       - French
       - Not specified (only if truly cannot determine)
       
    7. Teaching Experience Years: Estimate the total years of teaching experience based on their history. 
       Provide a numeric value. If uncertain, estimate based on career length.
       
    8. Current School: Name of their current or most recent school/educational institution.
       
    9. School Website: The website of their current school if available. 
       If not available in the data, respond with an empty string.
       
    10. Current Location Country: The country where they currently work or live.
        
    11. Current Location City: The city where they currently work or live.
    
    Format your response as JSON with all fields included:
    {{
        "subject": "...",
        "bio": "...",
        "nationality": "...",
        "preferred_grade_level": "...",
        "is_currently_teacher": boolean,
        "curriculum_experience": "...",
        "teaching_experience_years": number,
        "current_school": "...",
        "school_website": "...",
        "current_location_country": "...",
        "current_location_city": "..."
    }}
    """
    
    try:
        # Get model configuration
        config = get_model_config("teacher_profile")
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are an expert in education who creates comprehensive structured data about teachers. Extract and infer all required information accurately based on the given data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,  # Increased for comprehensive response
            temperature=0.2,  # Lower temperature for more consistent results
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Apply validations and fixes
        result = validate_teacher_profile(teacher_data, result)
        
        return result
        
    except Exception as e:
        print(f"Error enriching teacher profile: {str(e)}")
        # Return default values on error
        return {
            "subject": "Unknown", 
            "bio": "Professional educator with teaching experience.", 
            "nationality": "Unknown",
            "preferred_grade_level": "Not specified",
            "is_currently_teacher": False,
            "curriculum_experience": "Not specified",
            "teaching_experience_years": 0,
            "current_school": "",
            "school_website": "",
            "current_location_country": "",
            "current_location_city": ""
        }


def validate_teacher_profile(teacher_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix the teacher profile data returned from the API.
    
    Args:
        teacher_data: Original teacher data
        result: The enriched profile from the API
        
    Returns:
        Dict with validated and fixed data
    """
    # Ensure all fields exist
    required_fields = [
        "subject", "bio", "nationality", "preferred_grade_level", 
        "is_currently_teacher", "curriculum_experience", "teaching_experience_years",
        "current_school", "school_website", "current_location_country", "current_location_city"
    ]
    
    for field in required_fields:
        if field not in result or result[field] is None:
            if field == "teaching_experience_years":
                result[field] = 0
            elif field == "is_currently_teacher":
                result[field] = False
            else:
                result[field] = ""
                
    # Convert teaching_experience_years to float
    try:
        result["teaching_experience_years"] = float(result["teaching_experience_years"])
    except (ValueError, TypeError):
        result["teaching_experience_years"] = 0.0
        
    # Convert is_currently_teacher to bool if it's not already
    if isinstance(result["is_currently_teacher"], str):
        result["is_currently_teacher"] = result["is_currently_teacher"].lower() in ["true", "yes", "1"]
        
    # Fix nationality if it's missing or unknown
    if not result["nationality"] or result["nationality"].lower() in ["unknown", "not specified", "unspecified"]:
        name = teacher_data.get("name", "")
        if name:
            try:
                result["nationality"] = infer_nationality_from_name(name)
            except Exception as e:
                print(f"Error inferring nationality: {str(e)}")
                
    # Fix school website if provided
    if result["school_website"] and not result["school_website"].startswith(('http://', 'https://')):
        result["school_website"] = 'https://' + result["school_website"]
                
    return result

def infer_teacher_subject(teacher_data: Dict[str, Any]) -> str:
    """
    Infers the subject a teacher teaches based on their information using OpenAI's API.
    
    Args:
        teacher_data (dict): Dictionary containing teacher information
        
    Returns:
        str: Inferred subject or "Unknown" if inference fails
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Prepare the prompt for the AI
    prompt = f"""Based on the following teacher information, what subject do they most likely teach?
    
    Teacher Information:
    {teacher_data}
    
    Subject:"""
    
    try:
        # Get model configuration
        config = get_model_config("teacher_subject")
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant that determines what subject a teacher teaches based on their information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        
        subject = response.choices[0].message.content.strip()
        return subject if subject else "Unknown"
        
    except Exception as e:
        print(f"Error inferring subject: {str(e)}")
        return "Unknown"
    finally:
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

def generate_teacher_bio(teacher_data: Dict[str, Any]) -> str:
    """
    Generates a clean, anonymized bio for a teacher based on their information.
    
    Args:
        teacher_data (dict): Dictionary containing teacher information
        
    Returns:
        str: Generated bio with all identifiable information removed
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Prepare the prompt for the AI
    prompt = f"""Create a professional, anonymized bio for a teacher based on the following information.
    
    Instructions:
    1. Remove all personally identifiable information (names, specific schools, locations, etc.)
    2. Focus on their teaching experience, subjects, and educational background
    3. Keep it professional and concise (2-3 sentences)
    4. Use generic terms (e.g., "international school" instead of school names)
    5. Do not include any specific years or durations
    
    Teacher Information:
    {teacher_data}
    
    Bio:"""
    
    try:
        # Get model configuration
        config = get_model_config("teacher_bio")
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates professional, anonymized teacher bios. Remove all personally identifiable information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        
        bio = response.choices[0].message.content.strip()
        return bio if bio else "Professional educator with teaching experience."
        
    except Exception as e:
        print(f"Error generating bio: {str(e)}")
        return "Professional educator with teaching experience."
    finally:
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)

def extract_teaching_experience(teacher_data: Union[Dict[str, Any], str]) -> int:
    """
    Extracts the total years of teaching experience using AI.
    
    Args:
        teacher_data: Either a dictionary containing teacher information or a string with experience text
        
    Returns:
        int: Total years of teaching experience, or 0 if not found
    """
    try:
        # If input is a dictionary, convert relevant fields to a string
        if isinstance(teacher_data, dict):
            # Extract relevant fields that might contain experience information
            relevant_fields = [
                'experience', 'work_experience', 'employment_history',
                'teaching_experience', 'background', 'summary', 'headline', 'bio'
            ]
            
            # Combine all relevant fields into a single string
            text_parts = []
            for field in relevant_fields:
                if field in teacher_data and teacher_data[field]:
                    text_parts.append(str(teacher_data[field]))
            
            if not text_parts:
                return 0
                
            text = " ".join(text_parts)
        else:
            text = str(teacher_data)
        
        if not text.strip():
            return 0
            
        # Prepare the prompt for the AI
        prompt = f"""Analyze the following text and extract the total years of teaching experience.
        Return ONLY a single number representing the total years of teaching experience.
        
        Instructions:
        1. Look for phrases indicating teaching experience (e.g., "X years teaching", "taught for X years")
        2. Sum up all teaching experience if mentioned in multiple places
        3. Return 0 if no teaching experience is mentioned
        4. Only return a number, no text or explanations
        
        Examples:
        - "5 years teaching experience" -> 5
        - "No teaching experience" -> 0
        - "Over a decade of teaching" -> 10
        - "Teacher at XYZ School (2015-2020), Professor at ABC University (2020-present)" -> 9
        
        Here's the text to analyze:
        {text}"""
        
        # Print the prompt for debugging
        print("\nAnalyzing teaching experience...")
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing teaching experience and extracting the total years of experience. You must return only a single number between 0 and 60."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        # Extract the first number from the response
        import re
        response_text = response.choices[0].message.content.strip()
        print(f"AI Response: {response_text}")
        
        # Look for the first number in the response
        match = re.search(r'\d+', response_text)
        if match:
            years = int(match.group(0))
            # Ensure the result is within a reasonable range
            return min(max(years, 0), 60)
            
        return 0
        
    except Exception as e:
        print(f"Error extracting teaching experience: {e}")
        return 0

def infer_preferred_grade_level(teacher_data: Union[Dict[str, Any], str]) -> str:
    """
    Infers the preferred grade level for a teacher based on their experience, background,
    school history, and nationality.
    
    Args:
        teacher_data: Either a dictionary containing teacher information or a string with the bio
        
    Returns:
        str: Inferred grade level (e.g., 'Elementary', 'Middle School', 'High School', 'Early Childhood')
    """
    if not os.getenv("OPENAI_API_KEY"):
        return "Not specified"
    
    # If input is a dictionary, extract relevant information
    if isinstance(teacher_data, dict):
        # Get relevant fields
        bio = teacher_data.get('bio', '')
        experience = teacher_data.get('experience', '')
        subject = teacher_data.get('subject', '')
        education = teacher_data.get('education', '')
        
        # Combine relevant information
        text = f"""Bio: {bio}
        Experience: {experience}
        Subject: {subject}
        Education: {education}"""
    else:
        text = str(teacher_data)
    
    # Get nationality for additional context if available
    nationality = ""
    if isinstance(teacher_data, dict) and 'nationality' in teacher_data:
        nationality = f"Nationality: {teacher_data['nationality']}"
    
    nationality_context = f"""
    Consider the following nationality information which might influence educational systems:
    {nationality}
    """ if nationality else ""
    
    # Prepare the prompt for the AI
    prompt = f"""Based on the following teacher information, determine the most suitable grade level they would prefer to teach.
    
    GRADE LEVEL OPTIONS (MUST CHOOSE ONE):
    - Early Childhood (Pre-K to Kindergarten, ages 3-5)
    - Elementary (Grades 1-5, ages 6-10)
    - Middle School (Grades 6-8, ages 11-13)
    - High School (Grades 9-12, ages 14-18)
    - All Levels (if they have experience across multiple levels)
    
    CONSIDER THESE FACTORS:
    1. Teaching experience and subjects taught
    2. Educational background and qualifications
    3. Any specific age groups mentioned
    4. Cultural or educational system preferences
    {nationality_context}
    
    Respond with ONLY the grade level from the options above, nothing else."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are an expert in international education who can determine the most suitable grade level for teachers based on their experience, subject matter, school curriculum, and educational background. You understand different educational systems worldwide."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.1  # Lower temperature for more consistent results
        )
        
        grade_level = response.choices[0].message.content.strip()
        
        # Validate the response matches one of our expected values
        valid_levels = ["Early Childhood", "Elementary", "Middle School", "High School", "All Levels"]
        if grade_level not in valid_levels:
            return "Not specified"
            
        return grade_level
        
    except Exception as e:
        print(f"Error inferring grade level: {str(e)}")
        return "Not specified"

def infer_curriculum_experience(teacher_data: Dict[str, Any]) -> str:
    """
    Infers the most likely curriculum experience based on teacher information.
    Uses nationality and school information to determine the most likely curriculum.
    
    Args:
        teacher_data (dict): Dictionary containing teacher information
        
    Returns:
        str: Inferred curriculum (British, American, IB, Indian, UAE, or 'Not specified')
    """
    print("\n=== Starting curriculum inference ===")
    print(f"Teacher data keys: {list(teacher_data.keys())}")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return "Not specified"
    
    # Get relevant information with debug output
    nationality = str(teacher_data.get('nationality', '')).strip()
    current_school = str(teacher_data.get('current_school', '')).strip()
    experience = str(teacher_data.get('experience', '')).strip()
    education = str(teacher_data.get('education', '')).strip()
    
    print(f"Nationality: {nationality}")
    print(f"Current School: {current_school}")
    print(f"Experience: {experience[:100]}..." if len(experience) > 100 else f"Experience: {experience}")
    print(f"Education: {education[:100]}..." if len(education) > 100 else f"Education: {education}")
    
    # Prepare the prompt
    prompt = f"""Based on the teacher's information below, determine the most likely curriculum they have experience with.
    
    Teacher Information:
    - Nationality: {nationality}
    - Current School: {current_school}
    - Experience: {experience}
    - Education: {education}
    
    CURRICULUM OPTIONS (respond with ONLY one of these):
    - British
    - American
    - IB
    - Indian
    - UAE
    - French
    - Not specified
    
    Respond with ONLY the curriculum name from the options above:"""
    
    try:
        print("\nSending request to OpenAI...")
        # Get model configuration
        config = get_model_config("curriculum")
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are an expert in international education systems. Analyze the teacher's nationality and school information to determine the most likely curriculum they have experience with. Respond with ONLY the curriculum name from the provided options."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        
        curriculum = response.choices[0].message.content.strip()
        valid_curricula = ["British", "American", "IB", "Indian", "UAE", "French", "Australian", "Not specified"]
        
        print(f"Raw response: {curriculum}")
        
        # Clean up the response
        curriculum = curriculum.strip('.').strip()
        
        if curriculum not in valid_curricula:
            print(f"Warning: Invalid curriculum '{curriculum}' received. Defaulting to 'Not specified'")
            return "Not specified"
            
        print(f"Inferred curriculum: {curriculum}")
        return curriculum
        
    except Exception as e:
        print(f"Error inferring curriculum: {str(e)}")
        return "Not specified"
    finally:
        time.sleep(0.5)
        print("=== End of curriculum inference ===\n")

def infer_nationality_from_name(name: str) -> str:
    """
    Infers the most likely nationality based on a person's name using AI.
    
    Args:
        name (str): The full name of the person
        
    Returns:
        str: Inferred nationality (country name) or 'Not specified' if uncertain
    """
    if not os.getenv("OPENAI_API_KEY"):
        return "Not specified"
    
    if not name or not isinstance(name, str) or len(name.strip()) < 2:
        return "Not specified"
    
    prompt = f"""Based on the following name, infer the most likely nationality (country of origin).
    Consider common naming patterns, surnames, and given names associated with different cultures.
    
    RULES:
    1. Respond with ONLY the country name in English (e.g., "Egyptian", "Indian", "British")
    2. If uncertain, respond with "Not specified"
    3. Use demonyms (e.g., "Egyptian" not "Egypt", "American" not "United States")
    4. NOTE: Emirati nationality is quite rare, most arab names are from Egypt, Lebanon, Palestine, Jordan.
    5. If the name is arab but you cannt infer a specific country, respond with fallback "Middle Eastern" (this should be rare). 
    
    Name: {name}
    
    Most likely nationality:"""
    
    try:
        # Get model configuration
        config = get_model_config("nationality")
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are an expert in onomastics and cultural naming conventions. Analyze the name and provide the most likely nationality."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        
        nationality = response.choices[0].message.content.strip()
        
        # Basic validation of the response
        if not nationality or nationality.lower() == 'not specified' or len(nationality) > 30:
            return "Not specified"
            
        return nationality
        
    except Exception as e:
        print(f"Error inferring nationality: {str(e)}")
        return "Not specified"
    finally:
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
