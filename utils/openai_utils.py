import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional, Union, List
import time

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    prompt = f"""Based on the following teacher information, infer the most likely subject they teach.
    Respond with just the subject name (e.g., "Mathematics", "English Literature"). 
    If uncertain, respond with "Not specified" - only use this if you are really uncertain.
    
    Teacher Information:
    {teacher_data}
    
    Subject:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # Using a valid model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant that determines what subject a teacher teaches based on their information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.3
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
    prompt = f"""Create a concise, professional bio based on the following teacher information.
    The bio should be highlighting their experience and expertise.
    
    IMPORTANT RULES:
    - NEVER mention any names (teacher's name, school names, or organization names)
    - NEVER include any specific location information
    - Keep it professional and focused on their teaching experience
    - Keep it to maximum 3 sentences.
    - Use generic terms instead of specific institution names (e.g., "international schools" instead of school names)
    - If no relevant information is available, return "Professional educator with teaching experience."
    
    Teacher Information:
    {teacher_data}
    
    Bio:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates professional, anonymized teacher bios."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
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
            # Extract all non-empty fields that might contain experience information
            relevant_fields = []
            
            # Check all possible fields that might contain experience
            for field in teacher_data:
                if teacher_data[field] and isinstance(teacher_data[field], (str, int, float)):
                    relevant_fields.append(str(teacher_data[field]))
            
            text = ' '.join(relevant_fields)
        else:
            text = str(teacher_data)
        
        if not text.strip():
            return 0
        
        prompt = f"""Analyze the following text and extract the total years of teaching experience. 
        Consider all teaching roles, even if they're not explicitly labeled as such.
        
        Rules:
        1. Only count years spent in actual teaching roles (classroom teaching, tutoring, etc.)
        2. If multiple time periods are given, sum them up
        3. If a range is given (e.g., 2010-2015), count the number of years
        4. If 'present' or 'current' is mentioned, count up to the current year (2025)
        5. If no clear teaching experience is found, return 0
        
        Return ONLY a single number between 0 and 60 representing the total years of teaching experience.
        
        Example inputs and outputs:
        - "Taught math from 2010 to 2015" -> 5
        - "Teaching since 2018" -> 7
        - "5 years teaching experience" -> 5
        - "No teaching experience" -> 0
        - "Over a decade of teaching" -> 10
        - "Teacher at XYZ School (2015-2020), Professor at ABC University (2020-present)" -> 9
        
        Here's the text to analyze:
        {text}"""
        
        # Print the prompt for debugging
        print("\nAnalyzing teaching experience...")
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
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
    Infers the preferred grade level for a teacher based on their experience and background.
    
    Args:
        teacher_data: Either a dictionary containing teacher information or a string with the bio
        
    Returns:
        str: Inferred grade level (e.g., 'Elementary', 'Middle School', 'High School', 'Early Childhood')
    """
    if not os.getenv("OPENAI_API_KEY"):
        return "Not specified"
    
    # Extract bio from teacher_data if it's a dictionary
    if isinstance(teacher_data, dict):
        bio = teacher_data.get('bio', '')
        subject = teacher_data.get('subject', '')
        headline = teacher_data.get('headline', '')
        experience = str(teacher_data.get('years_of_teaching_experience', ''))
    else:
        bio = str(teacher_data)
        subject = ''
        headline = ''
        experience = ''
    
    prompt = f"""Based on the following teacher information, determine their most likely preferred grade level.
    Choose the most appropriate level from these options:
    - Early Childhood (Ages 0-5)
    - Elementary (Grades 1-5, Ages 6-10)
    - Middle School (Grades 6-8, Ages 11-13)
    - High School (Grades 9-12, Ages 14-18)
    - All Levels (if they have experience across multiple levels)
    
    Consider their subject, experience, and any specific mentions of grade levels.
    
    Teacher Information:
    Subject: {subject}
    Headline: {headline}
    Experience: {experience} years
    Bio: {bio}
    
    Respond with ONLY the grade level from the options above, nothing else."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in education who can determine the most suitable grade level for teachers based on their experience and background."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.2
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
    First tries to match against known Dubai schools, then falls back to AI inference.
    
    Args:
        teacher_data (dict): Dictionary containing teacher information
        
    Returns:
        str: Inferred curriculum (British, American, IB, Indian, UAE, or 'Not specified')
    """
    from utils.school_utils import load_dubai_schools, get_curriculum_from_school
    
    # Only use school matching for specific fields that are likely to contain school names
    school_fields = ['current_school', 'previous_school', 'education', 'experience', 'headline']
    schools_data = load_dubai_schools()
    
    # Check specific fields that are likely to contain school names
    for field in school_fields:
        if field not in teacher_data or not teacher_data[field] or not isinstance(teacher_data[field], str):
            continue
            
        value = teacher_data[field]
        if len(value) < 5:  # Skip very short values
            continue
            
        # Check if this field contains a known school
        matched_school, curriculum = get_curriculum_from_school(value, schools_data)
        if curriculum and matched_school:
            # Only use the match if we're very confident
            clean_value = ' '.join([w for w in value.lower().split() if len(w) > 3])
            clean_school = ' '.join([w for w in matched_school.lower().split() if len(w) > 3])
            
            # If the school name is clearly mentioned in the field
            if clean_school in clean_value or clean_value in clean_school:
                print(f"Confident school match: {matched_school} -> {curriculum}")
                # Map curriculum to our standard format
                curriculum = curriculum.strip()
                if 'british' in curriculum.lower() or 'uk' in curriculum.lower():
                    return "British"
                elif 'american' in curriculum.lower() or 'us' in curriculum.lower() or 'u.s.' in curriculum.lower():
                    return "American"
                elif 'ib' in curriculum.upper() or 'international baccalaureate' in curriculum.lower():
                    return "IB"
                elif 'indian' in curriculum.lower() or 'cbse' in curriculum.lower() or 'icse' in curriculum.lower():
                    return "Indian"
                elif 'uae' in curriculum.upper() or 'u.a.e' in curriculum.upper() or 'ministry of education' in curriculum.lower():
                    return "UAE"
                elif 'french' in curriculum.lower():
                    return "French"
                elif 'australian' in curriculum.lower():
                    return "Australian"
    
    # If no confident school match found, use AI inference
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Prepare the prompt for the AI with more specific instructions
    prompt = f"""Analyze the following teacher information and determine the most likely curriculum they have experience with.
    
    IMPORTANT: Only respond with ONE of the exact options below, nothing else.
    
    CURRICULUM OPTIONS (MUST USE EXACTLY THESE):
    - British (for UK/England/Scotland curriculum)
    - American (for US curriculum, including Common Core)
    - IB (for International Baccalaureate)
    - Indian (for CBSE, ICSE, or other Indian boards)
    - UAE (for UAE national curriculum)
    - French (for French curriculum)
    - Australian (for Australian curriculum)
    - Not specified (only if you cannot determine from the information)
    
    CONSIDER THESE FACTORS:
    1. Nationality and education background
    2. Work experience at specific schools
    3. Subjects and grade levels taught
    4. Any curriculum-specific terminology
    
    If the information is unclear or mixed, choose the most likely single option.
    Only use "Not specified" if there's truly no indication of curriculum experience.
    
    Teacher Information:
    {teacher_data}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert in international education systems. Your task is to determine the most likely curriculum a teacher has experience with based on their background and experience."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=20,
            temperature=0.1  # Keep it low for more deterministic responses
        )
        
        curriculum = response.choices[0].message.content.strip()
        
        # Validate the response is one of our allowed options
        valid_curricula = ["British", "American", "IB", "Indian", "UAE", "French", "Australian", "Not specified"]
        return curriculum if curriculum in valid_curricula else "Not specified"
        
    except Exception as e:
        print(f"Error inferring curriculum: {str(e)}")
        return "Not specified"
    finally:
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
