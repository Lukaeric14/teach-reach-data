import os
import sys
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
    - Australian
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
