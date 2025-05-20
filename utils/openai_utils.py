import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional
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
    The bio should be 1-2 sentences highlighting their experience and expertise.
    
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
