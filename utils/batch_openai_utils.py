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

def validate_teacher_status(teacher_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and corrects the is_currently_teacher flag based on role keywords.
    Also ensures the subject is specific and not too generic.
    
    Args:
        teacher_data: Original teacher data
        result: Current result with is_currently_teacher flag and subject
        
    Returns:
        Updated result with validated is_currently_teacher flag and subject
    """
    # Start with the current values from result
    is_teacher = result.get('is_currently_teacher', False)  # Default to False to be more strict
    current_subject = result.get('subject', '')
    
    # Get all relevant role information
    current_role = ''
    if isinstance(teacher_data, dict):
        current_role = ' '.join([
            str(teacher_data.get('headline', '')),
            str(teacher_data.get('current_position', '')),
            str(teacher_data.get('title', ''))
        ]).lower()
    
    # More specific teaching indicators
    teaching_indicators = [
        # Teaching roles
        'teacher', 'instructor', 'professor', 'lecturer', 'educator', 'faculty',
        'tutor', 'teacher assistant', 'teaching assistant', 'ta', 'adjunct',
        'education specialist', 'learning specialist', 'classroom teacher',
        'subject teacher', 'subject specialist', 'subject lead',
        
        # Department heads and leadership
        'head of', 'head teacher', 'head of year', 'head of department',
        'head of school', 'headmaster', 'headmistress', 'head of primary',
        'head of secondary', 'head of key stage', 'head of ks', 'head of ks1',
        'head of ks2', 'head of ks3', 'head of ks4', 'head of ks5',
        'department chair', 'department head', 'curriculum lead',
        'academic lead', 'academic coordinator', 'education coordinator',
        
        # Subject-specific indicators
        'math teacher', 'science teacher', 'english teacher', 'history teacher',
        'physics teacher', 'chemistry teacher', 'biology teacher',
        'computer science teacher', 'art teacher', 'music teacher',
        'pe teacher', 'physical education teacher', 'language teacher',
        'spanish teacher', 'french teacher', 'german teacher', 'arabic teacher',
        'chinese teacher', 'japanese teacher', 'esl teacher', 'special ed teacher',
        'special education teacher', 'gifted teacher', 'elementary teacher',
        'primary teacher', 'secondary teacher', 'high school teacher',
        'middle school teacher', 'early years teacher', 'kindergarten teacher',
        'preschool teacher', 'nursery teacher'
    ]
    
    # More comprehensive non-teaching roles
    non_teaching_roles = [
        # Administrative roles
        'administrator', 'principal', 'vice principal', 'director', 'head',
        'counselor', 'coordinator', 'manager', 'supervisor', 'superintendent',
        'headmaster', 'headmistress', 'head of school', 'head of department',
        'head of year', 'head of house', 'dean', 'provost', 'chancellor',
        'registrar', 'bursar', 'business manager', 'finance manager',
        'human resources', 'hr', 'recruiter', 'talent acquisition',
        'admissions officer', 'admissions director', 'admissions coordinator',
        'development director', 'fundraising', 'alumni relations',
        'communications', 'marketing', 'public relations', 'pr',
        'it support', 'systems administrator', 'network administrator',
        'librarian', 'media specialist', 'technology specialist',
        'curriculum developer', 'instructional designer', 'education consultant',
        'researcher', 'research assistant', 'research fellow',
        'teaching fellow', 'graduate assistant', 'teaching associate',
        'teaching fellow', 'adjunct professor', 'adjunct faculty',
        'visiting professor', 'visiting lecturer', 'visiting scholar',
        'postdoctoral fellow', 'postdoc', 'post-doc', 'post doc',
        'research scientist', 'scientist', 'engineer', 'analyst',
        'data analyst', 'data scientist', 'statistician', 'economist',
        'psychologist', 'social worker', 'therapist', 'counselor',
        'nurse', 'doctor', 'physician', 'physician assistant',
        'nurse practitioner', 'physical therapist', 'occupational therapist',
        'speech therapist', 'speech pathologist', 'audiologist',
        'dietitian', 'nutritionist', 'librarian', 'archivist',
        'curator', 'conservator', 'registrar', 'archaeologist',
        'anthropologist', 'sociologist', 'political scientist',
        'economist', 'historian', 'geographer', 'demographer',
        'statistician', 'mathematician', 'physicist', 'chemist',
        'biologist', 'geologist', 'meteorologist', 'astronomer',
        'oceanographer', 'environmental scientist', 'environmental specialist',
        'environmental engineer', 'civil engineer', 'mechanical engineer',
        'electrical engineer', 'computer engineer', 'software engineer',
        'computer programmer', 'web developer', 'web designer',
        'graphic designer', 'artist', 'musician', 'performer',
        'actor', 'actress', 'dancer', 'choreographer', 'producer',
        'director', 'editor', 'writer', 'author', 'journalist',
        'reporter', 'correspondent', 'announcer', 'broadcaster',
        'public relations specialist', 'publicist', 'advertising',
        'marketing specialist', 'market research analyst', 'sales',
        'retail sales', 'wholesale sales', 'insurance sales',
        'real estate broker', 'real estate agent', 'financial advisor',
        'investment advisor', 'accountant', 'auditor', 'bookkeeper',
        'tax preparer', 'budget analyst', 'financial analyst',
        'personal financial advisor', 'loan officer', 'credit analyst',
        'insurance underwriter', 'actuary', 'appraiser', 'assessor',
        'claims adjuster', 'claims appraiser', 'investigator',
        'compliance officer', 'cost estimator', 'human resources',
        'training and development', 'labor relations', 'management analyst',
        'meeting planner', 'fundraiser', 'compensation', 'benefits',
        'job analysis', 'training', 'development', 'logistician',
        'purchasing manager', 'purchasing agent', 'buyer', 'wholesale',
        'retail buyer', 'procurement', 'supply chain', 'traffic technician',
        'dispatcher', 'power plant operator', 'power distributor',
        'power dispatcher', 'nuclear technician', 'nuclear engineer',
        'nuclear power reactor operator', 'power plant operator',
        'power distributor', 'power dispatcher', 'nuclear technician',
        'nuclear engineer', 'nuclear power reactor operator',
        'power plant operator', 'power distributor', 'power dispatcher',
        'nuclear technician', 'nuclear engineer', 'nuclear power reactor operator'
    ]
    
    # Check for teaching indicators first (case insensitive)
    current_role_lower = current_role.lower()
    has_teaching_indicator = any(indicator.lower() in current_role_lower for indicator in teaching_indicators)
    
    # Check for non-teaching indicators (case insensitive)
    has_non_teaching_indicator = any(role.lower() in current_role_lower for role in non_teaching_roles)
    
    # Special case for "Head of [Subject]" pattern
    is_head_of_subject = bool(re.search(r'head of (?:the )?(?:department of )?\w+', current_role_lower, re.IGNORECASE))
    
    # Determine teacher status with more nuanced logic
    if is_head_of_subject:
        is_teacher = True
    elif has_teaching_indicator and not has_non_teaching_indicator:
        is_teacher = True
    elif has_non_teaching_indicator and not has_teaching_indicator:
        is_teacher = False
    # If both indicators are present, default to False to be more strict
    elif has_teaching_indicator and has_non_teaching_indicator:
        is_teacher = False
    
    # Improve subject specificity
    generic_subjects = ['education', 'general studies', 'general education', 'teaching']
    if current_subject.lower() in generic_subjects and current_role:
        # Try to extract a more specific subject from the role
        subject_keywords = [
            'math', 'mathematics', 'algebra', 'calculus', 'statistics',
            'science', 'physics', 'chemistry', 'biology', 'geology', 'astronomy',
            'english', 'literature', 'writing', 'reading', 'language arts',
            'history', 'social studies', 'geography', 'economics', 'government',
            'computer science', 'programming', 'coding', 'computer programming',
            'art', 'music', 'drama', 'theater', 'dance', 'visual arts',
            'physical education', 'pe', 'health', 'health education',
            'foreign language', 'spanish', 'french', 'german', 'chinese', 'arabic',
            'special education', 'gifted education', 'esl', 'english as a second language'
        ]
        
        # Look for subject keywords in the role
        for keyword in subject_keywords:
            if keyword in current_role_lower:
                # Capitalize the first letter of each word for better formatting
                current_subject = ' '.join(word.capitalize() for word in keyword.split())
                break
    
    # Update the result
    result['is_currently_teacher'] = is_teacher
    result['subject'] = current_subject
    
    return result
    
    return result

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
    
    1. Subject: Be specific about the subject they teach. For example:
       - Instead of "English", use "English Literature" or "English as a Second Language (ESL)"
       - Instead of "Math", use "Mathematics", "Calculus", or "Statistics"
       - Instead of "Science", use "Physics", "Chemistry", "Biology", etc.
       - For primary/elementary teachers that don't have a specific subject, use "Primary Education" or "Elementary Education"
       - Only use "Education" as a last resort if when no specific subject can be determined.
    
    2. Bio: A professional, anonymized 2-3 sentence bio. Remove all personally identifiable information.
    
    3. Nationality: Most likely nationality based on their name (use demonym form, e.g., "Egyptian" not "Egypt")
    
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
    
    Teacher Information:
    {teacher_info}
    
    Format your response as JSON:
    {{
        "subject": "Specific subject name (be specific, avoid generic terms like 'Education' or 'General Studies')",
        "bio": "Professional bio that is anonymized and 2-3 sentences long",
        "nationality": "Nationality (demonym form)",
        "preferred_grade_level": "One of the exact grade levels listed above",
        "is_currently_teacher": boolean
    }}
    """
    
    try:
        # Get model configuration
        config = get_model_config("teacher_profile")
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are an expert in education who creates structured data about teachers. Be very strict about who qualifies as a teacher."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["max_tokens"],
            temperature=0.2,  # Lower temperature for more consistent results
            response_format=config.get("response_format", None)
        )
        
        # Parse the JSON response and validate
        result = json.loads(response.choices[0].message.content)
        
        # Apply additional validation
        result = validate_teacher_status(teacher_data, result)
        
        return result
        
    except Exception as e:
        print(f"Error processing teacher profile: {str(e)}")
        # Default to False on error to avoid false positives
        return {
            "subject": "Unknown", 
            "bio": "Professional educator with teaching experience.", 
            "nationality": "Not specified",
            "preferred_grade_level": "Not specified",
            "is_currently_teacher": False
        }

def batch_curriculum_and_school(teacher_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get curriculum experience, teaching experience years, current school, 
    and school website by processing teacher data.
    
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
    
    if not isinstance(teacher_data, dict):
        return result
        
    # Extract employment history
    employment_history = []
    i = 0
    while True:
        org_key = f'employment_history/{i}/organization_name'
        if org_key not in teacher_data:
            break
            
        # Get employment entry
        entry = {
            'organization': str(teacher_data.get(org_key, '')).strip(),
            'title': str(teacher_data.get(f'employment_history/{i}/title', '')).strip(),
            'current': bool(teacher_data.get(f'employment_history/{i}/current', False)),
            'start_date': str(teacher_data.get(f'employment_history/{i}/start_date', '')).strip()
        }
        
        # Only add if we have an organization name
        if entry['organization'] and entry['organization'].lower() not in ['none', 'n/a', 'not specified']:
            employment_history.append(entry)
        
        i += 1
    
    # Sort by current status (current first) and then by start_date (most recent first)
    employment_history.sort(
        key=lambda x: (
            not x['current'],  # Current jobs first
            x['start_date'] if x['start_date'] else '1900-01-01'  # Then by start date
        ),
        reverse=True  # Most recent first
    )
    
    # Try to get current school from employment history
    current_school = ""
    for job in employment_history:
        if job['current'] or not current_school:  # Take first current job or first job if no current
            current_school = job['organization']
            if current_school:
                break
    
    # If no school found, try other fields
    if not current_school:
        for field in ['current_employer', 'company', 'organization']:
            if field in teacher_data and teacher_data[field]:
                current_school = str(teacher_data[field]).strip()
                if current_school and current_school.lower() not in ['none', 'n/a', 'not specified']:
                    break
    
    # Clean up school name
    if current_school:
        # Remove common prefixes/suffixes
        current_school = re.sub(r'^\s*(?:at|from|,|\bat\b)\s*', '', current_school, flags=re.IGNORECASE)
        current_school = current_school.strip()
        result['current_school'] = current_school
    
    # Try to get teaching experience from employment history
    teaching_years = 0
    teaching_keywords = ['teacher', 'educator', 'instructor', 'professor', 'lecturer', 'faculty']
    
    for job in employment_history:
        job_title = job['title'].lower()
        if any(keyword in job_title for keyword in teaching_keywords):
            # Try to extract years from title (e.g., "5 years")
            year_match = re.search(r'(\d+)\s*(?:year|yr|yrs)', job_title)
            if year_match:
                teaching_years += int(year_match.group(1))
            else:
                # Default to 1 year if no specific duration mentioned
                teaching_years += 1
    
    if teaching_years > 0:
        result['teaching_experience_years'] = min(teaching_years, 50)  # Cap at 50 years
    
    # Try to determine curriculum based on school name
    if current_school:
        school_lower = current_school.lower()
        if 'gems' in school_lower:
            result['curriculum_experience'] = 'British'
        elif 'sabis' in school_lower:
            result['curriculum_experience'] = 'IB'
        elif 'american' in school_lower or 'u.s.' in school_lower or 'us ' in school_lower:
            result['curriculum_experience'] = 'American'
        elif 'indian' in school_lower or 'cbse' in school_lower or 'icse' in school_lower:
            result['curriculum_experience'] = 'Indian'
    
    # Try to get school website if available
    if 'school_website' in teacher_data and teacher_data['school_website']:
        website = str(teacher_data['school_website']).strip()
        if website and website.lower() not in ['none', 'n/a', 'not specified', 'unknown', '']:
            if not website.startswith(('http://', 'https://')):
                website = 'https://' + website
            result['school_website'] = website
    
    # Final validation of results
    if not result['current_school'] and 'current_school' in teacher_data and teacher_data['current_school']:
        result['current_school'] = teacher_data['current_school']
    
    if not result['school_website'] and 'school_website' in teacher_data and teacher_data['school_website']:
        website = str(teacher_data['school_website']).strip()
        if website and website.lower() not in ['none', 'n/a', 'not specified', 'unknown', '']:
            if not website.startswith(('http://', 'https://')):
                website = 'https://' + website
            result['school_website'] = website
    
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
