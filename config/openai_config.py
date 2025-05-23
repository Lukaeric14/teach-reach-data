"""
OpenAI API configuration settings.

This file contains all OpenAI model configurations used across the application.
All models are set to use the most cost-effective options by default.
"""

# Base model configuration
DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"  # Using the most cost-effective model that meets our needs

# Model configurations for different tasks
MODEL_CONFIGS = {
    # Teacher profile processing (batch)
    "teacher_profile": {
        "model": DEFAULT_MODEL,
        "temperature": 0.5,
        "max_tokens": 350,
        "response_format": {"type": "json_object"}
    },
    
    # Curriculum and school processing (batch)
    "curriculum_school": {
        "model": DEFAULT_MODEL,
        "temperature": 0.5,
        "max_tokens": 350,
        "response_format": {"type": "json_object"}
    },
    
    # Individual teacher subject inference
    "teacher_subject": {
        "model": DEFAULT_MODEL,
        "temperature": 0.3,
        "max_tokens": 20
    },
    
    # Teacher bio generation
    "teacher_bio": {
        "model": DEFAULT_MODEL,
        "temperature": 0.7,
        "max_tokens": 100
    },
    
    # Teaching experience extraction
    "teaching_experience": {
        "model": DEFAULT_MODEL,
        "temperature": 0.1,
        "max_tokens": 10
    },
    
    # Grade level inference
    "grade_level": {
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "max_tokens": 20
    },
    
    # Curriculum experience inference
    "curriculum": {
        "model": DEFAULT_MODEL,
        "temperature": 0.1,
        "max_tokens": 20
    },
    
    # Nationality from name inference
    "nationality": {
        "model": DEFAULT_MODEL,
        "temperature": 0.1,
        "max_tokens": 30
    },
    
    # General text processing
    "general": {
        "model": DEFAULT_MODEL,
        "temperature": 0.3,
        "max_tokens": 500
    }
}

def get_model_config(config_name: str) -> dict:
    """
    Get the model configuration for a specific task.
    
    Args:
        config_name (str): Name of the configuration to retrieve
        
    Returns:
        dict: Model configuration dictionary
    """
    return MODEL_CONFIGS.get(config_name, {
        "model": DEFAULT_MODEL,
        "temperature": 0.3,
        "max_tokens": 500
    })
