import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
# Import transformations with proper module syntax for numbered files
from transformations import t_01_add_teacher_id as t01
from transformations import t_02_add_name_column as t02
from transformations import t_03_infer_subject as t03
from transformations import t_04_add_headline_column as t04
from transformations import t_05_add_teacher_bio as t05
from transformations import t_06_add_empty_columns as t06
from transformations import t_07_add_curriculum_experience as t07
from transformations import t_08_calculate_teaching_experience as t08
from transformations import t_09_add_current_location as t09
from transformations import t_10_add_linkedin_url as t10
from transformations import t_11_add_preferred_grade_level as t11
from transformations import t_12_add_created_at as t12
from transformations import t_13_infer_nationality as t13
from transformations import t_14_current_school as t14
from transformations import t_15_current_school_website as t15
from transformations import t_16_add_email_column as t16
from transformations import t_17_add_source_id as t17

def load_transformations():
    """
    Load and return all transformation functions in the correct order.
    """
    return [
        t01.transform,  # Add teacher ID
        t02.transform,  # Add name column
        t03.transform,  # Infer subject
        t04.transform,  # Add headline column
        t05.transform,  # Add teacher bio
        t06.transform,  # Add empty columns
        t07.transform,  # Add curriculum experience
        t08.transform,  # Calculate teaching experience
        t09.transform,  # Add current location country
        t10.transform,  # Add LinkedIn profile URL
        t11.transform,  # Add preferred grade level
        t12.transform,  # Add created_at timestamp
        t13.transform,  # Infer nationality from name
        t14.transform,  # Add current school
        t15.transform,  # Add current school website
        t16.transform,  # Add Email column
        t17.transform   # Add source_id column
    ]

def process_file(input_file, output_file):
    """
    Processes the input file through all transformations and saves the result.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path where the output CSV file will be saved
    """
    # Read the input CSV
    print(f"Reading input file: {input_file}")
    input_df = pd.read_csv(input_file)
    
    # Initialize with empty dataframe
    df = pd.DataFrame()
    
    # Get all transformations
    transformations = load_transformations()
    
    # Apply each transformation
    for i, transform_func in enumerate(transformations, 1):
        print(f"Applying transformation {i}: {transform_func.__name__}")
        df = transform_func(df, input_df)
    
    # Save the result
    print(f"Saving output to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Successfully processed {len(df)} records")

def list_available_models():
    """List all available models from the OpenAI API."""
    try:
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Initialize the client with the API key from environment variables
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        print("\nFetching available models...")
        models = client.models.list()
        
        print("\nAvailable models:")
        for model in sorted(model.id for model in models.data):
            print(f"- {model}")
            
    except Exception as e:
        print(f"\nError listing models: {e}")
        print("\nPlease make sure you have set the OPENAI_API_KEY environment variable.")
        print("You can set it temporarily by running:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr create a .env file in the project root with:")
        print("OPENAI_API_KEY=your-api-key-here")

if __name__ == "__main__":
    # First list available models
    list_available_models()
    
    # Then run the transformations
    input_file = "inputv2.csv"
    output_file = "output.csv"
    process_file(input_file, output_file)
