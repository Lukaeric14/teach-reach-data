import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, List
import datetime

# Import our new batched utilities
from utils.batch_openai_utils import batch_teacher_profile, batch_curriculum_and_school

# Import only essential transformations that don't require OpenAI API calls
from transformations import t_01_add_teacher_id as t01
from transformations import t_02_add_name_column as t02
from transformations import t_04_add_headline_column as t04
from transformations import t_06_add_empty_columns as t06
from transformations import t_10_add_linkedin_url as t10
from transformations import t_12_add_created_at as t12
from transformations import t_16_add_email_column as t16
from transformations import t_17_add_source_id as t17
from transformations import t_50_calculate_profile_completion as t50

def load_base_transformations():
    """
    Load and return the transformation functions in the order they should be applied.
    """
    return [
        t01.transform,  # Add teacher ID
        t02.transform,  # Add name column
        t04.transform,  # Add headline column
        t06.transform,  # Add empty columns
        t10.transform,  # Add LinkedIn profile URL
        t12.transform,  # Add created_at timestamp
        t16.transform,  # Add Email column
        t17.transform,  # Add source_id column
        t50.transform   # Calculate profile completion percentage (must be last)
    ]

def process_file(input_file, output_file, batch_size=5, continue_from_existing=True):
    """
    Processes the input file through transformations in batches and saves the result.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path where the output CSV file will be saved
        batch_size (int): Number of teachers to process in each batch
        continue_from_existing (bool): Whether to continue from an existing output file
    """
    # Read the input CSV
    print(f"Reading input file: {input_file}")
    input_df = pd.read_csv(input_file)
    total_teachers = len(input_df)
    
    # Check if we should continue from an existing file
    if os.path.exists(output_file) and continue_from_existing:
        try:
            existing_df = pd.read_csv(output_file)
            if len(existing_df) == total_teachers:
                print(f"Output file already contains {total_teachers} records. Using it as the base.")
                df = existing_df
                # Skip batched processing and go to final transformations
                apply_final_transformations(df, input_df, output_file)
                return
            elif len(existing_df) > 0:
                print(f"Output file contains {len(existing_df)} partial records. Starting a new one.")
        except Exception as e:
            print(f"Error reading existing output file: {e}. Starting fresh.")
    
    # Initialize with a fresh dataframe for base transformations
    df = pd.DataFrame()
    
    # Apply base transformations that don't use OpenAI
    base_transformations = load_base_transformations()
    print("Applying base transformations...")
    for i, transform_func in enumerate(base_transformations, 1):
        print(f"  Base transformation {i}: {transform_func.__name__}")
        df = transform_func(df, input_df)
    
    # Save intermediate result
    print(f"Saving base transformations to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Process teachers in batches
    process_in_batches(df, input_df, output_file, batch_size)
    
    # Final save and report
    print(f"Successfully processed {len(df)} records")

def process_in_batches(df, input_df, output_file, batch_size=5):
    """
    Process teachers in batches using the batched API calls.
    
    Args:
        df (pd.DataFrame): DataFrame with base transformations applied
        input_df (pd.DataFrame): Original input DataFrame
        output_file (str): Path where the output CSV file will be saved
        batch_size (int): Number of teachers to process in each batch
    """
    total_teachers = len(input_df)
    
    # Create batches
    num_batches = (total_teachers + batch_size - 1) // batch_size  # Ceiling division
    
    # Process each batch
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_teachers)
        
        print(f"\nProcessing batch {batch_idx+1}/{num_batches} (teachers {start_idx+1}-{end_idx} of {total_teachers})")
        
        batch_start_time = time.time()
        
        # Process each teacher in the batch
        for idx in range(start_idx, end_idx):
            # Get both input row and transformed row
            input_row = input_df.iloc[idx].dropna().to_dict()
            transformed_row = df.iloc[idx].dropna().to_dict() if idx < len(df) else {}
            
            # Combine data from both sources
            teacher_data = {**input_row, **transformed_row}
            
            # Get location data directly from input (not from API)
            country = input_row.get('country', 'United Arab Emirates')
            city = input_row.get('city', 'Dubai')
            
            # Process teacher with batched API calls
            print(f"  Processing teacher {idx+1}...")
            
            # First API call - profile data
            profile_data = batch_teacher_profile(teacher_data)
            time.sleep(0.5)  # Small delay to avoid rate limiting
            
            # Second API call - curriculum and school data
            school_data = batch_curriculum_and_school(teacher_data)
            
            # Update the dataframe with all results, ensuring proper data types
            for field, value in profile_data.items():
                # Convert boolean to string for CSV compatibility
                if field == 'is_currently_teacher' and isinstance(value, bool):
                    value = str(value).lower()
                df.at[idx, field] = value
                
            for field, value in school_data.items():
                df.at[idx, field] = value
                
            # Add location data from input
            df.at[idx, 'current_location_country'] = country
            df.at[idx, 'current_location_city'] = city
            
            # Small delay between teachers
            time.sleep(0.2)
        
        batch_end_time = time.time()
        print(f"Batch processed in {batch_end_time - batch_start_time:.2f} seconds")
        
        # Save after each batch
        print(f"Saving batch results to: {output_file}")
        df.to_csv(output_file, index=False)
        
        # Add a delay between batches if not the last batch
        if batch_idx < num_batches - 1:
            print("Waiting between batches...")
            time.sleep(2)
    
    return df

def apply_final_transformations(df, input_df, output_file):
    """
    Apply any final transformations or validations to the dataframe.
    
    Args:
        df (pd.DataFrame): The processed dataframe
        input_df (pd.DataFrame): The original input dataframe
        output_file (str): Path where the output CSV file will be saved
    """
    # Ensure all required columns exist
    required_columns = [
        'teacher_id', 'name', 'subject', 'headline', 'bio', 'curriculum_experience',
        'teaching_experience_years', 'current_location_country', 'current_location_city', 
        'nationality', 'preferred_grade_level', 'current_school', 'school_website', 
        'linkedin_url', 'email', 'source_id', 'created_at'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Adding missing column: {col}")
            df[col] = ''
    
    # Handle migration from current_location to country/city split
    if 'current_location' in df.columns and (
        df['current_location_country'].isna().all() or 
        (df['current_location_country'] == '').all()):
        print("Migrating current_location to separate country and city fields...")
        
        # Process each row to extract country and city
        for idx, row in df.iterrows():
            location = row.get('current_location', '')
            if not location or location == 'Not specified':
                df.at[idx, 'current_location_country'] = 'United Arab Emirates'
                df.at[idx, 'current_location_city'] = 'Dubai'
                continue
                
            # Handle common patterns like "Dubai, United Arab Emirates"
            parts = location.split(',')
            if len(parts) >= 2:
                df.at[idx, 'current_location_city'] = parts[0].strip()
                df.at[idx, 'current_location_country'] = parts[1].strip()
            else:
                # If only one part, determine if it's a city or country
                if location in ['Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman', 'Ras Al Khaimah', 'Fujairah', 'Umm Al Quwain']:
                    df.at[idx, 'current_location_city'] = location
                    df.at[idx, 'current_location_country'] = 'United Arab Emirates'
                else:
                    df.at[idx, 'current_location_country'] = location
                    df.at[idx, 'current_location_city'] = ''
        
        # Remove the old column to avoid confusion
        df = df.drop('current_location', axis=1)
    
    # Save the final result
    print(f"Saving final output to: {output_file}")
    df.to_csv(output_file, index=False)
    
    return df

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
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process teacher data with AI enhancements.')
    parser.add_argument('-i', '--input', default='inputv2.csv', 
                        help='Input CSV file (default: inputv2.csv)')
    parser.add_argument('-o', '--output', 
                        help='Output CSV file (default: output_<timestamp>.csv)')
    parser.add_argument('-b', '--batch-size', type=int, default=5,
                        help='Number of teachers to process in each batch (default: 5)')
    parser.add_argument('--continue', dest='continue_existing', action='store_true',
                        help='Continue from existing output file if it exists')
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if not args.output:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output_{timestamp}.csv"
    
    # First list available models
    list_available_models()
    
    # Backup existing output if it exists and not continuing
    if not args.continue_existing and os.path.exists(args.output) and not os.path.exists(f"{args.output}.backup"):
        print(f"Creating backup of {args.output}")
        os.system(f"cp \"{args.output}\" \"{args.output}.backup\"")
    
    # Read the input file to check city/country columns
    try:
        input_df = pd.read_csv(args.input)
        print(f"\nChecking input data for location information...")
        print(f"Input file: {args.input}")
        print(f"Input columns: {list(input_df.columns)}")
        if 'city' in input_df.columns:
            print(f"City column exists with {input_df['city'].count()} non-null values")
        if 'country' in input_df.columns:
            print(f"Country column exists with {input_df['country'].count()} non-null values")
    except Exception as e:
        print(f"Error reading input file {args.input}: {e}")
        exit(1)
    
    # Remove the existing output file to force reprocessing if not continuing
    if not args.continue_existing and os.path.exists(args.output):
        print(f"\nRemoving existing {args.output} to force reprocessing")
        os.remove(args.output)
    
    # Process the file
    start_time = time.time()
    try:
        process_file(args.input, args.output, batch_size=args.batch_size, 
                   continue_from_existing=args.continue_existing)
        end_time = time.time()
        print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
        print(f"Output saved to: {os.path.abspath(args.output)}")
    except Exception as e:
        print(f"\nError during processing: {e}")
        exit(1)
