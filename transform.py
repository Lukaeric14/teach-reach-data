import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, List
import datetime

# Import our utilities
from utils.openai_utils import enrich_teacher_profile

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

def process_file(input_file, output_file, batch_size=20, continue_from_existing=True):
    """
    Processes the input file through transformations in batches and saves the result.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path where the output CSV file will be saved
        batch_size (int): Number of teachers to process in total (not per batch)
        continue_from_existing (bool): Whether to continue from an existing output file
    """
    # Read the input CSV
    print(f"Reading input file: {input_file}")
    input_df = pd.read_csv(input_file)
    total_teachers = len(input_df)
    
    # Limit the number of teachers to process to batch_size
    if batch_size < total_teachers:
        print(f"Limiting processing to first {batch_size} teachers (out of {total_teachers})")
        input_df = input_df.head(batch_size)
        total_teachers = len(input_df)
    
    # Initialize with a fresh dataframe for base transformations
    df = pd.DataFrame()
    
    # Apply base transformations that don't use OpenAI
    base_transformations = load_base_transformations()
    print("Applying base transformations...")
    for i, transform_func in enumerate(base_transformations, 1):
        print(f"  Base transformation {i}: {transform_func.__name__}")
        df = transform_func(df, input_df)
    
    # Check if we should continue from an existing file
    start_idx = 0
    if os.path.exists(output_file) and continue_from_existing:
        try:
            existing_df = pd.read_csv(output_file)
            if len(existing_df) >= total_teachers:
                print(f"Output file already contains {len(existing_df)} records. Using it as the base.")
                # Skip processing and go to final transformations
                apply_final_transformations(existing_df.head(total_teachers), input_df, output_file)
                return
            elif len(existing_df) > 0:
                print(f"Output file contains {len(existing_df)} partial records. Resuming from record {len(existing_df) + 1}.")
                start_idx = len(existing_df)
                # Load the existing data to continue from where we left off
                df = existing_df
        except Exception as e:
            print(f"Error reading existing output file: {e}. Starting fresh.")
    
    # If we're not continuing from an existing file, save the base transformations
    if start_idx == 0:
        print(f"Saving base transformations to: {output_file}")
        df.to_csv(output_file, index=False)
    
    # Process teachers individually 
    print(f"Processing {total_teachers - start_idx} teachers individually...")
    process_teachers_individually(df, input_df, output_file, start_idx=start_idx)
    
    # Final save and report
    print(f"Successfully processed {total_teachers} records")

def process_teachers_individually(df, input_df, output_file, calculate_completion=True, start_idx=0):
    """
    Process teachers individually using a comprehensive single API call per teacher.
    
    Args:
        df (pd.DataFrame): DataFrame with base transformations applied
        input_df (pd.DataFrame): Original input DataFrame
        output_file (str): Path where the output CSV file will be saved
        calculate_completion (bool): Whether to calculate profile completion after processing
        start_idx (int): Index to start processing from (for resuming)
    """
    total_teachers = len(df)
    
    # Adjust total teachers based on start index
    remaining_teachers = total_teachers - start_idx
    if remaining_teachers <= 0:
        print("No teachers remaining to process.")
        return
    
    print(f"Starting individual teacher processing from index {start_idx} (total: {remaining_teachers} teachers)")
    
    # Initialize progress tracking
    processed_count = 0
    
    # Process teachers one by one starting from start_idx
    for teacher_idx in range(start_idx, total_teachers):
        teacher_start_time = time.time()
        teacher_df_slice = df.iloc[[teacher_idx]].copy() 
        teacher_name = teacher_df_slice.at[teacher_df_slice.index[0], 'name'] if 'name' in teacher_df_slice.columns else f"Teacher #{teacher_idx+1}"
        
        print(f"\nProcessing teacher {teacher_idx + 1} of {total_teachers}: {teacher_name}")
        
        try:
            # Convert row to dict and ensure all values are strings
            teacher_dict = {k: str(v) if v is not None else '' for k, v in teacher_df_slice.iloc[0].to_dict().items()}
            
            # Make a single comprehensive API call for this teacher
            print(f"  Calling enrich_teacher_profile for {teacher_name}...")
            enriched_data = enrich_teacher_profile(teacher_dict)
            print(f"  Received comprehensive profile data")
            
            # Update the teacher row (teacher_df_slice) with the enriched data
            for key, value in enriched_data.items():
                if pd.notna(value) and str(value).strip():
                    teacher_df_slice.at[teacher_df_slice.index[0], key] = value
            
            # Calculate profile completion for this teacher if requested
            if calculate_completion:
                teacher_df_slice = t50.transform(teacher_df_slice, input_df) 
            
            # Debug: Print what we're about to save for this row
            print(f"\nData for {teacher_name} after enrichment and completion:")
            for col in ['subject', 'bio', 'nationality', 'preferred_grade_level', 
                       'is_currently_teacher', 'curriculum_experience', 
                       'teaching_experience_years', 'current_school', 'school_website', 
                       'current_location_country', 'current_location_city', 'profile_completion_percentage']:
                if col in teacher_df_slice.columns:
                    print(f"  {col}: {teacher_df_slice.at[teacher_df_slice.index[0], col]}")
            
            # Save this teacher's data
            write_header = not os.path.exists(output_file) or (os.path.exists(output_file) and os.path.getsize(output_file) == 0) or (teacher_idx == start_idx)
            
            teacher_df_slice.to_csv(output_file, mode='a' if not write_header else 'w', header=write_header, index=False)
            
            # Report progress for this teacher
            processed_count += 1
            teacher_end_time = time.time()
            print(f"Teacher processed and saved in {teacher_end_time - teacher_start_time:.2f} seconds")
            print(f"Saved teacher data to: {output_file}")

        except Exception as e:
            print(f"Error processing teacher {teacher_name}: {e}")
            try:
                error_df = pd.DataFrame([{'teacher_id': teacher_df_slice.at[teacher_df_slice.index[0], 'teacher_id'] if 'teacher_id' in teacher_df_slice.columns else 'UNKNOWN',
                                          'name': teacher_name, 
                                          'error': str(e)}])
                error_log_file = os.path.join(os.path.dirname(output_file), "error_log.csv")
                log_header = not os.path.exists(error_log_file) or (os.path.exists(error_log_file) and os.path.getsize(error_log_file) == 0)
                error_df.to_csv(error_log_file, mode='a' if not log_header else 'w', header=log_header, index=False)
                print(f"Logged error for {teacher_name} to {error_log_file}")
            except Exception as log_err:
                print(f"Could not log error for {teacher_name}: {log_err}")

        if teacher_idx < total_teachers - 1:  
            print(f"Waiting 1 second before next teacher...")
            time.sleep(1)

    print(f"\nFinished processing {processed_count} teachers.")

def apply_final_transformations(df, input_df, output_file):
    """
    Apply final transformations to the DataFrame and save to CSV.
    
    Args:
        df (pd.DataFrame): The DataFrame to transform
        input_df (pd.DataFrame): Original input DataFrame
        output_file (str): Path where the output CSV file will be saved
    """
    print("\nApplying final transformations...")
    
    # Ensure all required columns exist
    required_columns = [
        'teacher_id', 'name', 'subject', 'headline', 'bio', 'curriculum_experience',
        'teaching_experience_years', 'current_location_country', 'current_location_city', 
        'nationality', 'preferred_grade_level', 'current_school', 'school_website', 
        'linkedin_url', 'email', 'source_id', 'created_at', 'is_currently_teacher',
        'profile_completion_percentage'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Adding missing column: {col}")
            df[col] = ''
    
    # Apply the profile completion calculation if not already done
    if 'profile_completion_percentage' not in df.columns or df['profile_completion_percentage'].isna().all():
        print("Calculating profile completion for final output...")
        df = t50.transform(df, input_df)
    
    # Ensure the completion percentage is an integer between 0 and 50
    if 'profile_completion_percentage' in df.columns:
        df['profile_completion_percentage'] = df['profile_completion_percentage'].fillna(0).astype(int)
    
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
            parts = [p.strip() for p in location.split(',') if p.strip()]
            if len(parts) >= 2:
                df.at[idx, 'current_location_city'] = parts[0]
                df.at[idx, 'current_location_country'] = parts[-1]
            else:
                # If only one part, determine if it's a city or country
                if any(country.lower() in location.lower() for country in ['UAE', 'United Arab Emirates']):
                    df.at[idx, 'current_location_country'] = 'United Arab Emirates'
                    df.at[idx, 'current_location_city'] = 'Dubai'  # Default city
                else:
                    df.at[idx, 'current_location_city'] = location.strip()
                    df.at[idx, 'current_location_country'] = 'United Arab Emirates'  # Default country
        df = df.drop('current_location', axis=1)
    
    # Apply profile completion calculation
    print("Calculating profile completion percentages...")
    df = t50.transform(df, input_df)
    
    # Ensure is_currently_teacher is properly set
    if 'is_currently_teacher' in df.columns:
        df['is_currently_teacher'] = df['is_currently_teacher'].fillna(False).astype(bool)
    
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
    
    # Create output directory based on current date
    output_dir = os.path.join("outputs", datetime.datetime.now().strftime("%m-%d"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default output filename if not provided
    if not args.output:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(output_dir, f"processed_teachers_{timestamp}.csv")
    
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
        print(f"Output directory: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"\nError during processing: {e}")
        exit(1)
