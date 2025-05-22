import os
import pandas as pd
import sys
from dotenv import load_dotenv
from utils.openai_utils import infer_curriculum_experience

def main():
    # Load environment variables
    load_dotenv()
    
    # Read the input file
    input_file = 'inputv2.csv'
    df = pd.read_csv(input_file)
    
    # Get the first teacher's data
    teacher_data = df.iloc[0].dropna().to_dict()
    
    print("Testing curriculum inference with teacher data:")
    print("-" * 50)
    print(f"Name: {teacher_data.get('first_name', '')} {teacher_data.get('last_name', '')}")
    print(f"Nationality: {teacher_data.get('nationality', 'Not specified')}")
    print(f"Current School: {teacher_data.get('current_school', 'Not specified')}")
    
    print("\nCalling infer_curriculum_experience...")
    curriculum = infer_curriculum_experience(teacher_data)
    print(f"\nFinal curriculum: {curriculum}")

if __name__ == "__main__":
    main()
