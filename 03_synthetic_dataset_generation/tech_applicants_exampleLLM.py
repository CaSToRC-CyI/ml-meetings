import json
from typing import Any, Dict, List
from pydantic import BaseModel, create_model
import pandas as pd
from ollama import chat
from tqdm import tqdm

model_name = "deepseek-r1:latest"

class JobApplicant(BaseModel):
    full_name: str
    email: str
    current_role: str
    years_experience: int
    technical_skills: list[str]
    desired_role: str
    preferred_tech_stack: list[str]
    education_level: str
    remote_preference: bool
    expected_salary: int

def generate_applicants(num_records: int = 1000) -> pd.DataFrame:
    """
    Generate fictional job applicants data using an LLM.
    
    Args:
        num_records: Number of applicant records to generate
        
    Returns:
        DataFrame containing the generated applicants data
    """
    applicants = []
    
    prompt = """Generate a fictional tech job applicant profile including:
    - Their current role and years of experience
    - Technical skills they possess
    - The role they're applying for
    - Their preferred technologies
    - Education level
    - Whether they prefer remote work
    - Expected salary
    Make it realistic and varied."""

    for _ in tqdm(range(num_records)):
        try:
            response = chat(
                messages=[{'role': 'user', 'content': prompt}],
                model=model_name,
                format=JobApplicant.model_json_schema(),
            )
            
            applicant = JobApplicant.model_validate_json(response.message.content)
            applicants.append(applicant)
            
        except Exception as e:
            print(f"Error generating record: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame([a.model_dump() for a in applicants])
    
    return df

if __name__ == "__main__":
    # Generate the data
    df = generate_applicants(5)
    
    # Basic data analysis
    print("\nData Overview:")
    print(df.describe(include='all'))
    
    print("\nMost Common Desired Roles:")
    print(df['desired_role'].value_counts().head())
    
    print("\nAverage Expected Salary by Years of Experience:")
    print(df.groupby('years_experience')['expected_salary'].mean().sort_index())
    
    # Save to CSV
    df.to_csv('tech_applicants.csv', index=False)
    print("\nData saved to tech_applicants.csv")