import json
from typing import Any, Dict, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import pandas as pd
from ollama import chat
from tqdm import tqdm
import random

model_name = "deepseek-r1:latest"

class GeneticMarker(BaseModel):
    marker_id: str
    status: str  # "positive" or "negative"
    significance: str

class Complication(BaseModel):
    condition: str
    onset_date: str
    severity: str  # mild, moderate, severe
    status: str  # active, resolved, managed

class Medication(BaseModel):
    name: str
    start_date: str
    end_date: str | None
    dosage: str
    frequency: str
    effectiveness_score: float = Field(ge=0, le=100)

class Patient(BaseModel):
    # Demographics
    patient_id: str
    age: int = Field(ge=0, le=100)
    gender: str
    ethnicity: str
    family_history: bool
    genetic_markers: List[GeneticMarker]
    
    # Disease Information
    diagnosis_date: str
    initial_severity_score: float = Field(ge=1, le=10)
    progression_rate: float = Field(ge=0.1, le=5.0)
    biomarker_levels: List[float]
    complications: List[Complication]
    
    # Treatment Information
    medication_history: List[Medication]
    treatment_response_score: float = Field(ge=0, le=100)

def generate_prompt(index: int) -> str:
    return f"""Generate data for a fictional rare disease patient following these STRICT value ranges:

1. Patient Demographics:
   - age: integer between 0 and 100
   - gender: "M" or "F"
   - initial_severity_score: decimal between 1.0 and 10.0
   - progression_rate: decimal between 0.1 and 5.0
   - treatment_response_score: decimal between 0 and 100

2. Disease Context:
   - genetic disorder affecting multiple systems
   - diagnosis typically occurs in childhood
   - disease progression is variable
   - multiple treatment options with varying effectiveness
   - complications affect multiple organ systems

3. Data Requirements:
   - biomarker_levels: array of 3-5 numbers between 0-100
   - genetic_markers: 2-4 markers with status "positive" or "negative"
   - complications: 1-3 complications with severity "mild", "moderate", or "severe"
   - medication_history: 1-3 medications with effectiveness_score between 0-100

4. Ensure consistency:
   - All dates must be in YYYY-MM-DD format
   - Dates must be chronologically valid (diagnosis before complications)
   - Disease progression should match complications
   - Treatment response should align with complications

This is patient #{index} in the dataset. Strictly follow all numeric ranges specified above."""

def validate_and_fix_data(data: dict) -> dict:
    """Pre-validate and fix common issues in the LLM response"""
    if 'initial_severity_score' in data:
        # Ensure initial_severity_score is float and within range
        try:
            score = float(data['initial_severity_score'])
            data['initial_severity_score'] = min(10.0, max(1.0, score))
        except (ValueError, TypeError):
            print("error when generating initial_severity_score")
            data['initial_severity_score'] = 5.0  # fallback to middle value

    if 'biomarker_levels' in data:
        # Ensure biomarker_levels are floats and within range
        data['biomarker_levels'] = [
            min(100.0, max(0.0, float(x))) for x in data['biomarker_levels']
        ]

    if 'progression_rate' in data:
        # Ensure progression_rate is float and within range
        try:
            rate = float(data['progression_rate'])
            data['progression_rate'] = min(5.0, max(0.1, rate))
        except (ValueError, TypeError):
            print("error when generating progression_rate")

            data['progression_rate'] = 1.0  # fallback to middle value

    return data

def create_dataset(num_records: int = 1000) -> pd.DataFrame:
    patients = []
    max_retries = 3  # Maximum number of retries per record
    
    for i in tqdm(range(num_records)):
        for retry in range(max_retries):
            try:
                response = chat(
                    messages=[{
                        'role': 'user', 
                        'content': generate_prompt(i)
                    }],
                    model=model_name,
                    format=Patient.model_json_schema(),
                )
                
                # Parse the response and pre-validate/fix data
                data = json.loads(response.message.content)
                fixed_data = validate_and_fix_data(data)
                
                # Create Patient object with fixed data
                patient = Patient.model_validate(fixed_data)
                patients.append(patient)
                break  # Success, move to next record
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error for patient {i}, attempt {retry + 1}: {str(e)}")
                if retry == max_retries - 1:
                    print(f"Failed to generate valid JSON for patient {i} after {max_retries} attempts")
            except Exception as e:
                print(f"Error generating patient {i}, attempt {retry + 1}: {str(e)}")
                if retry == max_retries - 1:
                    print(f"Failed to generate patient {i} after {max_retries} attempts")
    
    # Convert to DataFrame
    df = pd.DataFrame([p.model_dump() for p in patients])
    
    # Post-processing to ensure data quality
    df['age'] = df['age'].clip(0, 100)
    df['initial_severity_score'] = df['initial_severity_score'].clip(1, 10)
    df['progression_rate'] = df['progression_rate'].clip(0.1, 5.0)
    df['treatment_response_score'] = df['treatment_response_score'].clip(0, 100)
    
    return df

def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform validation checks on the generated dataset
    """
    validation_results = {
        'total_records': len(df),
        'age_distribution': df['age'].describe(),
        'gender_distribution': df['gender'].value_counts(normalize=True),
        'avg_complications': df['complications'].apply(len).mean(),
        'avg_medications': df['medication_history'].apply(len).mean(),
        'avg_response_score': df['treatment_response_score'].mean(),
        'missing_values': df.isnull().sum(),
    }
    
    return validation_results

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic patient records...")
    df = create_dataset(num_records=5)
    
    # Validate dataset
    print("\nValidating dataset...")
    validation_results = validate_dataset(df)
    
    # Print validation results
    print("\nDataset Statistics:")
    for key, value in validation_results.items():
        print(f"\n{key}:")
        print(value)
    
    # Save to CSV
    output_file = 'rare_disease_patients.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to {output_file}")
    
    # Display sample records
    print("\nSample Records:")
    print(df.head())