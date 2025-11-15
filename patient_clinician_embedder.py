"""
Patient-Clinician Embedding Generator for Optum Match POC
Supports multiple embedding models with semantic survey response processing
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import openai
from openai import OpenAI

# Optionally import VoyageAI if available
try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False
    print("VoyageAI not available. Install with: pip install voyageai")

@dataclass
class EmbeddingConfig:
    """Configuration for embedding model"""
    name: str
    provider: str
    model: str
    dimensions: int
    description: str

class PatientClinicianEmbedder:
    """
    Multi-model embedding generator for patient-clinician matching
    Converts survey responses into semantic vectors for similarity analysis
    """
    
    def __init__(self, openai_key: Optional[str] = None, voyage_key: Optional[str] = None):
        """Initialize embedder with API keys"""
        self.openai_key = openai_key
        self.voyage_key = voyage_key
        
        # Initialize OpenAI client if key provided
        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
        else:
            self.openai_client = None
            
        # Initialize VoyageAI client if available and key provided
        if VOYAGEAI_AVAILABLE and voyage_key:
            self.voyage_client = voyageai.Client(api_key=voyage_key)
        else:
            self.voyage_client = None
            
        # Available embedding models
        self.embedding_models = {
            "openai_ada_002": EmbeddingConfig(
                name="openai_ada_002",
                provider="openai", 
                model="text-embedding-ada-002",
                dimensions=1536,
                description="OpenAI Ada-002 - Legacy model, good baseline"
            ),
            "openai_small": EmbeddingConfig(
                name="openai_small",
                provider="openai",
                model="text-embedding-3-small", 
                dimensions=1536,
                description="OpenAI Small - Cost-effective, fast"
            ),
            "openai_large": EmbeddingConfig(
                name="openai_large", 
                provider="openai",
                model="text-embedding-3-large",
                dimensions=3072,
                description="OpenAI Large - High quality, expensive"
            ),
            "voyage_large": EmbeddingConfig(
                name="voyage_large",
                provider="voyage",
                model="voyage-3-large",
                dimensions=1024, 
                description="VoyageAI Large - Healthcare optimized"
            ),
            "voyage_3": EmbeddingConfig(
                name="voyage_3",
                provider="voyage", 
                model="voyage-3",
                dimensions=1024,
                description="VoyageAI v3 - Latest general model"
            )
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def convert_survey_to_text(self, survey_data: Dict, survey_type: str = "patient") -> str:
        """
        Convert patient or clinician survey response to semantic text
        
        Args:
            survey_data: Patient or clinician survey response
            survey_type: "patient" or "clinician"
        
        Returns:
            Rich text representation of survey responses
        """
        text_parts = []
        
        # Q1: Communication leadership preference
        if "Q1" in survey_data:
            leadership = survey_data["Q1"]
            if survey_type == "patient":
                text_parts.append(f"Prefers {leadership.lower()} to lead healthcare conversations")
            else:
                text_parts.append(f"Believes {leadership.lower()} should lead healthcare conversations")
        
        # Q2: Processing style  
        if "Q2" in survey_data:
            style = survey_data["Q2"]
            text_parts.append(f"Communication style: {style.lower()}")
            
        # Q3: Clinical services (rich multi-select content)
        if "Q3" in survey_data:
            services = survey_data["Q3"]
            if services:
                if survey_type == "patient":
                    text_parts.append(f"Seeks clinical services: {', '.join(services)}")
                else:
                    text_parts.append(f"Provides clinical services: {', '.join(services)}")
                    
        # Q4 & Q4-1: Integrative medicine
        if "Q4" in survey_data:
            interest = survey_data["Q4"]
            if survey_type == "patient":
                text_parts.append(f"Interest in natural medicine: {interest.lower()}")
            else:
                text_parts.append(f"Willingness to provide natural medicine guidance: {interest.lower()}")
                
        if "Q4-1" in survey_data and survey_data["Q4-1"]:
            treatments = survey_data["Q4-1"]  
            if survey_type == "patient":
                text_parts.append(f"Interested in treatments: {', '.join(treatments)}")
            else:
                text_parts.append(f"Supports treatments: {', '.join(treatments)}")
                
        # Q5: Languages
        if "Q5" in survey_data:
            languages = survey_data["Q5"]
            if isinstance(languages, list):
                text_parts.append(f"Comfortable speaking: {', '.join(languages)}")
            else:
                text_parts.append(f"Comfortable speaking: {languages}")
                
        # Q6: Gender identity  
        if "Q6" in survey_data:
            gender = survey_data["Q6"]
            if isinstance(gender, list):
                text_parts.append(f"Gender identity: {', '.join(gender)}")
            else:
                text_parts.append(f"Gender identity: {gender}")
                
        # Q6-1: Gender preference (patients only)
        if "Q6-1" in survey_data and survey_type == "patient":
            pref = survey_data["Q6-1"]
            text_parts.append(f"Doctor gender preference: {pref.lower()}")
            
        # Q7: Military service
        if "Q7" in survey_data:
            military = survey_data["Q7"]
            if military == "Yes":
                text_parts.append("Military service experience")
                
        # Q8: Digital services
        if "Q8" in survey_data and survey_data["Q8"]:
            digital = survey_data["Q8"]
            if survey_type == "patient":
                text_parts.append(f"Prefers digital services: {', '.join(digital)}")
            else:
                text_parts.append(f"Offers digital services: {', '.join(digital)}")
                
        # Q9: Special care areas (patients) OR expertise areas (clinicians Q9-Q26)
        if survey_type == "patient" and "Q9" in survey_data and survey_data["Q9"]:
            care_areas = survey_data["Q9"]
            text_parts.append(f"Needs specialized care in: {', '.join(care_areas)}")
        elif survey_type == "clinician":
            # Map expertise levels for clinicians (Q9-Q26)
            expertise_areas = []
            expert_areas = []
            experienced_areas = []
            
            expertise_map = {
                "Q9": "aging well", "Q10": "BIPOC health", "Q11": "body positive care",
                "Q12": "caregiver support", "Q13": "deaf and hard of hearing care", 
                "Q14": "end of life care", "Q15": "gender-affirming care",
                "Q16": "immigrant/refugee health", "Q17": "LGBTQ+ health",
                "Q18": "men's health care", "Q19": "women's health care",
                "Q20": "neurodiversity-affirming care", "Q21": "period positive care",
                "Q22": "sex positive care", "Q23": "substance use recovery", 
                "Q24": "trauma-informed care", "Q25": "veteran care", "Q26": "spiritual support"
            }
            
            for q_num, area in expertise_map.items():
                if q_num in survey_data:
                    level = survey_data[q_num]
                    if "expert" in level.lower():
                        expert_areas.append(area)
                    elif "supportive and have experience" in level.lower():
                        experienced_areas.append(area)
                        
            if expert_areas:
                text_parts.append(f"Expert in: {', '.join(expert_areas)}")
            if experienced_areas:
                text_parts.append(f"Experienced in: {', '.join(experienced_areas)}")
                
        # Q27: Free text expertise (clinicians only)
        if "Q27" in survey_data and survey_data["Q27"] and survey_type == "clinician":
            expertise = survey_data["Q27"]
            text_parts.append(f"Additional expertise: {expertise}")
            
        return ". ".join(text_parts) + "."
        
    def generate_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Generate embedding for given text using specified model
        
        Args:
            text: Text to embed
            model_name: Name of embedding model to use
            
        Returns:
            Embedding vector or None if failed
        """
        if model_name not in self.embedding_models:
            self.logger.error(f"Unknown model: {model_name}")
            return None
            
        config = self.embedding_models[model_name]
        
        try:
            if config.provider == "openai":
                if not self.openai_client:
                    self.logger.warning(f"OpenAI client not available for {model_name}")
                    return None
                    
                response = self.openai_client.embeddings.create(
                    input=text,
                    model=config.model
                )
                return response.data[0].embedding
                
            elif config.provider == "voyage":
                if not self.voyage_client:
                    self.logger.warning(f"VoyageAI client not available for {model_name}")
                    return None
                    
                response = self.voyage_client.embed(
                    texts=[text],
                    model=config.model
                )
                return response.embeddings[0]
                
        except Exception as e:
            self.logger.error(f"Error generating embedding with {model_name}: {str(e)}")
            return None
            
    def process_survey_data(self, patient_file: str, clinician_file: str, 
                          models: List[str] = None) -> Dict[str, Any]:
        """
        Process patient and clinician survey data into embeddings
        
        Args:
            patient_file: Path to patient responses JSON
            clinician_file: Path to clinician responses JSON  
            models: List of model names to use (default: all available)
            
        Returns:
            Dictionary with embeddings and metadata
        """
        if models is None:
            # Use available models based on API keys
            models = []
            if self.openai_client:
                models.extend(["openai_ada_002", "openai_small", "openai_large"])
            if self.voyage_client:
                models.extend(["voyage_large", "voyage_3"])
                
        if not models:
            self.logger.error("No models available - need API keys")
            return {}
            
        # Load survey data
        try:
            with open(patient_file, 'r') as f:
                patients = json.load(f)
            with open(clinician_file, 'r') as f:
                clinicians = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading survey data: {str(e)}")
            return {}
            
        results = {
            "metadata": {
                "patients_count": len(patients),
                "clinicians_count": len(clinicians), 
                "models_used": models,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "patients": {},
            "clinicians": {}
        }
        
        # Process patients
        self.logger.info(f"Processing {len(patients)} patients...")
        for patient in patients:
            patient_id = str(patient["patientId"])
            text = self.convert_survey_to_text(patient, "patient")
            
            results["patients"][patient_id] = {
                "text": text,
                "embeddings": {},
                "raw_data": patient
            }
            
            for model_name in models:
                embedding = self.generate_embedding(text, model_name) 
                if embedding:
                    results["patients"][patient_id]["embeddings"][model_name] = embedding
                time.sleep(0.1)  # Rate limiting
                
        # Process clinicians  
        self.logger.info(f"Processing {len(clinicians)} clinicians...")
        for clinician in clinicians:
            clinician_id = str(clinician["clinicianId"])
            text = self.convert_survey_to_text(clinician, "clinician")
            
            results["clinicians"][clinician_id] = {
                "text": text,
                "embeddings": {},
                "raw_data": clinician
            }
            
            for model_name in models:
                embedding = self.generate_embedding(text, model_name)
                if embedding:
                    results["clinicians"][clinician_id]["embeddings"][model_name] = embedding
                time.sleep(0.1)  # Rate limiting
                
        self.logger.info("Embedding generation complete!")
        return results


def main():
    """Demo usage of PatientClinicianEmbedder"""
    print("🏥 OPTUM PATIENT-CLINICIAN EMBEDDING GENERATOR")
    print("=" * 60)
    
    # Initialize embedder (replace with your API keys)
    embedder = PatientClinicianEmbedder(
        openai_key="<YOUR_OPENAI_KEY>",  # Replace with actual key
        voyage_key="<YOUR_VOYAGE_KEY>"   # Replace with actual key
    )
    
    # Show available models
    print("\n📋 Available Embedding Models:")
    for name, config in embedder.embedding_models.items():
        status = "✅" if (config.provider == "openai" and embedder.openai_client) or \
                       (config.provider == "voyage" and embedder.voyage_client) else "❌"
        print(f"  {status} {name}: {config.description}")
        
    # Generate embeddings (with demo data)
    print("\n🔄 Processing Survey Data...")
    results = embedder.process_survey_data(
        "data/patient_responses.json",
        "data/clinician_responses.json"
    )
    
    if results:
        print(f"\n✅ Success! Processed:")
        print(f"  • {results['metadata']['patients_count']} patients")  
        print(f"  • {results['metadata']['clinicians_count']} clinicians")
        print(f"  • {len(results['metadata']['models_used'])} embedding models")
        
        # Show sample conversion
        if results["patients"]:
            patient_id = list(results["patients"].keys())[0] 
            print(f"\n📝 Sample Patient Text Conversion (ID {patient_id}):")
            print(f"  {results['patients'][patient_id]['text'][:200]}...")
            
    else:
        print("\n❌ No embeddings generated - check API keys")


if __name__ == "__main__":
    main()