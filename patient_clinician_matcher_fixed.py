"""
Patient-Clinician Matching Engine for Optum Match POC
Calculates similarity scores and generates explainable match results
"""

import json
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity

@dataclass  
class MatchResult:
    """Individual match result between patient and clinician"""
    patient_id: str
    clinician_id: str
    similarity_score: float
    model_used: str
    contributing_factors: List[str]
    business_logic_boost: float = 0.0
    final_score: float = 0.0

@dataclass
class PatientMatchResults:
    """Complete match results for a patient"""
    patient_id: str
    patient_text: str
    matches: List[MatchResult]
    top_matches: List[MatchResult]
    
class PatientClinicianMatcher:
    """
    Advanced matching engine for patient-clinician similarity analysis
    Supports multi-model embeddings with explainable results
    """
    
    def __init__(self):
        """Initialize matcher"""
        self.logger = logging.getLogger(__name__)
        
        # Business logic boost factors
        self.boost_factors = {
            "language_match": 0.08,      # Language compatibility
            "gender_preference": 0.05,   # Gender preference match  
            "integrative_medicine": 0.03, # Integrative medicine alignment
            "special_care_overlap": 0.10, # Special care area alignment
            "communication_style": 0.02,  # Communication style match
            "military_experience": 0.03   # Military experience match
        }
        
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0][0]
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
            
    def analyze_business_logic_factors(self, patient_data: Dict, clinician_data: Dict) -> Tuple[float, List[str]]:
        """
        Analyze business logic factors for boosting similarity
        
        Args:
            patient_data: Patient survey response
            clinician_data: Clinician survey response
            
        Returns:
            Tuple of (boost_score, contributing_factors)
        """
        boost_score = 0.0
        factors = []
        
        # Language compatibility
        patient_languages = patient_data.get("Q5", [])
        clinician_languages = clinician_data.get("Q5", [])
        
        if isinstance(patient_languages, str):
            patient_languages = [patient_languages]
        if isinstance(clinician_languages, str):
            clinician_languages = [clinician_languages]
            
        language_overlap = set(patient_languages) & set(clinician_languages)
        if language_overlap and "Prefer not to say" not in patient_languages:
            boost_score += self.boost_factors["language_match"]
            factors.append(f"Language compatibility: {', '.join(language_overlap)}")
            
        # Gender preference matching
        patient_gender_pref = patient_data.get("Q6-1")
        clinician_gender = clinician_data.get("Q6")
        patient_gender = patient_data.get("Q6")
        
        if patient_gender_pref == "Yes" and patient_gender and clinician_gender:
            if isinstance(clinician_gender, list):
                clinician_gender_set = set(clinician_gender)
            else:
                clinician_gender_set = {clinician_gender}
                
            if isinstance(patient_gender, list):
                patient_gender_set = set(patient_gender)  
            else:
                patient_gender_set = {patient_gender}
                
            if patient_gender_set & clinician_gender_set:
                boost_score += self.boost_factors["gender_preference"]
                factors.append("Gender preference alignment")
                
        # Integrative medicine alignment
        patient_interest = patient_data.get("Q4", "")
        clinician_willing = clinician_data.get("Q4", "")
        
        if patient_interest in ["Very interested", "Somewhat interested"] and \
           clinician_willing in ["Very willing", "Somewhat willing"]:
            patient_treatments = set(patient_data.get("Q4-1", []))
            clinician_treatments = set(clinician_data.get("Q4-1", []))
            treatment_overlap = patient_treatments & clinician_treatments
            
            if treatment_overlap:
                boost_score += self.boost_factors["integrative_medicine"] * len(treatment_overlap)
                factors.append(f"Integrative medicine alignment: {', '.join(list(treatment_overlap)[:3])}")
                
        # Special care area matching
        patient_care_needs = set(patient_data.get("Q9", []))
        
        # Map clinician expertise (Q9-Q26) to care areas
        clinician_expert_areas = set()
        clinician_experienced_areas = set()
        
        expertise_map = {
            "Q9": "Aging well", "Q10": "BIPOC health", "Q11": "Body positive care",
            "Q12": "Caregiver support", "Q13": "Caring for deaf and hard of hearing patients",
            "Q14": "End of life care", "Q15": "Gender affirming care", 
            "Q16": "Immigrant or refugee health", "Q17": "LGBTQ+ health",
            "Q18": "Men's health care", "Q19": "Women's health care",
            "Q20": "Neurodiversity affirming care", "Q21": "Period positive care",
            "Q22": "Sex positive care", "Q23": "Substance use and addiction recovery",
            "Q24": "Trauma informed care", "Q25": "Veteran supportive care", "Q26": "Spiritual care"
        }
        
        for q_num, care_area in expertise_map.items():
            if q_num in clinician_data:
                level = clinician_data[q_num]
                if "expert" in level.lower():
                    clinician_expert_areas.add(care_area)
                elif "supportive and have experience" in level.lower():
                    clinician_experienced_areas.add(care_area)
                    
        # Check for exact matches in care needs
        expert_matches = patient_care_needs & clinician_expert_areas
        experienced_matches = patient_care_needs & clinician_experienced_areas
        
        if expert_matches:
            boost_score += self.boost_factors["special_care_overlap"] * len(expert_matches)
            factors.append(f"Expert care match: {', '.join(list(expert_matches)[:2])}")
            
        if experienced_matches:
            boost_score += self.boost_factors["special_care_overlap"] * 0.5 * len(experienced_matches)
            factors.append(f"Experienced care match: {', '.join(list(experienced_matches)[:2])}")
            
        # Communication style matching
        patient_comm = patient_data.get("Q2")
        clinician_comm = clinician_data.get("Q2")
        
        if patient_comm == clinician_comm:
            boost_score += self.boost_factors["communication_style"]
            factors.append(f"Communication style match: {patient_comm.lower()}")
            
        # Military experience matching
        patient_military = patient_data.get("Q7") == "Yes"
        clinician_military = clinician_data.get("Q7") == "Yes"
        
        if patient_military and clinician_military:
            boost_score += self.boost_factors["military_experience"]
            factors.append("Shared military experience")
            
        return boost_score, factors
        
    def match_patient_to_clinicians(self, patient_id: str, embeddings_data: Dict, 
                                  model_name: str = "openai_small") -> PatientMatchResults:
        """
        Match a patient to all clinicians using specified embedding model
        
        Args:
            patient_id: ID of patient to match
            embeddings_data: Embeddings data from PatientClinicianEmbedder
            model_name: Embedding model to use for similarity calculation
            
        Returns:
            PatientMatchResults with ranked clinician matches
        """
        if patient_id not in embeddings_data["patients"]:
            raise ValueError(f"Patient {patient_id} not found in embeddings data")
            
        patient_info = embeddings_data["patients"][patient_id]
        patient_embedding = patient_info["embeddings"].get(model_name)
        
        if not patient_embedding:
            raise ValueError(f"No {model_name} embedding found for patient {patient_id}")
            
        matches = []
        
        # Calculate similarity with each clinician
        for clinician_id, clinician_info in embeddings_data["clinicians"].items():
            clinician_embedding = clinician_info["embeddings"].get(model_name)
            
            if not clinician_embedding:
                self.logger.warning(f"No {model_name} embedding for clinician {clinician_id}")
                continue
                
            # Calculate base cosine similarity
            base_similarity = self.calculate_cosine_similarity(
                patient_embedding, clinician_embedding
            )
            
            # Calculate business logic boost
            boost_score, contributing_factors = self.analyze_business_logic_factors(
                patient_info["raw_data"], clinician_info["raw_data"]
            )
            
            # Calculate final score
            final_score = min(1.0, base_similarity + boost_score)
            
            match = MatchResult(
                patient_id=patient_id,
                clinician_id=clinician_id,
                similarity_score=base_similarity,
                model_used=model_name,
                contributing_factors=contributing_factors,
                business_logic_boost=boost_score,
                final_score=final_score
            )
            
            matches.append(match)
            
        # Sort by final score (highest first)
        matches.sort(key=lambda x: x.final_score, reverse=True)
        
        # Get top 10 matches
        top_matches = matches[:10]
        
        return PatientMatchResults(
            patient_id=patient_id,
            patient_text=patient_info["text"],
            matches=matches,
            top_matches=top_matches
        )
        
    def generate_comprehensive_matches(self, embeddings_data: Dict, 
                                     model_name: str = "openai_small") -> Dict[str, PatientMatchResults]:
        """
        Generate matches for all patients against all clinicians
        
        Args:
            embeddings_data: Embeddings data from PatientClinicianEmbedder
            model_name: Embedding model to use
            
        Returns:
            Dictionary mapping patient IDs to their match results
        """
        all_matches = {}
        
        self.logger.info(f"Generating comprehensive matches using {model_name}...")
        
        for patient_id in embeddings_data["patients"].keys():
            try:
                matches = self.match_patient_to_clinicians(
                    patient_id, embeddings_data, model_name
                )
                all_matches[patient_id] = matches
                
            except Exception as e:
                self.logger.error(f"Error matching patient {patient_id}: {str(e)}")
                continue
                
        return all_matches
        
    def export_results_to_json(self, match_results: Dict[str, PatientMatchResults], 
                             output_file: str) -> None:
        """
        Export match results to JSON format
        
        Args:
            match_results: Match results from generate_comprehensive_matches
            output_file: Output JSON file path
        """
        export_data = {
            "metadata": {
                "total_patients": len(match_results),
                "matches_per_patient": 10,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": {}
        }
        
        for patient_id, results in match_results.items():
            export_data["results"][patient_id] = {
                "patient_text": results.patient_text,
                "top_matches": [
                    {
                        "rank": i + 1,
                        "clinician_id": match.clinician_id,
                        "similarity_score": round(match.similarity_score, 3),
                        "business_logic_boost": round(match.business_logic_boost, 3),
                        "final_score": round(match.final_score, 3),
                        "contributing_factors": match.contributing_factors,
                        "model_used": match.model_used
                    }
                    for i, match in enumerate(results.top_matches)
                ]
            }
            
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Results exported to {output_file}")
        
    def print_match_summary(self, match_results: Dict[str, PatientMatchResults]) -> None:
        """
        Print a summary of match results
        
        Args:
            match_results: Match results to summarize
        """
        print(f"\n📊 MATCH RESULTS SUMMARY")
        print("=" * 50)
        
        total_patients = len(match_results)
        avg_top_score = np.mean([results.top_matches[0].final_score 
                               for results in match_results.values()])
        avg_boost = np.mean([match.business_logic_boost 
                           for results in match_results.values() 
                           for match in results.top_matches[:3]])
        
        print(f"📈 Patients Matched: {total_patients}")
        print(f"⭐ Average Top Match Score: {avg_top_score:.3f}")
        print(f"🚀 Average Business Logic Boost: {avg_boost:.3f}")
        
        # Show sample matches
        sample_patient = list(match_results.keys())[0]
        sample_results = match_results[sample_patient]
        
        print(f"\n🎯 Sample Matches for Patient {sample_patient}:")
        print(f"Patient Profile: {sample_results.patient_text[:100]}...")
        
        for i, match in enumerate(sample_results.top_matches[:3]):
            print(f"\n  #{i+1} Clinician {match.clinician_id} (Score: {match.final_score:.3f})")
            print(f"      Base Similarity: {match.similarity_score:.3f}")
            if match.business_logic_boost > 0:
                print(f"      Business Boost: +{match.business_logic_boost:.3f}")
            if match.contributing_factors:
                print(f"      Factors: {'; '.join(match.contributing_factors[:2])}")
                

def main():
    """Demo usage of PatientClinicianMatcher"""
    print("🎯 OPTUM PATIENT-CLINICIAN MATCHING ENGINE")
    print("=" * 60)
    
    # This would normally load embeddings from PatientClinicianEmbedder
    print("\n⚠️  Demo mode - need embeddings data from PatientClinicianEmbedder")
    print("   1. First run patient_clinician_embedder.py with API keys")
    print("   2. Then use the embeddings data with this matcher")
    
    matcher = PatientClinicianMatcher()
    print(f"\n🔧 Business Logic Boost Factors:")
    for factor, boost in matcher.boost_factors.items():
        print(f"  • {factor.replace('_', ' ').title()}: +{boost:.3f}")
        
        
if __name__ == "__main__":
    main()