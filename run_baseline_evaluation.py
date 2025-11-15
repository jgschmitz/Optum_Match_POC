#!/usr/bin/env python3
"""
Baseline Comparison Evaluation for Optum Patient-Clinician Matching POC
Compares semantic embedding matches against simple rules-based baseline
"""

import json
import logging
import time
import random
from typing import Dict, List, Tuple, Any
import pandas as pd

# Import our custom modules
from patient_clinician_embedder import PatientClinicianEmbedder
from patient_clinician_matcher_fixed import PatientClinicianMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineRulesMatcher:
    """Simple rules-based matcher for baseline comparison"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_rules_based_score(self, patient_data: Dict, clinician_data: Dict) -> Tuple[float, List[str]]:
        """
        Calculate compatibility score using only simple rule matching
        
        Args:
            patient_data: Patient survey response
            clinician_data: Clinician survey response
            
        Returns:
            Tuple of (score, matching_factors)
        """
        score = 0.0
        factors = []
        
        # Language matching (30% weight)
        patient_languages = set(patient_data.get("Q5", []))
        clinician_languages = set(clinician_data.get("Q5", []))
        if isinstance(patient_languages, str):
            patient_languages = {patient_languages}
        if isinstance(clinician_languages, str):
            clinician_languages = {clinician_languages}
            
        language_overlap = patient_languages & clinician_languages
        if language_overlap and "Prefer not to say" not in patient_languages:
            score += 0.30
            factors.append(f"Language: {', '.join(language_overlap)}")
            
        # Gender preference (20% weight)
        patient_gender_pref = patient_data.get("Q6-1")
        if patient_gender_pref == "Yes":
            patient_gender = patient_data.get("Q6")
            clinician_gender = clinician_data.get("Q6")
            if patient_gender == clinician_gender:
                score += 0.20
                factors.append("Gender preference match")
                
        # Special care needs (25% weight)
        patient_care_needs = set(patient_data.get("Q9", []))
        clinician_expertise = set()
        
        # Simple mapping of clinician Q9-Q26 to care areas
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
            if q_num in clinician_data and "expert" in clinician_data[q_num].lower():
                clinician_expertise.add(care_area)
                
        care_overlap = patient_care_needs & clinician_expertise
        if care_overlap:
            score += 0.25 * min(1.0, len(care_overlap) / 3.0)  # Scale by overlap
            factors.append(f"Care expertise: {', '.join(list(care_overlap)[:2])}")
            
        # Communication style (15% weight)
        if patient_data.get("Q2") == clinician_data.get("Q2"):
            score += 0.15
            factors.append("Communication style match")
            
        # Military experience (10% weight)
        if patient_data.get("Q7") == "Yes" and clinician_data.get("Q7") == "Yes":
            score += 0.10
            factors.append("Military experience")
            
        return min(1.0, score), factors
        
    def generate_baseline_matches(self, patient_file: str, clinician_file: str) -> Dict[str, List[Dict]]:
        """Generate baseline rule-based matches"""
        
        # Load data
        with open(patient_file, 'r') as f:
            patients_list = json.load(f)
        with open(clinician_file, 'r') as f:
            clinicians_list = json.load(f)
            
        results = {}
        
        for patient_data in patients_list:
            patient_id = str(patient_data.get('patientId', patient_data.get('id', 'unknown')))
            matches = []
            
            for clinician_data in clinicians_list:
                clinician_id = str(clinician_data.get('clinicianId', clinician_data.get('id', 'unknown')))
                score, factors = self.calculate_rules_based_score(patient_data, clinician_data)
                
                matches.append({
                    'clinician_id': clinician_id,
                    'score': score,
                    'factors': factors,
                    'method': 'rules_based'
                })
                
            # Sort by score
            matches.sort(key=lambda x: x['score'], reverse=True)
            results[patient_id] = matches[:10]  # Top 10
            
        return results

def run_baseline_comparison():
    """Run comprehensive baseline comparison evaluation"""
    
    print("📊 BASELINE COMPARISON EVALUATION")
    print("=" * 50)
    
    # Initialize components
    embedder = PatientClinicianEmbedder()
    semantic_matcher = PatientClinicianMatcher()
    baseline_matcher = BaselineRulesMatcher()
    
    print("\n🔄 Generating baseline rule-based matches...")
    baseline_results = baseline_matcher.generate_baseline_matches(
        "data/patient_responses.json",
        "data/clinician_responses.json"
    )
    
    print("✅ Generated baseline matches")
    
    # Generate semantic matches (if API keys available)
    print("\n🤖 Generating semantic embedding matches...")
    try:
        embeddings_data = embedder.process_survey_data(
            "data/patient_responses.json",
            "data/clinician_responses.json",
            models=None
        )
        
        if embeddings_data and embeddings_data['metadata']['models_used']:
            primary_model = embeddings_data['metadata']['models_used'][0]
            semantic_results = semantic_matcher.generate_comprehensive_matches(
                embeddings_data, primary_model
            )
            print(f"✅ Generated semantic matches using {primary_model}")
            has_semantic = True
        else:
            print("⚠️  No API keys - semantic comparison unavailable")
            semantic_results = {}
            has_semantic = False
            
    except Exception as e:
        logger.error(f"Error generating semantic matches: {str(e)}")
        print(f"⚠️  Semantic matching failed: {str(e)}")
        semantic_results = {}
        has_semantic = False
    
    # Compare approaches
    print("\n📈 COMPARISON ANALYSIS")
    print("=" * 30)
    
    comparison_data = []
    
    for patient_id in baseline_results.keys():
        baseline_top_score = baseline_results[patient_id][0]['score'] if baseline_results[patient_id] else 0
        baseline_top_clinician = baseline_results[patient_id][0]['clinician_id'] if baseline_results[patient_id] else None
        
        if has_semantic and patient_id in semantic_results:
            semantic_top_score = semantic_results[patient_id].top_matches[0].final_score if semantic_results[patient_id].top_matches else 0
            semantic_top_clinician = semantic_results[patient_id].top_matches[0].clinician_id if semantic_results[patient_id].top_matches else None
            agreement = (baseline_top_clinician == semantic_top_clinician)
        else:
            semantic_top_score = None
            semantic_top_clinician = None
            agreement = None
            
        comparison_data.append({
            'patient_id': patient_id,
            'baseline_top_score': baseline_top_score,
            'baseline_top_clinician': baseline_top_clinician,
            'semantic_top_score': semantic_top_score,
            'semantic_top_clinician': semantic_top_clinician,
            'top_match_agreement': agreement
        })
    
    # Print summary
    if has_semantic:
        total_patients = len(comparison_data)
        agreements = sum(1 for d in comparison_data if d['top_match_agreement'])
        agreement_rate = (agreements / total_patients) * 100 if total_patients > 0 else 0
        
        avg_baseline_score = sum(d['baseline_top_score'] for d in comparison_data) / total_patients
        avg_semantic_score = sum(d['semantic_top_score'] for d in comparison_data if d['semantic_top_score']) / total_patients
        
        print(f"📊 Comparison Results:")
        print(f"  • Patients evaluated: {total_patients}")
        print(f"  • Top match agreement rate: {agreement_rate:.1f}%")
        print(f"  • Average baseline score: {avg_baseline_score:.3f}")
        print(f"  • Average semantic score: {avg_semantic_score:.3f}")
        
        # Show examples of disagreements
        disagreements = [d for d in comparison_data if not d['top_match_agreement']]
        if disagreements:
            print(f"\n🔍 Example Disagreements:")
            for i, disagree in enumerate(disagreements[:3]):
                print(f"  Patient {disagree['patient_id']}:")
                print(f"    Baseline choice: Clinician {disagree['baseline_top_clinician']} ({disagree['baseline_top_score']:.3f})")
                print(f"    Semantic choice: Clinician {disagree['semantic_top_clinician']} ({disagree['semantic_top_score']:.3f})")
    else:
        avg_baseline_score = sum(d['baseline_top_score'] for d in comparison_data) / len(comparison_data)
        print(f"📊 Baseline-Only Results:")
        print(f"  • Patients evaluated: {len(comparison_data)}")  
        print(f"  • Average baseline score: {avg_baseline_score:.3f}")
        print(f"  • Note: Add API keys to enable semantic comparison")
    
    # Export results
    export_filename = f"baseline_comparison_{int(time.time())}.csv"
    df = pd.DataFrame(comparison_data)
    df.to_csv(export_filename, index=False)
    print(f"\n💾 Results exported to: {export_filename}")
    
    print(f"\n🎯 Baseline evaluation complete!")

def main():
    """Main baseline evaluation runner"""
    try:
        run_baseline_comparison()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {str(e)}")
        print(f"\n❌ Baseline evaluation failed: {str(e)}")
        print(f"   Check logs for details")

if __name__ == "__main__":
    main()