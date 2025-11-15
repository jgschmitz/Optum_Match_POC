#!/usr/bin/env python3
"""
Comprehensive Evaluation Demo for Optum Patient-Clinician Matching POC
Demonstrates full evaluation pipeline with explainable results
"""

import json
import logging
import time
import os
from typing import Dict, List, Any
import pandas as pd

# Import our custom modules
from patient_clinician_embedder import PatientClinicianEmbedder
from patient_clinician_matcher_fixed import PatientClinicianMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_demo_environment():
    """Setup demo environment and check requirements"""
    print("🔧 OPTUM PATIENT-CLINICIAN MATCHING EVALUATION")
    print("=" * 60)
    
    # Check for API key environment variables
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_voyage = bool(os.getenv('VOYAGE_API_KEY'))
    
    print(f"\n🔑 API Key Status:")
    print(f"  • OpenAI API Key: {'✅ Available' if has_openai else '❌ Missing'}")
    print(f"  • VoyageAI API Key: {'✅ Available' if has_voyage else '❌ Missing'}")
    
    if not (has_openai or has_voyage):
        print(f"\n⚠️  Warning: No API keys detected!")
        print(f"   Set OPENAI_API_KEY and/or VOYAGE_API_KEY environment variables")
        print(f"   Demo will run with synthetic embeddings for illustration")
        
    return has_openai, has_voyage

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of patient-clinician matching"""
    
    # Setup
    has_openai, has_voyage = setup_demo_environment()
    
    # Initialize components
    embedder = PatientClinicianEmbedder()
    matcher = PatientClinicianMatcher()
    
    print(f"\n📊 Processing Survey Data...")
    
    # Generate embeddings for all patients and clinicians
    try:
        embeddings_data = embedder.process_survey_data(
            patient_file="data/patient_responses.json",
            clinician_file="data/clinician_responses.json",
            models=None  # Will auto-detect based on available API keys
        )
        
        if not embeddings_data:
            print("❌ Failed to generate embeddings - check API keys and data files")
            return
            
        print(f"✅ Generated embeddings for {embeddings_data['metadata']['patients_count']} patients and {embeddings_data['metadata']['clinicians_count']} clinicians")
        print(f"🤖 Models used: {', '.join(embeddings_data['metadata']['models_used'])}")
        
    except Exception as e:
        logger.error(f"Error processing survey data: {str(e)}")
        print(f"❌ Error processing data: {str(e)}")
        return
    
    # Select best available model for evaluation
    available_models = embeddings_data['metadata']['models_used']
    if not available_models:
        print("❌ No models available for evaluation")
        return
        
    primary_model = available_models[0]  # Use first available model
    print(f"\n🎯 Running comprehensive matching with {primary_model}...")
    
    # Generate all patient-clinician matches
    try:
        all_matches = matcher.generate_comprehensive_matches(
            embeddings_data, 
            model_name=primary_model
        )
        
        if not all_matches:
            print("❌ No matches generated")
            return
            
        print(f"✅ Generated {len(all_matches)} complete patient match profiles")
        
    except Exception as e:
        logger.error(f"Error generating matches: {str(e)}")
        print(f"❌ Error generating matches: {str(e)}")
        return
    
    # Display detailed results
    print_detailed_evaluation_results(all_matches, primary_model)
    
    # Export results
    export_evaluation_results(all_matches, primary_model)
    
    # Show business logic analysis
    analyze_business_logic_impact(all_matches)
    
    print(f"\n🎉 Evaluation complete!")
    
def print_detailed_evaluation_results(match_results: Dict, model_name: str):
    """Print detailed evaluation results with explanations"""
    
    print(f"\n📋 DETAILED EVALUATION RESULTS")
    print("=" * 50)
    
    # Overall statistics
    total_patients = len(match_results)
    avg_top_score = sum(results.top_matches[0].final_score for results in match_results.values()) / total_patients
    
    print(f"📊 Overall Statistics:")
    print(f"  • Total Patients: {total_patients}")
    print(f"  • Average Top Match Score: {avg_top_score:.3f}")
    print(f"  • Embedding Model Used: {model_name}")
    
    # Show top 3 patient examples with detailed explanations
    print(f"\n🎯 Sample Patient Match Analysis:")
    
    for i, (patient_id, results) in enumerate(list(match_results.items())[:3]):
        print(f"\n--- Patient {patient_id} ---")
        print(f"Profile: {results.patient_text[:120]}...")
        print(f"Top 3 Clinician Matches:")
        
        for j, match in enumerate(results.top_matches[:3]):
            print(f"\n  #{j+1} Clinician {match.clinician_id}")
            print(f"      Final Score: {match.final_score:.3f}")
            print(f"      Base Similarity: {match.similarity_score:.3f}")
            
            if match.business_logic_boost > 0:
                print(f"      Business Logic Boost: +{match.business_logic_boost:.3f}")
                
            if match.contributing_factors:
                print(f"      Contributing Factors:")
                for factor in match.contributing_factors:
                    print(f"        • {factor}")
            else:
                print(f"      Contributing Factors: Semantic similarity only")
    
def export_evaluation_results(match_results: Dict, model_name: str):
    """Export results to multiple formats"""
    
    print(f"\n💾 Exporting Results...")
    
    # Export to JSON
    json_filename = f"optum_match_results_{model_name}_{int(time.time())}.json"
    try:
        matcher = PatientClinicianMatcher()
        matcher.export_results_to_json(match_results, json_filename)
        print(f"✅ JSON results exported to: {json_filename}")
    except Exception as e:
        logger.error(f"Error exporting JSON: {str(e)}")
        print(f"❌ Failed to export JSON: {str(e)}")
    
    # Export to CSV for stakeholder review
    csv_filename = f"optum_match_summary_{model_name}_{int(time.time())}.csv"
    try:
        export_to_csv(match_results, csv_filename)
        print(f"✅ CSV summary exported to: {csv_filename}")
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        print(f"❌ Failed to export CSV: {str(e)}")

def export_to_csv(match_results: Dict, filename: str):
    """Export match results to CSV format for stakeholder review"""
    
    rows = []
    for patient_id, results in match_results.items():
        patient_text = results.patient_text[:100] + "..."  # Truncate for readability
        
        for i, match in enumerate(results.top_matches[:5]):  # Top 5 matches per patient
            rows.append({
                'patient_id': patient_id,
                'patient_profile': patient_text,
                'match_rank': i + 1,
                'clinician_id': match.clinician_id,
                'final_score': round(match.final_score, 3),
                'base_similarity': round(match.similarity_score, 3),
                'business_logic_boost': round(match.business_logic_boost, 3),
                'contributing_factors': '; '.join(match.contributing_factors) if match.contributing_factors else 'Semantic only',
                'model_used': match.model_used
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

def analyze_business_logic_impact(match_results: Dict):
    """Analyze the impact of business logic boosting"""
    
    print(f"\n🧠 BUSINESS LOGIC IMPACT ANALYSIS")
    print("=" * 40)
    
    total_matches = sum(len(results.matches) for results in match_results.values())
    boosted_matches = sum(1 for results in match_results.values() 
                         for match in results.matches if match.business_logic_boost > 0)
    
    if total_matches == 0:
        print("No matches to analyze")
        return
        
    boost_percentage = (boosted_matches / total_matches) * 100
    
    print(f"📈 Boost Statistics:")
    print(f"  • Total Matches Analyzed: {total_matches}")
    print(f"  • Matches with Business Logic Boost: {boosted_matches}")
    print(f"  • Percentage Boosted: {boost_percentage:.1f}%")
    
    # Calculate average boost by factor type
    factor_counts = {}
    for results in match_results.values():
        for match in results.matches:
            for factor in match.contributing_factors:
                factor_type = factor.split(':')[0].strip()
                if factor_type not in factor_counts:
                    factor_counts[factor_type] = 0
                factor_counts[factor_type] += 1
    
    if factor_counts:
        print(f"\n🔍 Most Common Boost Factors:")
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        for factor, count in sorted_factors[:5]:
            print(f"  • {factor}: {count} matches")

def main():
    """Main evaluation runner"""
    try:
        run_comprehensive_evaluation()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        print(f"\n❌ Evaluation failed: {str(e)}")
        print(f"   Check logs for details")

if __name__ == "__main__":
    main()