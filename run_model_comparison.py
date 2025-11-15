#!/usr/bin/env python3
"""
Multi-Model Comparison Evaluation for Optum Patient-Clinician Matching POC
Compares different embedding models on the same dataset
"""

import json
import logging
import time
import os
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Import our custom modules
from patient_clinician_embedder import PatientClinicianEmbedder
from patient_clinician_matcher_fixed import PatientClinicianMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_model_comparison():
    """Run comprehensive multi-model comparison evaluation"""
    
    print("🤖 MULTI-MODEL COMPARISON EVALUATION")
    print("=" * 50)
    
    # Check API keys
    has_openai = bool(os.getenv('OPENAI_API_KEY'))
    has_voyage = bool(os.getenv('VOYAGE_API_KEY'))
    
    print(f"\n🔑 API Key Status:")
    print(f"  • OpenAI: {'✅' if has_openai else '❌'}")
    print(f"  • VoyageAI: {'✅' if has_voyage else '❌'}")
    
    if not (has_openai or has_voyage):
        print(f"\n⚠️  No API keys available - comparison requires at least one model")
        print(f"   Set OPENAI_API_KEY and/or VOYAGE_API_KEY environment variables")
        return
    
    # Initialize components
    embedder = PatientClinicianEmbedder()
    matcher = PatientClinicianMatcher()
    
    print(f"\n📊 Processing Survey Data...")
    
    # Generate embeddings for all available models
    try:
        embeddings_data = embedder.process_survey_data(
            "data/patient_responses.json",
            "data/clinician_responses.json",
            models=None  # Auto-detect available models
        )
        
        if not embeddings_data or not embeddings_data['metadata']['models_used']:
            print("❌ No embeddings generated")
            return
            
        available_models = embeddings_data['metadata']['models_used']
        print(f"✅ Generated embeddings using models: {', '.join(available_models)}")
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return
    
    # Compare all available models
    print(f"\n🔄 Running matches for each model...")
    
    model_results = {}
    model_stats = {}
    
    for model_name in available_models:
        try:
            print(f"  🎯 Testing {model_name}...")
            
            # Generate matches for this model
            matches = matcher.generate_comprehensive_matches(
                embeddings_data, model_name
            )
            
            if not matches:
                print(f"    ❌ No matches generated for {model_name}")
                continue
                
            model_results[model_name] = matches
            
            # Calculate statistics
            total_patients = len(matches)
            avg_top_score = np.mean([results.top_matches[0].final_score 
                                   for results in matches.values()])
            avg_base_similarity = np.mean([results.top_matches[0].similarity_score
                                         for results in matches.values()])
            avg_business_boost = np.mean([results.top_matches[0].business_logic_boost
                                        for results in matches.values()])
            
            # Count patients with business logic boosts
            boosted_patients = sum(1 for results in matches.values()
                                 if results.top_matches[0].business_logic_boost > 0)
            boost_percentage = (boosted_patients / total_patients) * 100
            
            model_stats[model_name] = {
                'total_patients': total_patients,
                'avg_top_score': avg_top_score,
                'avg_base_similarity': avg_base_similarity,
                'avg_business_boost': avg_business_boost,
                'boosted_patients': boosted_patients,
                'boost_percentage': boost_percentage
            }
            
            print(f"    ✅ {model_name}: {total_patients} patients, avg score {avg_top_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error with model {model_name}: {str(e)}")
            print(f"    ❌ Error with {model_name}: {str(e)}")
            continue
    
    if not model_results:
        print("❌ No successful model results")
        return
    
    # Display comparison results
    print(f"\n📈 MODEL COMPARISON RESULTS")
    print("=" * 40)
    
    print(f"{'Model':<20} {'Avg Score':<12} {'Base Sim':<12} {'Boost':<12} {'Boost %':<10}")
    print("-" * 65)
    
    for model_name, stats in model_stats.items():
        print(f"{model_name:<20} "
              f"{stats['avg_top_score']:<12.3f} "
              f"{stats['avg_base_similarity']:<12.3f} "
              f"{stats['avg_business_boost']:<12.3f} "
              f"{stats['boost_percentage']:<10.1f}")
    
    # Analyze agreement between models
    if len(model_results) > 1:
        print(f"\n🔍 MODEL AGREEMENT ANALYSIS")
        print("-" * 30)
        
        model_names = list(model_results.keys())
        agreements = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                agreement_count = 0
                total_comparisons = 0
                
                for patient_id in model_results[model1].keys():
                    if patient_id in model_results[model2]:
                        top1 = model_results[model1][patient_id].top_matches[0].clinician_id
                        top2 = model_results[model2][patient_id].top_matches[0].clinician_id
                        
                        if top1 == top2:
                            agreement_count += 1
                        total_comparisons += 1
                
                agreement_rate = (agreement_count / total_comparisons * 100) if total_comparisons > 0 else 0
                agreements[f"{model1} vs {model2}"] = agreement_rate
                print(f"  {model1} vs {model2}: {agreement_rate:.1f}% agreement ({agreement_count}/{total_comparisons})")
    
    # Show detailed example for best performing model
    best_model = max(model_stats.keys(), key=lambda m: model_stats[m]['avg_top_score'])
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
    print("-" * 30)
    
    sample_patient = list(model_results[best_model].keys())[0]
    sample_results = model_results[best_model][sample_patient]
    
    print(f"Sample Patient {sample_patient} matches:")
    for i, match in enumerate(sample_results.top_matches[:3]):
        print(f"  #{i+1} Clinician {match.clinician_id}")
        print(f"      Final Score: {match.final_score:.3f}")
        print(f"      (Base: {match.similarity_score:.3f} + Boost: {match.business_logic_boost:.3f})")
        if match.contributing_factors:
            print(f"      Factors: {'; '.join(match.contributing_factors[:2])}")
    
    # Export detailed comparison
    export_filename = f"model_comparison_{int(time.time())}.csv"
    export_data = []
    
    for model_name, matches in model_results.items():
        for patient_id, results in matches.items():
            for i, match in enumerate(results.top_matches[:5]):  # Top 5 per patient
                export_data.append({
                    'model': model_name,
                    'patient_id': patient_id,
                    'rank': i + 1,
                    'clinician_id': match.clinician_id,
                    'final_score': round(match.final_score, 3),
                    'base_similarity': round(match.similarity_score, 3),
                    'business_boost': round(match.business_logic_boost, 3),
                    'contributing_factors': '; '.join(match.contributing_factors) if match.contributing_factors else 'None'
                })
    
    df = pd.DataFrame(export_data)
    df.to_csv(export_filename, index=False)
    print(f"\n💾 Detailed results exported to: {export_filename}")
    
    # Export summary statistics
    summary_filename = f"model_summary_{int(time.time())}.csv"
    summary_df = pd.DataFrame([
        {
            'model': model_name,
            **stats
        }
        for model_name, stats in model_stats.items()
    ])
    summary_df.to_csv(summary_filename, index=False)
    print(f"📊 Summary statistics exported to: {summary_filename}")
    
    print(f"\n🎉 Model comparison complete!")

def main():
    """Main model comparison runner"""
    try:
        run_model_comparison()
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        print(f"\n❌ Model comparison failed: {str(e)}")
        print(f"   Check logs for details")

if __name__ == "__main__":
    main()