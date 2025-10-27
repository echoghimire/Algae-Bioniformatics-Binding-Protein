"""
Main Integration Script for Algae Protein Optimization System
Demonstrates complete workflow from data loading to optimization and analysis
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import our custom modules
from algae_protein_optimizer import AlgaeProteinOptimizer
from co2_binding_analyzer import CO2BindingAnalyzer
from comprehensive_analyzer import ComprehensiveAnalyzer

def main():
    """Main function demonstrating the complete optimization workflow"""
    
    print("🧬 ALGAE PROTEIN OPTIMIZATION SYSTEM")
    print("=" * 60)
    print("Optimizing proteins for enhanced CO2 absorption in algae")
    print("Targeting Carbonic Anhydrase from Chlorella sorokiniana")
    print("=" * 60)
    
    # Set up workspace
    workspace_path = r"c:\Users\Gunjan Ghimire\Downloads\Testting code\Algae-Bioniformatics-Binding-Protein"
    
    try:
        # Initialize all components
        print("\\n1. Initializing Optimization System...")
        optimizer = AlgaeProteinOptimizer(workspace_path)
        co2_analyzer = CO2BindingAnalyzer()
        comprehensive_analyzer = ComprehensiveAnalyzer(workspace_path)
        
        # Load target proteins
        print("\\n2. Loading Target Proteins...")
        proteins = optimizer.load_target_proteins("Carbonic Anhydrase FASTA sequence.fasta")
        
        if not proteins:
            print("❌ Error: Could not load FASTA file")
            return
        
        print(f"✅ Loaded {len(proteins)} target protein(s)")
        for accession, data in proteins.items():
            print(f"   - {accession}: {data['description'][:80]}...")
        
        # Analyze existing trial data
        print("\\n3. Analyzing Existing Trial Data...")
        comprehensive_analyzer.load_all_trials()
        comparison_df = comprehensive_analyzer.create_comprehensive_comparison()
        
        if not comparison_df.empty:
            print(f"✅ Found {len(comparison_df)} existing trials")
            print("\\nExisting Trial Summary:")
            print(comparison_df[['Trial', 'Format', 'Best Score (Old)', 'Total Fitness']].head())
        else:
            print("ℹ️  No existing trials found - this will be the first run")
        
        # Run enhanced optimization for the main protein
        print("\\n4. Running Enhanced Genetic Algorithm...")
        main_accession = list(proteins.keys())[0]  # Use first protein
        
        # Configuration for demonstration
        config = {
            'population_size': 30,      # Reduced for faster demo
            'generations': 50,          # Reduced for faster demo
            'sequence_length': 25,
            'mutation_rate': 0.15,
            'crossover_rate': 0.7,
            'elite_count': 5
        }
        
        trial_name = f"Enhanced_Trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"   Trial Name: {trial_name}")
        print(f"   Target: {main_accession}")
        print(f"   Configuration: {config}")
        
        # Run optimization
        trial_results = optimizer.run_optimization_trial(
            accession=main_accession,
            trial_name=trial_name,
            config=config
        )
        
        print("\\n✅ Optimization Complete!")
        print(f"   Best Sequence: {trial_results['results']['best_sequence']}")
        print(f"   Fitness Scores: {trial_results['results']['best_fitness']}")
        
        # Detailed CO2 binding analysis
        print("\\n5. Performing Detailed CO2 Binding Analysis...")
        best_sequence = trial_results['results']['best_sequence']
        
        # Analyze CO2 binding properties
        co2_analysis = co2_analyzer.predict_co2_binding_affinity(best_sequence)
        catalytic_analysis = co2_analyzer.calculate_catalytic_efficiency(best_sequence)
        suggestions = co2_analyzer.generate_optimization_suggestions(best_sequence)
        
        print("\\nCO2 Binding Analysis Results:")
        print(f"   Overall Affinity: {co2_analysis['overall_affinity']:.4f}")
        print(f"   Zinc Binding Score: {co2_analysis['zinc_binding_score']:.4f}")
        print(f"   Catalytic Score: {co2_analysis['catalytic_score']:.4f}")
        print(f"   Overall Catalytic Efficiency: {catalytic_analysis['overall_efficiency']:.4f}")
        
        if suggestions:
            print("\\nOptimization Suggestions:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion}")
        
        # Create visualizations
        print("\\n6. Creating Visualizations...")
        
        # CO2 binding analysis visualization
        save_path = Path(workspace_path) / "Results" / trial_name / "co2_analysis.png"
        co2_analyzer.visualize_sequence_analysis(best_sequence, str(save_path))
        
        # Create interactive 3D viewer
        print("\\n7. Creating Interactive 3D Viewer...")
        try:
            viewer = optimizer.create_interactive_viewer(trial_name)
            if viewer:
                print("✅ 3D viewer created successfully")
                # In a Jupyter environment, this would display the viewer
                # viewer.show()
            else:
                print("⚠️  Could not create 3D viewer (may need Jupyter environment)")
        except Exception as e:
            print(f"⚠️  3D viewer creation failed: {e}")
        
        # Comprehensive analysis
        print("\\n8. Generating Comprehensive Analysis...")
        
        # Reload trials including the new one
        comprehensive_analyzer.load_all_trials()
        comparison_df = comprehensive_analyzer.create_comprehensive_comparison()
        
        # Performance analysis
        performance_analysis = comprehensive_analyzer.analyze_performance_trends()
        
        print("\\nPerformance Summary:")
        print(f"   Total Trials: {performance_analysis['overall_statistics']['total_trials']}")
        print(f"   Average Sequence Length: {performance_analysis['overall_statistics']['avg_sequence_length']:.1f}")
        
        if 'best_performers' in performance_analysis:
            if 'best_new_format' in performance_analysis['best_performers']:
                best = performance_analysis['best_performers']['best_new_format']
                print(f"   Best Overall Trial: {best['trial']} (Fitness: {best['total_fitness']:.4f})")
        
        # Generate comprehensive report
        print("\\n9. Generating Reports...")
        report = comprehensive_analyzer.generate_comprehensive_report()
        
        # Create all visualizations
        comprehensive_analyzer.create_comprehensive_visualizations()
        
        # Compare with known CA sequences (if available)
        print("\\n10. Comparing with Known Carbonic Anhydrase Sequences...")
        target_sequence = proteins[main_accession]['sequence']
        comparison_results = co2_analyzer.compare_with_known_ca(
            best_sequence, 
            [target_sequence]
        )
        
        print(f"   Similarity to Target: {comparison_results['best_match']:.4f}")
        
        # Summary and recommendations
        print("\\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"✅ Successfully optimized protein for CO2 absorption")
        print(f"✅ Generated optimized sequence with {len(best_sequence)} amino acids")
        print(f"✅ Achieved CO2 binding affinity score: {co2_analysis['overall_affinity']:.4f}")
        print(f"✅ Achieved catalytic efficiency score: {catalytic_analysis['overall_efficiency']:.4f}")
        print(f"✅ Created comprehensive analysis and visualizations")
        print(f"✅ Generated detailed reports in Results/{trial_name}/")
        
        print("\\nNEXT STEPS:")
        print("1. Review the generated visualizations and reports")
        print("2. Consider experimental validation of top sequences")
        print("3. Run additional trials with different parameters")
        print("4. Analyze binding site interactions in 3D viewer")
        
        # Show key file locations
        results_dir = Path(workspace_path) / "Results" / trial_name
        reports_dir = Path(workspace_path) / "Reports"
        
        print(f"\\nOUTPUT LOCATIONS:")
        print(f"📁 Trial Results: {results_dir}")
        print(f"📁 Comprehensive Reports: {reports_dir}")
        print(f"📊 Visualizations: Multiple PNG files in both directories")
        
        return {
            'optimizer': optimizer,
            'trial_results': trial_results,
            'co2_analysis': co2_analysis,
            'catalytic_analysis': catalytic_analysis,
            'performance_analysis': performance_analysis,
            'trial_name': trial_name
        }
        
    except Exception as e:
        print(f"\\n❌ Error in optimization workflow: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_analysis_features():
    """Demonstrate additional analysis features"""
    print("\\n" + "="*60)
    print("DEMONSTRATING ADVANCED ANALYSIS FEATURES")
    print("="*60)
    
    workspace_path = r"c:\Users\Gunjan Ghimire\Downloads\Testting code\Algae-Bioniformatics-Binding-Protein"
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(workspace_path)
    analyzer.load_all_trials()
    
    if analyzer.trial_data:
        print(f"\\n📊 Analyzing {len(analyzer.trial_data)} trials...")
        
        # Create comparison table
        comparison_df = analyzer.create_comprehensive_comparison()
        print("\\nTrial Comparison Table:")
        print(comparison_df[['Trial', 'Format', 'Total Fitness', 'Sequence Length']].to_string())
        
        # Sequence analysis
        sequence_analysis = analyzer.perform_sequence_analysis()
        print("\\nSequence Analysis:")
        print(f"   Average sequence diversity: {sequence_analysis.get('sequence_diversity', {}).get('mean_diversity', 'N/A')}")
        
        # Top amino acids
        aa_freq = sequence_analysis.get('amino_acid_frequency', {})
        if aa_freq:
            top_aa = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            print("   Top 5 amino acids:")
            for aa, freq in top_aa:
                print(f"     {aa}: {freq:.3f}")
    else:
        print("No trial data found for analysis")

def quick_co2_analysis_demo():
    """Quick demonstration of CO2 binding analysis"""
    print("\\n" + "="*60)
    print("CO2 BINDING ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Example sequences for analysis
    sequences = {
        "Original CA": "MRVAAALLALAVCANACSHVYFADSDLHAHGRRLTAAEGPTWNYNKGGSDWPGTCASGNK",
        "Optimized 1": "MHHEAALLALAVCANACSHVYFADSDLHAHGRRLTAAEGPTWNYNKGGSDWPGTCASGNK", 
        "Optimized 2": "MHHVAALLALAVCANACSHVYFADSDLHDHGRRLTAAEGPTWNYNKGGSDWPGTCASGNK"
    }
    
    analyzer = CO2BindingAnalyzer()
    
    print("\\n🔬 Analyzing CO2 binding potential...")
    
    for name, sequence in sequences.items():
        print(f"\\n{name}:")
        analysis = analyzer.predict_co2_binding_affinity(sequence)
        print(f"   Overall CO2 Affinity: {analysis['overall_affinity']:.4f}")
        print(f"   Zinc Binding Score: {analysis['zinc_binding_score']:.4f}")
        print(f"   Catalytic Score: {analysis['catalytic_score']:.4f}")
        
        # Count key residues
        his_count = sequence.count('H')
        de_count = sequence.count('D') + sequence.count('E')
        cys_count = sequence.count('C')
        
        print(f"   Key residues: H={his_count}, D+E={de_count}, C={cys_count}")

if __name__ == "__main__":
    print("Starting Algae Protein Optimization System...")
    
    # Run main optimization workflow
    results = main()
    
    if results:
        print("\\n✅ Main optimization completed successfully!")
        
        # Run additional demonstrations
        demonstrate_analysis_features()
        quick_co2_analysis_demo()
        
        print("\\n🎉 All demonstrations completed!")
        print("\\nThe system is ready for production use. You can now:")
        print("1. Run new optimization trials with different parameters")
        print("2. Analyze and compare results across trials")
        print("3. Generate detailed reports and visualizations")
        print("4. Explore protein structures in the 3D viewer")
        
    else:
        print("\\n❌ Optimization failed. Please check the error messages above.")
    
    print("\\n" + "="*60)
    print("Thank you for using the Algae Protein Optimization System!")
    print("="*60)