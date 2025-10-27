"""
Simple demonstration of the enhanced genetic algorithm system
"""

import sys
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.append('.')

# Import our modules
from co2_binding_analyzer import CO2BindingAnalyzer
from enhanced_ga_protein_optimizer import EnhancedGeneticAlgorithm, ProteinAnalyzer

def run_simple_demo():
    print("🧬 ALGAE PROTEIN OPTIMIZATION - SIMPLE DEMO")
    print("=" * 60)
    
    # Target sequence (from your carbonic anhydrase)
    target_sequence = "MRVAAALLALAVCANACSHVYFADSDLHAHGRRLTAAEGPTWNYNKGGSDWPGTCASGNK"
    
    print(f"Target Protein Sequence:")
    print(f"  {target_sequence}")
    print(f"  Length: {len(target_sequence)} amino acids")
    
    # Initialize analyzers
    print("\n1. Analyzing Original Sequence...")
    co2_analyzer = CO2BindingAnalyzer()
    
    # Analyze original sequence
    original_analysis = co2_analyzer.predict_co2_binding_affinity(target_sequence)
    print("   Original Sequence Analysis:")
    print(f"     CO2 Affinity: {original_analysis['overall_affinity']:.4f}")
    print(f"     Zinc Binding: {original_analysis['zinc_binding_score']:.4f}")
    print(f"     Catalytic Score: {original_analysis['catalytic_score']:.4f}")
    
    # Count key residues in original
    his_count = target_sequence.count('H')
    de_count = target_sequence.count('D') + target_sequence.count('E')
    cys_count = target_sequence.count('C')
    print(f"     Key residues: H={his_count}, D+E={de_count}, C={cys_count}")
    
    # Run genetic algorithm optimization
    print("\n2. Running Genetic Algorithm Optimization...")
    
    # Configuration for quick demo
    config = {
        'population_size': 20,
        'generations': 30,
        'sequence_length': 25,  # Shorter for demo
        'mutation_rate': 0.15,
        'crossover_rate': 0.7,
        'elite_count': 3
    }
    
    print(f"   Configuration: {config}")
    
    # Initialize and run GA
    ga = EnhancedGeneticAlgorithm(target_sequence, config)
    results = ga.run_optimization()
    
    print("\n3. Optimization Results:")
    best_sequence = results['best_sequence']
    best_fitness = results['best_fitness']
    
    print(f"   Best Optimized Sequence: {best_sequence}")
    print(f"   Fitness Scores:")
    print(f"     CO2 Affinity: {best_fitness[0]:.4f}")
    print(f"     Stability: {best_fitness[1]:.4f}")
    print(f"     Expression: {best_fitness[2]:.4f}")
    print(f"     Catalytic Efficiency: {best_fitness[3]:.4f}")
    print(f"     Total Fitness: {sum(best_fitness):.4f}")
    
    # Analyze optimized sequence
    print("\n4. Analyzing Optimized Sequence...")
    optimized_analysis = co2_analyzer.predict_co2_binding_affinity(best_sequence)
    
    print("   Optimized Sequence Analysis:")
    print(f"     CO2 Affinity: {optimized_analysis['overall_affinity']:.4f}")
    print(f"     Zinc Binding: {optimized_analysis['zinc_binding_score']:.4f}")
    print(f"     Catalytic Score: {optimized_analysis['catalytic_score']:.4f}")
    
    # Count key residues in optimized
    opt_his = best_sequence.count('H')
    opt_de = best_sequence.count('D') + best_sequence.count('E')
    opt_cys = best_sequence.count('C')
    print(f"     Key residues: H={opt_his}, D+E={opt_de}, C={opt_cys}")
    
    # Comparison
    print("\n5. Improvement Analysis:")
    co2_improvement = optimized_analysis['overall_affinity'] - original_analysis['overall_affinity']
    zinc_improvement = optimized_analysis['zinc_binding_score'] - original_analysis['zinc_binding_score']
    catalytic_improvement = optimized_analysis['catalytic_score'] - original_analysis['catalytic_score']
    
    print(f"   CO2 Affinity Improvement: {co2_improvement:+.4f}")
    print(f"   Zinc Binding Improvement: {zinc_improvement:+.4f}")
    print(f"   Catalytic Score Improvement: {catalytic_improvement:+.4f}")
    
    # Suggestions
    suggestions = co2_analyzer.generate_optimization_suggestions(best_sequence)
    if suggestions:
        print("\n6. Further Optimization Suggestions:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"   {i}. {suggestion}")
    
    print("\n" + "="*60)
    print("✅ DEMONSTRATION COMPLETE!")
    print("\nKey Improvements Achieved:")
    if co2_improvement > 0:
        print(f"  ✓ Enhanced CO2 binding affinity (+{co2_improvement:.3f})")
    if zinc_improvement > 0:
        print(f"  ✓ Improved zinc binding capacity (+{zinc_improvement:.3f})")
    if catalytic_improvement > 0:
        print(f"  ✓ Better catalytic potential (+{catalytic_improvement:.3f})")
    
    print(f"\nOptimized sequence shows {opt_his} histidines for zinc binding")
    print(f"and {opt_de} catalytic residues (D+E) for CO2 processing.")
    
    print("\nNext Steps:")
    print("  1. Run longer optimizations with more generations")
    print("  2. Try different sequence lengths and parameters")
    print("  3. Use the full analysis system for detailed reports")
    print("  4. Validate top sequences experimentally")
    
    return {
        'original_analysis': original_analysis,
        'optimized_analysis': optimized_analysis,
        'best_sequence': best_sequence,
        'best_fitness': best_fitness,
        'improvement': {
            'co2_affinity': co2_improvement,
            'zinc_binding': zinc_improvement,
            'catalytic': catalytic_improvement
        }
    }

if __name__ == "__main__":
    try:
        results = run_simple_demo()
        print(f"\n🎉 Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()