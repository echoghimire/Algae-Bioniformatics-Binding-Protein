"""
Enhanced Demo with Advanced Visualization
Demonstrates the complete system with comprehensive visual analysis
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Import our modules
from co2_binding_analyzer import CO2BindingAnalyzer
from enhanced_ga_protein_optimizer import EnhancedGeneticAlgorithm, ProteinAnalyzer
from advanced_visualizer import ProteinOptimizationVisualizer

def run_enhanced_demo_with_visualization():
    print("🧬 ENHANCED ALGAE PROTEIN OPTIMIZATION WITH ADVANCED VISUALIZATION")
    print("=" * 80)
    
    # Target sequence (from your carbonic anhydrase)
    target_sequence = "MRVAAALLALAVCANACSHVYFADSDLHAHGRRLTAAEGPTWNYNKGGSDWPGTCASGNK"
    
    print(f"Target Protein Sequence:")
    print(f"  {target_sequence}")
    print(f"  Length: {len(target_sequence)} amino acids")
    
    # Initialize all components
    print("\\n1. Initializing Analysis Components...")
    co2_analyzer = CO2BindingAnalyzer()
    visualizer = ProteinOptimizationVisualizer()
    
    # Analyze original sequence
    print("\\n2. Analyzing Original Sequence...")
    original_analysis = co2_analyzer.predict_co2_binding_affinity(target_sequence)
    print("   Original Sequence Analysis:")
    print(f"     CO2 Affinity: {original_analysis['overall_affinity']:.4f}")
    print(f"     Zinc Binding: {original_analysis['zinc_binding_score']:.4f}")
    print(f"     Catalytic Score: {original_analysis['catalytic_score']:.4f}")
    
    # Create original sequence visualization
    print("\\n3. Creating Original Sequence Visualization...")
    visualizer.visualize_sequence_analysis(
        target_sequence, 
        original_analysis,
        save_path="original_sequence_analysis.png"
    )
    
    # Run genetic algorithm optimization
    print("\\n4. Running Enhanced Genetic Algorithm...")
    
    config = {
        'population_size': 25,
        'generations': 40,  # Increased for better evolution tracking
        'sequence_length': 25,
        'mutation_rate': 0.15,
        'crossover_rate': 0.7,
        'elite_count': 3
    }
    
    print(f"   Configuration: {config}")
    
    # Initialize and run GA
    ga = EnhancedGeneticAlgorithm(target_sequence, config)
    results = ga.run_optimization()
    
    print("\\n5. Optimization Results:")
    best_sequence = results['best_sequence']
    best_fitness = results['best_fitness']
    
    print(f"   Best Optimized Sequence: {best_sequence}")
    print(f"   Fitness Scores:")
    print(f"     CO2 Affinity: {best_fitness[0]:.4f}")
    print(f"     Stability: {best_fitness[1]:.4f}")
    print(f"     Expression: {best_fitness[2]:.4f}")
    print(f"     Catalytic Efficiency: {best_fitness[3]:.4f}")
    print(f"     Total Fitness: {sum(best_fitness):.4f}")
    
    # Create evolution progress visualization
    print("\\n6. Creating Evolution Progress Visualizations...")
    
    # Static evolution plots
    visualizer.visualize_evolution_progress(
        results['generation_data'],
        save_path="evolution_progress.png",
        interactive=False
    )
    
    # Interactive evolution plots
    print("   Creating interactive evolution dashboard...")
    try:
        visualizer.visualize_evolution_progress(
            results['generation_data'],
            save_path="interactive_evolution",
            interactive=True
        )
    except Exception as e:
        print(f"   Note: Interactive plots require browser environment: {e}")
    
    # 3D fitness landscape
    print("\\n7. Creating 3D Fitness Landscape...")
    visualizer.create_3d_fitness_landscape(
        results['generation_data'],
        save_path="3d_fitness_landscape.png"
    )
    
    # Analyze optimized sequence
    print("\\n8. Analyzing Optimized Sequence...")
    optimized_analysis = co2_analyzer.predict_co2_binding_affinity(best_sequence)
    
    print("   Optimized Sequence Analysis:")
    print(f"     CO2 Affinity: {optimized_analysis['overall_affinity']:.4f}")
    print(f"     Zinc Binding: {optimized_analysis['zinc_binding_score']:.4f}")
    print(f"     Catalytic Score: {optimized_analysis['catalytic_score']:.4f}")
    
    # Create optimized sequence visualization
    print("\\n9. Creating Optimized Sequence Visualization...")
    visualizer.visualize_sequence_analysis(
        best_sequence, 
        optimized_analysis,
        save_path="optimized_sequence_analysis.png"
    )
    
    # Advanced CO2 binding analysis with visualization
    print("\\n10. Creating Advanced CO2 Binding Analysis...")
    try:
        co2_analyzer.visualize_sequence_analysis(
            best_sequence,
            save_path="co2_binding_analysis.png"
        )
    except Exception as e:
        print(f"   Note: CO2 analysis visualization: {e}")
    
    # Create animated evolution (if possible)
    print("\\n11. Creating Evolution Animation...")
    try:
        anim = visualizer.animate_evolution(
            results['generation_data'],
            save_path="evolution_animation"
        )
        print("   ✅ Evolution animation created successfully!")
    except Exception as e:
        print(f"   Note: Animation requires additional setup: {e}")
    
    # Comparison analysis
    print("\\n12. Improvement Analysis:")
    co2_improvement = optimized_analysis['overall_affinity'] - original_analysis['overall_affinity']
    zinc_improvement = optimized_analysis['zinc_binding_score'] - original_analysis['zinc_binding_score']
    catalytic_improvement = optimized_analysis['catalytic_score'] - original_analysis['catalytic_score']
    
    print(f"   CO2 Affinity Improvement: {co2_improvement:+.4f}")
    print(f"   Zinc Binding Improvement: {zinc_improvement:+.4f}")
    print(f"   Catalytic Score Improvement: {catalytic_improvement:+.4f}")
    
    # Key residue analysis
    original_his = target_sequence.count('H')
    original_de = target_sequence.count('D') + target_sequence.count('E')
    original_cys = target_sequence.count('C')
    
    opt_his = best_sequence.count('H')
    opt_de = best_sequence.count('D') + best_sequence.count('E')
    opt_cys = best_sequence.count('C')
    
    print("\\n13. Key Residue Comparison:")
    print(f"   Histidines (Zinc binding): {original_his} → {opt_his} ({opt_his - original_his:+d})")
    print(f"   D+E (Catalytic): {original_de} → {opt_de} ({opt_de - original_de:+d})")
    print(f"   Cysteines (Structural): {original_cys} → {opt_cys} ({opt_cys - original_cys:+d})")
    
    # Generate suggestions for further optimization
    suggestions = co2_analyzer.generate_optimization_suggestions(best_sequence)
    if suggestions:
        print("\\n14. Further Optimization Suggestions:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"    {i}. {suggestion}")
    
    # Create comparison visualization
    print("\\n15. Creating Sequence Comparison Visualization...")
    
    # Simple comparison plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('🧬 Original vs Optimized Sequence Comparison', fontsize=16, fontweight='bold')
    
    # Original sequence analysis
    axes[0].bar(['CO2 Affinity', 'Zinc Binding', 'Catalytic Score'], 
               [original_analysis['overall_affinity'], 
                original_analysis['zinc_binding_score'],
                original_analysis['catalytic_score']], 
               color=['red', 'blue', 'green'], alpha=0.7)
    axes[0].set_title('Original Sequence')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Optimized sequence analysis
    axes[1].bar(['CO2 Affinity', 'Zinc Binding', 'Catalytic Score'], 
               [optimized_analysis['overall_affinity'], 
                optimized_analysis['zinc_binding_score'],
                optimized_analysis['catalytic_score']], 
               color=['red', 'blue', 'green'], alpha=0.7)
    axes[1].set_title('Optimized Sequence')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("sequence_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\n" + "="*80)
    print("🎉 ENHANCED VISUALIZATION DEMONSTRATION COMPLETE!")
    print("="*80)
    
    print("\\n📊 GENERATED VISUALIZATIONS:")
    print("   📈 evolution_progress.png - Static evolution plots")
    print("   🌐 interactive_evolution.html - Interactive dashboard")
    print("   🎯 3d_fitness_landscape.png - 3D fitness visualization")
    print("   🧬 original_sequence_analysis.png - Original sequence analysis")
    print("   ⚡ optimized_sequence_analysis.png - Optimized sequence analysis")
    print("   🔬 co2_binding_analysis.png - CO2 binding analysis")
    print("   📊 sequence_comparison.png - Side-by-side comparison")
    print("   🎬 evolution_animation.gif - Animated evolution (if supported)")
    
    print("\\n🚀 KEY ACHIEVEMENTS:")
    if co2_improvement > 0:
        print(f"   ✅ Enhanced CO2 binding affinity (+{co2_improvement:.3f})")
    if zinc_improvement >= 0:
        print(f"   ✅ Maintained/improved zinc binding (+{zinc_improvement:.3f})")
    if catalytic_improvement > 0:
        print(f"   ✅ Better catalytic potential (+{catalytic_improvement:.3f})")
    
    print(f"\\n🧪 SCIENTIFIC INSIGHTS:")
    print(f"   • Optimized sequence has {opt_his} histidines for enhanced zinc coordination")
    print(f"   • {opt_de} catalytic residues (D+E) for improved CO2 processing")
    print(f"   • Total fitness improved from {sum(results['generation_data'][0]['best_fitness']):.3f} to {sum(best_fitness):.3f}")
    
    print("\\n🔬 NEXT STEPS FOR RESEARCH:")
    print("   1. Experimental validation of top-performing sequences")
    print("   2. Structural modeling with molecular dynamics simulations")
    print("   3. Expression testing in algae or bacterial systems")
    print("   4. CO2 absorption rate measurements")
    print("   5. Enzyme kinetics analysis (Km, Vmax, kcat)")
    
    return {
        'original_analysis': original_analysis,
        'optimized_analysis': optimized_analysis,
        'best_sequence': best_sequence,
        'best_fitness': best_fitness,
        'evolution_data': results['generation_data'],
        'improvements': {
            'co2_affinity': co2_improvement,
            'zinc_binding': zinc_improvement,
            'catalytic': catalytic_improvement
        }
    }

if __name__ == "__main__":
    try:
        print("🚀 Starting Enhanced Visualization Demo...")
        results = run_enhanced_demo_with_visualization()
        print("\\n✅ Demo completed successfully!")
        print("\\nAll visualizations have been saved to the current directory.")
        print("Open the generated PNG files to view the detailed analysis!")
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()