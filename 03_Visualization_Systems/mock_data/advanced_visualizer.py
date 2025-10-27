"""
Enhanced Visualization System for Genetic Algorithm Protein Optimization
Provides comprehensive visual analysis of optimization results and protein properties
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import io
import base64

class ProteinOptimizationVisualizer:
    """Advanced visualizer for protein optimization results"""
    
    def __init__(self):
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color schemes for different visualizations
        self.fitness_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        self.aa_colors = {
            'H': '#FF4444',  # Histidine - red (zinc binding)
            'D': '#4444FF',  # Aspartic acid - blue (catalytic)
            'E': '#4444FF',  # Glutamic acid - blue (catalytic) 
            'C': '#FFD700',  # Cysteine - gold (disulfide bonds)
            'T': '#00FF00',  # Threonine - green (substrate binding)
            'S': '#00FF00',  # Serine - green (substrate binding)
            'Y': '#FF8C00',  # Tyrosine - orange (aromatic)
            'W': '#FF8C00',  # Tryptophan - orange (aromatic)
            'F': '#FF8C00',  # Phenylalanine - orange (aromatic)
            'default': '#CCCCCC'  # Other amino acids - gray
        }
        
    def visualize_evolution_progress(self, generation_data: List[Dict], 
                                   save_path: Optional[str] = None, 
                                   interactive: bool = False) -> None:
        """Create comprehensive evolution progress visualization"""
        
        if interactive:
            self._create_interactive_evolution_plot(generation_data, save_path)
        else:
            self._create_static_evolution_plot(generation_data, save_path)
    
    def _create_static_evolution_plot(self, generation_data: List[Dict], save_path: Optional[str] = None):
        """Create static evolution progress plots"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🧬 Genetic Algorithm Evolution Progress', fontsize=20, fontweight='bold')
        
        generations = [d['generation'] for d in generation_data]
        
        # Extract fitness components
        co2_affinity = [d['best_fitness'][0] for d in generation_data]
        stability = [d['best_fitness'][1] for d in generation_data]
        expression = [d['best_fitness'][2] for d in generation_data]
        catalytic = [d['best_fitness'][3] for d in generation_data]
        total_fitness = [sum(d['best_fitness']) for d in generation_data]
        diversity = [d['population_diversity'] for d in generation_data]
        
        # 1. Fitness components over time
        axes[0, 0].plot(generations, co2_affinity, 'o-', color=self.fitness_colors[0], 
                       linewidth=3, markersize=6, label='CO2 Affinity')
        axes[0, 0].plot(generations, stability, 's-', color=self.fitness_colors[1], 
                       linewidth=3, markersize=6, label='Stability')
        axes[0, 0].plot(generations, expression, '^-', color=self.fitness_colors[2], 
                       linewidth=3, markersize=6, label='Expression')
        axes[0, 0].plot(generations, catalytic, 'd-', color=self.fitness_colors[3], 
                       linewidth=3, markersize=6, label='Catalytic Efficiency')
        
        axes[0, 0].set_title('Fitness Components Evolution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Total fitness with confidence interval
        axes[0, 1].plot(generations, total_fitness, 'o-', color='darkblue', 
                       linewidth=4, markersize=8, label='Total Fitness')
        axes[0, 1].fill_between(generations, 
                               [f - 0.1 for f in total_fitness], 
                               [f + 0.1 for f in total_fitness], 
                               alpha=0.2, color='darkblue')
        axes[0, 1].set_title('Total Fitness Evolution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Total Fitness')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Population diversity
        axes[0, 2].plot(generations, diversity, 'o-', color='purple', 
                       linewidth=3, markersize=6)
        axes[0, 2].set_title('Population Diversity', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Generation')
        axes[0, 2].set_ylabel('Diversity Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Improvement rate
        improvement_rates = []
        for i in range(1, len(total_fitness)):
            rate = total_fitness[i] - total_fitness[i-1]
            improvement_rates.append(rate)
        
        axes[1, 0].bar(generations[1:], improvement_rates, alpha=0.7, 
                      color=['green' if x > 0 else 'red' for x in improvement_rates])
        axes[1, 0].set_title('Generation-to-Generation Improvement', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Fitness Change')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Fitness correlation heatmap
        fitness_data = pd.DataFrame({
            'CO2 Affinity': co2_affinity,
            'Stability': stability,
            'Expression': expression,
            'Catalytic': catalytic
        })
        
        corr_matrix = fitness_data.corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[1, 1].set_yticklabels(corr_matrix.columns)
        axes[1, 1].set_title('Fitness Components Correlation', fontsize=14, fontweight='bold')
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha='center', va='center', color='white', fontweight='bold')
        
        # 6. Performance summary radar chart
        if len(generation_data) > 0:
            final_fitness = generation_data[-1]['best_fitness']
            categories = ['CO2\\nAffinity', 'Stability', 'Expression', 'Catalytic\\nEfficiency']
            values = list(final_fitness) + [final_fitness[0]]  # Close the polygon
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax_radar = plt.subplot(2, 3, 6, projection='polar')
            ax_radar.plot(angles, values, 'o-', linewidth=3, color='darkgreen', markersize=8)
            ax_radar.fill(angles, values, alpha=0.25, color='darkgreen')
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories, fontsize=12)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('Final Performance Profile', fontsize=14, fontweight='bold', pad=20)
            ax_radar.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evolution plot saved to: {save_path}")
        
        plt.show()
    
    def _create_interactive_evolution_plot(self, generation_data: List[Dict], save_path: Optional[str] = None):
        """Create interactive evolution progress plots using Plotly"""
        
        generations = [d['generation'] for d in generation_data]
        co2_affinity = [d['best_fitness'][0] for d in generation_data]
        stability = [d['best_fitness'][1] for d in generation_data]
        expression = [d['best_fitness'][2] for d in generation_data]
        catalytic = [d['best_fitness'][3] for d in generation_data]
        diversity = [d['population_diversity'] for d in generation_data]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fitness Components Evolution', 'Total Fitness Evolution',
                          'Population Diversity', 'Fitness Components Correlation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "heatmap"}]]
        )
        
        # Plot 1: Fitness components
        fig.add_trace(go.Scatter(x=generations, y=co2_affinity, mode='lines+markers',
                               name='CO2 Affinity', line=dict(color='red', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=generations, y=stability, mode='lines+markers',
                               name='Stability', line=dict(color='blue', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=generations, y=expression, mode='lines+markers',
                               name='Expression', line=dict(color='green', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=generations, y=catalytic, mode='lines+markers',
                               name='Catalytic', line=dict(color='orange', width=3)), row=1, col=1)
        
        # Plot 2: Total fitness
        total_fitness = [sum(d['best_fitness']) for d in generation_data]
        fig.add_trace(go.Scatter(x=generations, y=total_fitness, mode='lines+markers',
                               name='Total Fitness', line=dict(color='darkblue', width=4),
                               marker=dict(size=8)), row=1, col=2)
        
        # Plot 3: Diversity
        fig.add_trace(go.Scatter(x=generations, y=diversity, mode='lines+markers',
                               name='Diversity', line=dict(color='purple', width=3),
                               marker=dict(size=6)), row=2, col=1)
        
        # Plot 4: Correlation heatmap
        fitness_data = pd.DataFrame({
            'CO2': co2_affinity,
            'Stability': stability,
            'Expression': expression,
            'Catalytic': catalytic
        })
        corr_matrix = fitness_data.corr()
        
        fig.add_trace(go.Heatmap(z=corr_matrix.values,
                               x=corr_matrix.columns,
                               y=corr_matrix.columns,
                               colorscale='RdBu',
                               zmid=0,
                               text=corr_matrix.round(2).values,
                               texttemplate="%{text}",
                               textfont={"size": 12}), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="🧬 Interactive Genetic Algorithm Evolution Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Generation", row=1, col=1)
        fig.update_xaxes(title_text="Generation", row=1, col=2)
        fig.update_xaxes(title_text="Generation", row=2, col=1)
        fig.update_yaxes(title_text="Fitness Score", row=1, col=1)
        fig.update_yaxes(title_text="Total Fitness", row=1, col=2)
        fig.update_yaxes(title_text="Diversity", row=2, col=1)
        
        if save_path:
            # Save as HTML
            html_path = save_path.replace('.png', '.html') if save_path.endswith('.png') else save_path + '.html'
            fig.write_html(html_path)
            print(f"Interactive plot saved to: {html_path}")
        
        # Show the plot
        fig.show()
    
    def visualize_sequence_analysis(self, sequence: str, analysis_data: Dict, 
                                  save_path: Optional[str] = None) -> None:
        """Create comprehensive sequence analysis visualization"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'🧬 Protein Sequence Analysis - Length: {len(sequence)}', 
                     fontsize=20, fontweight='bold')
        
        # 1. Amino acid composition bar chart
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        aa_names = list(aa_counts.keys())
        aa_frequencies = [count / len(sequence) for count in aa_counts.values()]
        colors = [self.aa_colors.get(aa, self.aa_colors['default']) for aa in aa_names]
        
        bars = axes[0, 0].bar(aa_names, aa_frequencies, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('Amino Acid Composition', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Amino Acid')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, freq in zip(bars, aa_frequencies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{freq:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sequence visualization with color coding
        self._plot_sequence_visualization(axes[0, 1], sequence)
        
        # 3. Physicochemical properties radar chart
        if 'detailed_analysis' in analysis_data:
            self._plot_properties_radar(axes[0, 2], sequence)
        
        # 4. Position-wise properties
        self._plot_position_properties(axes[1, 0], sequence)
        
        # 5. Motif analysis
        self._plot_motif_analysis(axes[1, 1], sequence, analysis_data)
        
        # 6. Binding site prediction
        self._plot_binding_sites(axes[1, 2], sequence)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sequence analysis saved to: {save_path}")
        
        plt.show()
    
    def _plot_sequence_visualization(self, ax, sequence: str):
        """Plot sequence as colored blocks"""
        # Create a visual representation of the sequence
        seq_matrix = np.zeros((5, len(sequence)))
        
        for i, aa in enumerate(sequence):
            if aa in ['H']:  # Zinc binding
                seq_matrix[0, i] = 1
            elif aa in ['D', 'E']:  # Catalytic
                seq_matrix[1, i] = 1
            elif aa in ['C']:  # Structural
                seq_matrix[2, i] = 1
            elif aa in ['T', 'S', 'Y', 'W', 'F']:  # Substrate binding
                seq_matrix[3, i] = 1
            else:  # Other
                seq_matrix[4, i] = 1
        
        im = ax.imshow(seq_matrix, cmap='Set1', aspect='auto')
        ax.set_title('Sequence Function Map', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position')
        ax.set_yticks(range(5))
        ax.set_yticklabels(['Zinc Binding', 'Catalytic', 'Structural', 'Substrate Binding', 'Other'])
        
        # Add sequence letters at the bottom
        for i, aa in enumerate(sequence):
            if i % 2 == 0:  # Show every other letter to avoid crowding
                ax.text(i, -0.7, aa, ha='center', va='center', fontsize=8, fontweight='bold')
    
    def _plot_properties_radar(self, ax, sequence: str):
        """Plot physicochemical properties radar chart"""
        # Calculate average properties
        hydrophobic_count = sum(1 for aa in sequence if aa in ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y'])
        polar_count = sum(1 for aa in sequence if aa in ['S', 'T', 'N', 'Q', 'Y'])
        charged_count = sum(1 for aa in sequence if aa in ['D', 'E', 'K', 'R', 'H'])
        aromatic_count = sum(1 for aa in sequence if aa in ['F', 'W', 'Y'])
        small_count = sum(1 for aa in sequence if aa in ['A', 'G', 'S'])
        
        total = len(sequence)
        properties = [
            hydrophobic_count / total,
            polar_count / total,
            charged_count / total,
            aromatic_count / total,
            small_count / total
        ]
        
        labels = ['Hydrophobic', 'Polar', 'Charged', 'Aromatic', 'Small']
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        properties += properties[:1]  # Close the polygon
        angles += angles[:1]
        
        ax.remove()  # Remove the regular axes
        ax = plt.subplot(2, 3, 3, projection='polar')
        ax.plot(angles, properties, 'o-', linewidth=3, color='darkblue', markersize=8)
        ax.fill(angles, properties, alpha=0.25, color='darkblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title('Physicochemical Properties', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
    
    def _plot_position_properties(self, ax, sequence: str):
        """Plot position-wise properties"""
        positions = list(range(len(sequence)))
        
        # Define property functions
        co2_affinity = []
        hydrophobicity = []
        
        for aa in sequence:
            # CO2 binding affinity (simplified)
            if aa == 'H':
                co2_affinity.append(0.9)
            elif aa in ['D', 'E']:
                co2_affinity.append(0.8)
            elif aa in ['C', 'T', 'S']:
                co2_affinity.append(0.6)
            else:
                co2_affinity.append(0.3)
            
            # Hydrophobicity
            if aa in ['A', 'V', 'L', 'I', 'M', 'F', 'W']:
                hydrophobicity.append(1.0)
            elif aa in ['Y', 'P']:
                hydrophobicity.append(0.5)
            else:
                hydrophobicity.append(0.0)
        
        ax.plot(positions, co2_affinity, 'o-', color='red', linewidth=2, 
               markersize=4, label='CO2 Affinity', alpha=0.7)
        ax.plot(positions, hydrophobicity, 's-', color='blue', linewidth=2, 
               markersize=4, label='Hydrophobicity', alpha=0.7)
        
        ax.set_title('Position-wise Properties', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position')
        ax.set_ylabel('Property Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_motif_analysis(self, ax, sequence: str, analysis_data: Dict):
        """Plot motif analysis"""
        # Common CO2 binding motifs
        motifs = ['HHH', 'HXH', 'DXE', 'CXC', 'THR']
        motif_counts = []
        
        for motif in motifs:
            if 'X' in motif:
                # Pattern matching for variable motifs
                count = 0
                for i in range(len(sequence) - len(motif) + 1):
                    match = True
                    for j, char in enumerate(motif):
                        if char != 'X' and sequence[i + j] != char:
                            match = False
                            break
                    if match:
                        count += 1
                motif_counts.append(count)
            else:
                motif_counts.append(sequence.count(motif))
        
        bars = ax.bar(motifs, motif_counts, color=['red', 'orange', 'blue', 'gold', 'green'], alpha=0.7)
        ax.set_title('CO2 Binding Motifs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Motif')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, motif_counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       str(count), ha='center', va='bottom', fontweight='bold')
    
    def _plot_binding_sites(self, ax, sequence: str):
        """Plot predicted binding sites"""
        positions = list(range(len(sequence)))
        binding_scores = []
        
        # Simple binding site prediction based on residue types
        for i, aa in enumerate(sequence):
            score = 0
            
            # Zinc binding residues
            if aa == 'H':
                score += 0.8
            elif aa == 'C':
                score += 0.3
            
            # Catalytic residues
            if aa in ['D', 'E']:
                score += 0.6
            
            # Substrate binding
            if aa in ['T', 'Y', 'W']:
                score += 0.4
            
            # Context-dependent scoring (neighboring residues)
            if i > 0 and sequence[i-1] in ['H', 'D', 'E']:
                score += 0.2
            if i < len(sequence) - 1 and sequence[i+1] in ['H', 'D', 'E']:
                score += 0.2
            
            binding_scores.append(min(score, 1.0))
        
        # Plot binding scores
        colors = ['red' if score > 0.6 else 'orange' if score > 0.3 else 'lightblue' 
                 for score in binding_scores]
        
        bars = ax.bar(positions, binding_scores, color=colors, alpha=0.7, width=0.8)
        ax.set_title('Predicted Binding Sites', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position')
        ax.set_ylabel('Binding Score')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add threshold line
        ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='High Affinity')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate Affinity')
        ax.legend()
    
    def create_3d_fitness_landscape(self, generation_data: List[Dict], 
                                  save_path: Optional[str] = None) -> None:
        """Create 3D fitness landscape visualization"""
        
        fig = plt.figure(figsize=(15, 12))
        
        # Extract data
        generations = [d['generation'] for d in generation_data]
        co2_affinity = [d['best_fitness'][0] for d in generation_data]
        stability = [d['best_fitness'][1] for d in generation_data]
        expression = [d['best_fitness'][2] for d in generation_data]
        catalytic = [d['best_fitness'][3] for d in generation_data]
        
        # Create 3D plots
        
        # Plot 1: Evolution trajectory in 3D space
        ax1 = fig.add_subplot(221, projection='3d')
        scatter = ax1.scatter(co2_affinity, stability, expression, c=generations, 
                            cmap='viridis', s=100, alpha=0.8)
        ax1.plot(co2_affinity, stability, expression, 'r-', alpha=0.5, linewidth=2)
        ax1.set_xlabel('CO2 Affinity')
        ax1.set_ylabel('Stability')
        ax1.set_zlabel('Expression')
        ax1.set_title('3D Evolution Trajectory')
        plt.colorbar(scatter, ax=ax1, label='Generation', shrink=0.8)
        
        # Plot 2: Fitness surface
        ax2 = fig.add_subplot(222, projection='3d')
        
        # Create a grid for surface plot
        x = np.linspace(min(co2_affinity), max(co2_affinity), 20)
        y = np.linspace(min(stability), max(stability), 20)
        X, Y = np.meshgrid(x, y)
        
        # Interpolate expression values
        from scipy.interpolate import griddata
        points = list(zip(co2_affinity, stability))
        Z = griddata(points, expression, (X, Y), method='cubic', fill_value=0)
        
        surf = ax2.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
        ax2.scatter(co2_affinity, stability, expression, c='red', s=50)
        ax2.set_xlabel('CO2 Affinity')
        ax2.set_ylabel('Stability')
        ax2.set_zlabel('Expression')
        ax2.set_title('Fitness Landscape Surface')
        
        # Plot 3: Multi-objective Pareto front approximation
        ax3 = fig.add_subplot(223, projection='3d')
        
        # Take the best solutions from each generation
        pareto_points = []
        for data in generation_data[-10:]:  # Last 10 generations
            pareto_points.append(data['best_fitness'][:3])
        
        if pareto_points:
            pareto_array = np.array(pareto_points)
            ax3.scatter(pareto_array[:, 0], pareto_array[:, 1], pareto_array[:, 2], 
                       c='red', s=100, alpha=0.8, label='Best Solutions')
            
            # Connect points to show progression
            ax3.plot(pareto_array[:, 0], pareto_array[:, 1], pareto_array[:, 2], 
                    'b-', alpha=0.5, linewidth=2)
        
        ax3.set_xlabel('CO2 Affinity')
        ax3.set_ylabel('Stability')
        ax3.set_zlabel('Expression')
        ax3.set_title('Pareto Front Approximation')
        ax3.legend()
        
        # Plot 4: 4D visualization using color and size
        ax4 = fig.add_subplot(224, projection='3d')
        
        # Use catalytic efficiency for color and total fitness for size
        total_fitness = [sum(d['best_fitness']) for d in generation_data]
        sizes = [50 + f * 200 for f in total_fitness]  # Scale sizes
        
        scatter4 = ax4.scatter(co2_affinity, stability, expression, 
                             c=catalytic, s=sizes, cmap='plasma', alpha=0.7)
        ax4.set_xlabel('CO2 Affinity')
        ax4.set_ylabel('Stability')
        ax4.set_zlabel('Expression')
        ax4.set_title('4D Fitness Visualization\\n(Color=Catalytic, Size=Total)')
        plt.colorbar(scatter4, ax=ax4, label='Catalytic Efficiency', shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D fitness landscape saved to: {save_path}")
        
        plt.show()
    
    def create_comparison_dashboard(self, trial_results: List[Dict], 
                                  save_path: Optional[str] = None) -> None:
        """Create a comprehensive comparison dashboard for multiple trials"""
        
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('🧬 Multi-Trial Comparison Dashboard', fontsize=24, fontweight='bold')
        
        # Extract data from trials
        trial_names = [trial.get('trial_name', f'Trial {i}') for i, trial in enumerate(trial_results)]
        
        # Collect fitness data
        co2_scores = []
        stability_scores = []
        expression_scores = []
        catalytic_scores = []
        total_scores = []
        sequence_lengths = []
        
        for trial in trial_results:
            if 'results' in trial and 'best_fitness' in trial['results']:
                fitness = trial['results']['best_fitness']
                co2_scores.append(fitness[0])
                stability_scores.append(fitness[1])
                expression_scores.append(fitness[2])
                catalytic_scores.append(fitness[3])
                total_scores.append(sum(fitness))
                sequence_lengths.append(len(trial['results']['best_sequence']))
        
        # 1. Fitness comparison bar chart
        x_pos = np.arange(len(trial_names))
        width = 0.2
        
        axes[0, 0].bar(x_pos - 1.5*width, co2_scores, width, label='CO2 Affinity', 
                      color=self.fitness_colors[0], alpha=0.8)
        axes[0, 0].bar(x_pos - 0.5*width, stability_scores, width, label='Stability', 
                      color=self.fitness_colors[1], alpha=0.8)
        axes[0, 0].bar(x_pos + 0.5*width, expression_scores, width, label='Expression', 
                      color=self.fitness_colors[2], alpha=0.8)
        axes[0, 0].bar(x_pos + 1.5*width, catalytic_scores, width, label='Catalytic', 
                      color=self.fitness_colors[3], alpha=0.8)
        
        axes[0, 0].set_title('Fitness Components Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Fitness Score')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(trial_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Continue with other comparison plots...
        # (Additional plots would be implemented here)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison dashboard saved to: {save_path}")
        
        plt.show()
    
    def animate_evolution(self, generation_data: List[Dict], 
                         save_path: Optional[str] = None) -> None:
        """Create an animated visualization of the evolution process"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('🧬 Animated Evolution Process', fontsize=16, fontweight='bold')
        
        generations = [d['generation'] for d in generation_data]
        max_gen = max(generations)
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Current generation data
            current_data = generation_data[:frame+1]
            current_gens = [d['generation'] for d in current_data]
            
            # Plot 1: Fitness evolution
            co2_vals = [d['best_fitness'][0] for d in current_data]
            stability_vals = [d['best_fitness'][1] for d in current_data]
            expression_vals = [d['best_fitness'][2] for d in current_data]
            catalytic_vals = [d['best_fitness'][3] for d in current_data]
            
            ax1.plot(current_gens, co2_vals, 'r-o', label='CO2 Affinity', linewidth=2)
            ax1.plot(current_gens, stability_vals, 'b-s', label='Stability', linewidth=2)
            ax1.plot(current_gens, expression_vals, 'g-^', label='Expression', linewidth=2)
            ax1.plot(current_gens, catalytic_vals, 'orange', marker='d', label='Catalytic', linewidth=2)
            
            ax1.set_xlim(0, max_gen)
            ax1.set_ylim(0, 1.2)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness Score')
            ax1.set_title(f'Evolution Progress (Generation {frame})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Current best sequence visualization
            if frame < len(generation_data):
                current_best = generation_data[frame]['best_sequence']
                self._plot_sequence_bars(ax2, current_best)
                ax2.set_title(f'Best Sequence at Generation {frame}')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(generation_data), 
                           interval=200, repeat=True, blit=False)
        
        if save_path:
            # Save as GIF
            gif_path = save_path.replace('.png', '.gif') if save_path.endswith('.png') else save_path + '.gif'
            anim.save(gif_path, writer='pillow', fps=5)
            print(f"Animation saved to: {gif_path}")
        
        plt.show()
        return anim
    
    def _plot_sequence_bars(self, ax, sequence: str):
        """Helper function to plot sequence as bars"""
        positions = list(range(len(sequence)))
        heights = []
        colors = []
        
        for aa in sequence:
            if aa == 'H':
                heights.append(1.0)
                colors.append('red')
            elif aa in ['D', 'E']:
                heights.append(0.8)
                colors.append('blue')
            elif aa == 'C':
                heights.append(0.6)
                colors.append('gold')
            else:
                heights.append(0.3)
                colors.append('lightgray')
        
        ax.bar(positions, heights, color=colors, alpha=0.7)
        ax.set_xlabel('Position')
        ax.set_ylabel('Importance')
        ax.set_ylim(0, 1.2)
    
    def create_comprehensive_dashboard(self, optimization_data: Dict, 
                                     original_sequence: str, 
                                     optimized_sequence: str,
                                     save_dir: str = "./visualizations") -> List[str]:
        """Create all visualizations and save to specified directory"""
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        generated_files = []
        
        try:
            print(f"Creating visualizations in: {save_path}")
            
            # 1. Evolution progress plot
            evolution_file = save_path / "evolution_progress.png"
            self.visualize_evolution_progress(
                optimization_data['generation_data'], 
                str(evolution_file)
            )
            generated_files.append("evolution_progress.png")
            
            # 2. Interactive evolution plot
            interactive_file = save_path / "interactive_evolution.html"
            self.visualize_evolution_progress(
                optimization_data['generation_data'], 
                str(interactive_file),
                interactive=True
            )
            generated_files.append("interactive_evolution.html")
            
            # 3. 3D fitness landscape
            landscape_file = save_path / "3d_fitness_landscape.png"
            self.create_3d_fitness_landscape(
                optimization_data['fitness_history'],
                optimization_data['generation_data'],
                str(landscape_file)
            )
            generated_files.append("3d_fitness_landscape.png")
            
            # 4. Original sequence analysis
            original_file = save_path / "original_sequence_analysis.png"
            self.visualize_protein_properties(original_sequence, str(original_file))
            generated_files.append("original_sequence_analysis.png")
            
            # 5. Optimized sequence analysis
            optimized_file = save_path / "optimized_sequence_analysis.png"
            self.visualize_protein_properties(optimized_sequence, str(optimized_file))
            generated_files.append("optimized_sequence_analysis.png")
            
            # 6. CO2 binding analysis (with error handling)
            try:
                from co2_binding_analyzer import CO2BindingAnalyzer
                analyzer = CO2BindingAnalyzer()
                
                co2_file = save_path / "co2_binding_analysis.png"
                analyzer.visualize_binding_analysis(
                    original_sequence, optimized_sequence, str(co2_file)
                )
                generated_files.append("co2_binding_analysis.png")
            except Exception as e:
                print(f"Skipping CO2 analysis due to error: {e}")
            
            # 7. Sequence comparison
            comparison_file = save_path / "sequence_comparison.png"
            self.create_sequence_comparison(
                original_sequence, optimized_sequence, str(comparison_file)
            )
            generated_files.append("sequence_comparison.png")
            
            # 8. Evolution animation
            try:
                animation_file = save_path / "evolution_animation.gif"
                self.create_evolution_animation(
                    optimization_data['generation_data'], str(animation_file)
                )
                generated_files.append("evolution_animation.gif")
            except Exception as e:
                print(f"Skipping animation due to error: {e}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
        
        print(f"Generated {len(generated_files)} visualization files")
        return generated_files

def main():
    """Demonstration of the visualization system"""
    print("🎨 Protein Optimization Visualizer")
    print("Creating sample visualizations...")
    
    visualizer = ProteinOptimizationVisualizer()
    
    # Create sample data
    sample_generation_data = []
    for i in range(30):
        data = {
            'generation': i,
            'best_fitness': [
                0.3 + 0.4 * (1 - np.exp(-i/10)) + np.random.normal(0, 0.02),
                0.4 + 0.5 * (1 - np.exp(-i/15)) + np.random.normal(0, 0.02),
                0.2 + 0.6 * (1 - np.exp(-i/12)) + np.random.normal(0, 0.02),
                0.3 + 0.5 * (1 - np.exp(-i/8)) + np.random.normal(0, 0.02)
            ],
            'best_sequence': 'MHHVAALLALAVCANACSHVYFADSDLHDHGRRLT',
            'population_diversity': 0.8 - 0.3 * (i/29) + np.random.normal(0, 0.05)
        }
        sample_generation_data.append(data)
    
    print("✅ Visualizer ready for use!")
    return visualizer

if __name__ == "__main__":
    visualizer = main()