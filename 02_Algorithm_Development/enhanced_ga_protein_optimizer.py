"""
Enhanced Genetic Algorithm for Optimizing Algae Proteins for CO2 Absorption
Focuses on Carbonic Anhydrase optimization with improved fitness functions and 3D visualization
"""

import pandas as pd
import numpy as np
import requests
import random
import json
import pickle
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from Bio import SeqIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from deap import base, creator, tools, algorithms
import py3Dmol
import multiprocessing
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# === Enhanced Constants ===
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_PROPERTIES = {
    'A': {'hydrophobic': 1, 'size': 0.31, 'charge': 0, 'polar': 0},
    'C': {'hydrophobic': 1, 'size': 0.54, 'charge': 0, 'polar': 0.5},
    'D': {'hydrophobic': 0, 'size': 0.54, 'charge': -1, 'polar': 1},
    'E': {'hydrophobic': 0, 'size': 0.62, 'charge': -1, 'polar': 1},
    'F': {'hydrophobic': 1, 'size': 0.88, 'charge': 0, 'polar': 0},
    'G': {'hydrophobic': 0.5, 'size': 0, 'charge': 0, 'polar': 0},
    'H': {'hydrophobic': 0, 'size': 0.69, 'charge': 0.5, 'polar': 1},
    'I': {'hydrophobic': 1, 'size': 0.73, 'charge': 0, 'polar': 0},
    'K': {'hydrophobic': 0, 'size': 0.77, 'charge': 1, 'polar': 1},
    'L': {'hydrophobic': 1, 'size': 0.73, 'charge': 0, 'polar': 0},
    'M': {'hydrophobic': 1, 'size': 0.73, 'charge': 0, 'polar': 0},
    'N': {'hydrophobic': 0, 'size': 0.54, 'charge': 0, 'polar': 1},
    'P': {'hydrophobic': 0.5, 'size': 0.31, 'charge': 0, 'polar': 0},
    'Q': {'hydrophobic': 0, 'size': 0.62, 'charge': 0, 'polar': 1},
    'R': {'hydrophobic': 0, 'size': 0.88, 'charge': 1, 'polar': 1},
    'S': {'hydrophobic': 0, 'size': 0.31, 'charge': 0, 'polar': 1},
    'T': {'hydrophobic': 0.5, 'size': 0.39, 'charge': 0, 'polar': 1},
    'V': {'hydrophobic': 1, 'size': 0.54, 'charge': 0, 'polar': 0},
    'W': {'hydrophobic': 1, 'size': 1.0, 'charge': 0, 'polar': 0.5},
    'Y': {'hydrophobic': 0.5, 'size': 0.88, 'charge': 0, 'polar': 1}
}

# CO2 binding motifs in carbonic anhydrase
CO2_BINDING_MOTIFS = [
    "HHH",  # Zinc binding histidines
    "DX[TS]",  # Catalytic residues
    "CX{2,4}C",  # Disulfide bonds
    "GX{3}G",  # Flexible loops
]

# DEAP setup with enhanced fitness
for cname in ['FitnessMulti', 'Individual']:
    if hasattr(creator, cname): 
        delattr(creator, cname)

# Multi-objective fitness: (CO2_affinity, stability, expression_level, catalytic_efficiency)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

class ProteinAnalyzer:
    """Advanced protein analysis for CO2 binding optimization"""
    
    @staticmethod
    def calculate_hydrophobicity(sequence: str) -> float:
        """Calculate average hydrophobicity of sequence"""
        if not sequence:
            return 0.0
        return np.mean([AA_PROPERTIES.get(aa, {}).get('hydrophobic', 0) for aa in sequence])
    
    @staticmethod
    def calculate_charge_distribution(sequence: str) -> float:
        """Calculate charge distribution score"""
        charges = [AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in sequence]
        return np.std(charges) if charges else 0.0
    
    @staticmethod
    def find_co2_binding_motifs(sequence: str) -> int:
        """Count CO2 binding motifs in sequence"""
        import re
        motif_count = 0
        for motif in CO2_BINDING_MOTIFS:
            motif_count += len(re.findall(motif, sequence))
        return motif_count
    
    @staticmethod
    def calculate_secondary_structure_propensity(sequence: str) -> Dict[str, float]:
        """Estimate secondary structure propensities"""
        # Simplified Chou-Fasman propensities
        helix_formers = set("AEHKLMQR")
        sheet_formers = set("CFILTVY")
        turn_formers = set("DGHNPST")
        
        total_len = len(sequence)
        if total_len == 0:
            return {'helix': 0, 'sheet': 0, 'turn': 0}
            
        helix_prop = sum(1 for aa in sequence if aa in helix_formers) / total_len
        sheet_prop = sum(1 for aa in sequence if aa in sheet_formers) / total_len
        turn_prop = sum(1 for aa in sequence if aa in turn_formers) / total_len
        
        return {'helix': helix_prop, 'sheet': sheet_prop, 'turn': turn_prop}

class EnhancedGeneticAlgorithm:
    """Enhanced GA for protein optimization"""
    
    def __init__(self, target_sequence: str, config: Dict):
        self.target_sequence = target_sequence
        self.config = config
        self.analyzer = ProteinAnalyzer()
        self.generation_data = []
        
    def evaluate_protein_fitness(self, individual: List[str]) -> Tuple[float, float, float, float]:
        """Multi-objective fitness evaluation"""
        sequence = ''.join(individual)
        
        # 1. CO2 Binding Affinity (based on motifs and target similarity)
        motif_score = self.analyzer.find_co2_binding_motifs(sequence)
        target_similarity = self._calculate_sequence_similarity(sequence, self.target_sequence)
        co2_affinity = (motif_score * 0.3 + target_similarity * 0.7)
        
        # 2. Protein Stability (based on hydrophobicity and charge distribution)
        hydrophobicity = self.analyzer.calculate_hydrophobicity(sequence)
        charge_dist = self.analyzer.calculate_charge_distribution(sequence)
        stability = (0.5 - abs(hydrophobicity - 0.5)) + (1.0 - min(charge_dist, 1.0))
        
        # 3. Expression Level (based on codon usage and structure)
        secondary_struct = self.analyzer.calculate_secondary_structure_propensity(sequence)
        expression_level = secondary_struct['helix'] * 0.4 + secondary_struct['sheet'] * 0.4 + secondary_struct['turn'] * 0.2
        
        # 4. Catalytic Efficiency (specialized for carbonic anhydrase)
        catalytic_efficiency = self._calculate_catalytic_efficiency(sequence)
        
        return co2_affinity, stability, expression_level, catalytic_efficiency
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity with alignment"""
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        return matches / min_len
    
    def _calculate_catalytic_efficiency(self, sequence: str) -> float:
        """Calculate catalytic efficiency based on active site characteristics"""
        # Look for zinc-binding histidines and catalytic residues
        his_count = sequence.count('H')
        asp_glu_count = sequence.count('D') + sequence.count('E')
        cys_count = sequence.count('C')
        
        # Normalized scores
        his_score = min(his_count / 3.0, 1.0)  # Ideal ~3 histidines for zinc binding
        catalytic_score = min(asp_glu_count / 5.0, 1.0)  # Catalytic residues
        stability_score = min(cys_count / 4.0, 1.0)  # Disulfide bonds
        
        return (his_score * 0.5 + catalytic_score * 0.3 + stability_score * 0.2)
    
    def adaptive_mutation(self, individual: List[str]) -> Tuple[List[str]]:
        """Adaptive mutation based on generation and fitness"""
        # Use a default mutation rate since we can't access generation here
        mutation_rate = self.config['mutation_rate']
            
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Biased mutation towards beneficial amino acids
                if random.random() < 0.3:  # 30% chance for beneficial mutation
                    individual[i] = random.choice(['H', 'D', 'E', 'C'])  # CO2 binding residues
                else:
                    individual[i] = random.choice(AA)
        return individual,
    
    def intelligent_crossover(self, ind1: List[str], ind2: List[str]) -> Tuple[List[str], List[str]]:
        """Intelligent crossover preserving important motifs"""
        if random.random() < self.config['crossover_rate']:
            # Try to preserve CO2 binding motifs
            crossover_points = []
            for i in range(len(ind1) - 2):
                motif1 = ''.join(ind1[i:i+3])
                motif2 = ''.join(ind2[i:i+3])
                if any(motif1 in co2_motif or motif2 in co2_motif for co2_motif in CO2_BINDING_MOTIFS):
                    crossover_points.append(i)
            
            if crossover_points:
                point = random.choice(crossover_points)
            else:
                point = random.randint(1, len(ind1) - 1)
                
            ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
        
        return ind1, ind2
    
    def run_optimization(self) -> Dict:
        """Run the enhanced genetic algorithm"""
        # Setup toolbox
        toolbox = base.Toolbox()
        toolbox.register("attr_aa", lambda: random.choice(AA))
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                        toolbox.attr_aa, self.config['sequence_length'])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", self.intelligent_crossover)
        toolbox.register("mutate", self.adaptive_mutation)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", self.evaluate_protein_fitness)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        # Initialize population
        pop = toolbox.population(n=self.config['population_size'])
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Track evolution
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields
        
        # Evolution loop
        for gen in range(self.config['generations']):
            # Selection and variation
            offspring = algorithms.varAnd(pop, toolbox, 
                                        cxpb=self.config['crossover_rate'],
                                        mutpb=self.config['mutation_rate'])
            
            # Evaluate offspring
            fits = list(map(toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            # Environmental selection
            pop = toolbox.select(pop + offspring, self.config['population_size'])
            
            # Statistics
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(offspring), **record)
            
            # Store generation data
            best_ind = tools.selBest(pop, 1)[0]
            self.generation_data.append({
                'generation': gen,
                'best_fitness': best_ind.fitness.values,
                'best_sequence': ''.join(best_ind),
                'avg_fitness': record['avg'],
                'population_diversity': self._calculate_diversity(pop)
            })
            
            if gen % 20 == 0:
                print(f"Generation {gen}: Best fitness = {best_ind.fitness.values}")
        
        # Return results
        best_individual = tools.selBest(pop, 1)[0]
        return {
            'best_sequence': ''.join(best_individual),
            'best_fitness': best_individual.fitness.values,
            'generation_data': self.generation_data,
            'final_population': pop,
            'logbook': logbook
        }
    
    def _calculate_diversity(self, population: List) -> float:
        """Calculate population diversity"""
        sequences = [''.join(ind) for ind in population]
        if len(sequences) < 2:
            return 0.0
        
        total_distance = 0
        count = 0
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                distance = sum(a != b for a, b in zip(sequences[i], sequences[j]))
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0

class AdvancedVisualization:
    """Advanced 3D visualization for protein optimization results"""
    
    def __init__(self):
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    def plot_evolution_progress(self, generation_data: List[Dict], save_path: str = None):
        """Plot multi-objective evolution progress"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Genetic Algorithm Evolution Progress', fontsize=16)
        
        generations = [d['generation'] for d in generation_data]
        
        # Extract fitness components
        co2_affinity = [d['best_fitness'][0] for d in generation_data]
        stability = [d['best_fitness'][1] for d in generation_data]
        expression = [d['best_fitness'][2] for d in generation_data]
        catalytic = [d['best_fitness'][3] for d in generation_data]
        diversity = [d['population_diversity'] for d in generation_data]
        
        # Plot individual objectives
        axes[0, 0].plot(generations, co2_affinity, 'r-', linewidth=2, label='CO2 Affinity')
        axes[0, 0].set_title('CO2 Binding Affinity')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness Score')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(generations, stability, 'b-', linewidth=2, label='Stability')
        axes[0, 1].set_title('Protein Stability')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Fitness Score')
        axes[0, 1].grid(True)
        
        axes[0, 2].plot(generations, expression, 'g-', linewidth=2, label='Expression')
        axes[0, 2].set_title('Expression Level')
        axes[0, 2].set_xlabel('Generation')
        axes[0, 2].set_ylabel('Fitness Score')
        axes[0, 2].grid(True)
        
        axes[1, 0].plot(generations, catalytic, 'orange', linewidth=2, label='Catalytic Efficiency')
        axes[1, 0].set_title('Catalytic Efficiency')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Fitness Score')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(generations, diversity, 'purple', linewidth=2, label='Diversity')
        axes[1, 1].set_title('Population Diversity')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Diversity Score')
        axes[1, 1].grid(True)
        
        # Combined plot
        axes[1, 2].plot(generations, co2_affinity, 'r-', alpha=0.7, label='CO2 Affinity')
        axes[1, 2].plot(generations, stability, 'b-', alpha=0.7, label='Stability')
        axes[1, 2].plot(generations, expression, 'g-', alpha=0.7, label='Expression')
        axes[1, 2].plot(generations, catalytic, 'orange', alpha=0.7, label='Catalytic')
        axes[1, 2].set_title('All Objectives Combined')
        axes[1, 2].set_xlabel('Generation')
        axes[1, 2].set_ylabel('Fitness Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_3d_fitness_landscape(self, generation_data: List[Dict], save_path: str = None):
        """Create 3D fitness landscape visualization"""
        fig = plt.figure(figsize=(15, 5))
        
        # Extract data
        co2_affinity = [d['best_fitness'][0] for d in generation_data]
        stability = [d['best_fitness'][1] for d in generation_data]
        expression = [d['best_fitness'][2] for d in generation_data]
        generations = [d['generation'] for d in generation_data]
        
        # 3D scatter plot 1: CO2 vs Stability vs Expression
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(co2_affinity, stability, expression, 
                            c=generations, cmap='viridis', s=50)
        ax1.set_xlabel('CO2 Affinity')
        ax1.set_ylabel('Stability')
        ax1.set_zlabel('Expression')
        ax1.set_title('3D Fitness Evolution')
        fig.colorbar(scatter, ax=ax1, label='Generation')
        
        # 3D trajectory plot
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(co2_affinity, stability, expression, 'r-', alpha=0.7, linewidth=2)
        ax2.scatter(co2_affinity[0], stability[0], expression[0], 
                   color='green', s=100, label='Start')
        ax2.scatter(co2_affinity[-1], stability[-1], expression[-1], 
                   color='red', s=100, label='End')
        ax2.set_xlabel('CO2 Affinity')
        ax2.set_ylabel('Stability')
        ax2.set_zlabel('Expression')
        ax2.set_title('Evolution Trajectory')
        ax2.legend()
        
        # Pareto front approximation
        ax3 = fig.add_subplot(133, projection='3d')
        # Take last 20 generations for Pareto analysis
        recent_data = generation_data[-20:]
        recent_co2 = [d['best_fitness'][0] for d in recent_data]
        recent_stab = [d['best_fitness'][1] for d in recent_data]
        recent_expr = [d['best_fitness'][2] for d in recent_data]
        
        ax3.scatter(recent_co2, recent_stab, recent_expr, 
                   c='red', s=60, alpha=0.8)
        ax3.set_xlabel('CO2 Affinity')
        ax3.set_ylabel('Stability')
        ax3.set_zlabel('Expression')
        ax3.set_title('Recent Best Solutions')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_protein_viewer(self, protein_sequence: str, 
                                        pdb_url: str = None, 
                                        binder_sequence: str = None) -> py3Dmol.view:
        """Create interactive 3D protein viewer"""
        view = py3Dmol.view(width=800, height=600)
        
        if pdb_url and pdb_url.startswith("http"):
            try:
                pdb_data = requests.get(pdb_url).text
                view.addModel(pdb_data, "pdb")
                view.setStyle({'cartoon': {'color': 'spectrum'}})
            except:
                print("Could not load PDB structure")
        
        if binder_sequence:
            try:
                # Use ESMFold for structure prediction
                esmfold_url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
                binder_pdb = requests.post(esmfold_url, data=binder_sequence).text
                view.addModel(binder_pdb, "pdb")
                view.setStyle({'stick': {'colorscheme': 'redCarbon'}}, model=1)
            except:
                print("Could not predict binder structure")
        
        view.zoomTo()
        view.spin(True)
        return view

def main():
    """Main function to run the enhanced genetic algorithm"""
    # Configuration
    config = {
        'population_size': 50,
        'generations': 150,
        'sequence_length': 25,
        'mutation_rate': 0.15,
        'crossover_rate': 0.7,
        'elite_count': 5
    }
    
    print("Enhanced Genetic Algorithm for Algae Protein Optimization")
    print("=" * 60)
    
    return config

if __name__ == "__main__":
    config = main()