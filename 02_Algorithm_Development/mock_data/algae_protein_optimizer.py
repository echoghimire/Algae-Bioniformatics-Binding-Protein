"""
Complete Application for Algae Protein Optimization with Enhanced Visualization
Integrates genetic algorithm, data analysis, and interactive 3D visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
from pathlib import Path

# Import our enhanced genetic algorithm
from enhanced_ga_protein_optimizer import (
    EnhancedGeneticAlgorithm, 
    AdvancedVisualization, 
    ProteinAnalyzer,
    AA_PROPERTIES,
    CO2_BINDING_MOTIFS
)

class AlgaeProteinOptimizer:
    """Main application class for algae protein optimization"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.trials_path = self.workspace_path / "Trials Data"
        self.results_path = self.workspace_path / "Results"
        self.results_path.mkdir(exist_ok=True)
        
        self.analyzer = ProteinAnalyzer()
        self.visualizer = AdvancedVisualization()
        self.target_proteins = {}
        self.optimization_results = {}
        
    def load_target_proteins(self, fasta_file: str) -> Dict:
        """Load target proteins from FASTA file"""
        fasta_path = self.workspace_path / fasta_file
        proteins = {}
        
        try:
            for record in SeqIO.parse(fasta_path, "fasta"):
                accession = record.id.split("|")[1] if "|" in record.id else record.id
                proteins[accession] = {
                    'id': record.id,
                    'sequence': str(record.seq),
                    'description': record.description,
                    'length': len(record.seq)
                }
                print(f"Loaded protein {accession}: {record.description[:100]}...")
        except Exception as e:
            print(f"Error loading FASTA file: {e}")
            return {}
        
        self.target_proteins = proteins
        return proteins
    
    def fetch_protein_structure_data(self, accession: str) -> Dict:
        """Fetch protein structure and metadata from external APIs"""
        print(f"Fetching data for {accession}...")
        
        # UniProt data
        uniprot_data = self._fetch_uniprot_data(accession)
        
        # AlphaFold structure
        alphafold_data = self._fetch_alphafold_data(accession)
        
        return {
            'uniprot': uniprot_data,
            'alphafold': alphafold_data,
            'fetched_at': datetime.now().isoformat()
        }
    
    def _fetch_uniprot_data(self, accession: str) -> Dict:
        """Fetch UniProt protein data"""
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'protein_name': data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
                    'organism': data.get("organism", {}).get("scientificName", ""),
                    'function': next((comment.get("value") for comment in data.get("comments", []) if comment.get('type') == 'FUNCTION'), ""),
                    'keywords': [kw.get("value") for kw in data.get("keywords", [])],
                    'go_terms': [ref.get("id") for ref in data.get("dbReferences", []) if ref.get("type") == "GO"]
                }
        except Exception as e:
            print(f"Error fetching UniProt data: {e}")
        return {}
    
    def _fetch_alphafold_data(self, accession: str) -> Dict:
        """Fetch AlphaFold structure data"""
        url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    return {
                        'pdb_url': entry.get("pdbUrl", ""),
                        'confidence_url': entry.get("bcifUrl", ""),
                        'png_url': f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.png",
                        'confidence_score': entry.get("confidenceScore", 0)
                    }
        except Exception as e:
            print(f"Error fetching AlphaFold data: {e}")
        return {}
    
    def run_optimization_trial(self, accession: str, trial_name: str, config: Dict = None) -> Dict:
        """Run a complete optimization trial for a protein"""
        print(f"\\nStarting optimization trial: {trial_name} for protein {accession}")
        print("=" * 60)
        
        if accession not in self.target_proteins:
            raise ValueError(f"Protein {accession} not loaded. Load FASTA file first.")
        
        target_sequence = self.target_proteins[accession]['sequence']
        
        # Default configuration
        if config is None:
            config = {
                'population_size': 50,
                'generations': 150,
                'sequence_length': 25,
                'mutation_rate': 0.15,
                'crossover_rate': 0.7,
                'elite_count': 5
            }
        
        # Initialize and run genetic algorithm
        ga = EnhancedGeneticAlgorithm(target_sequence, config)
        results = ga.run_optimization()
        
        # Analyze results
        analysis = self._analyze_optimization_results(results, target_sequence)
        
        # Store results
        trial_data = {
            'trial_name': trial_name,
            'accession': accession,
            'target_protein': self.target_proteins[accession],
            'config': config,
            'results': results,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_results[trial_name] = trial_data
        
        # Save trial data
        self._save_trial_data(trial_data)
        
        # Generate visualizations
        self._generate_trial_visualizations(trial_data)
        
        return trial_data
    
    def _analyze_optimization_results(self, results: Dict, target_sequence: str) -> Dict:
        """Analyze optimization results"""
        best_sequence = results['best_sequence']
        
        analysis = {
            'sequence_analysis': {
                'length': len(best_sequence),
                'composition': {aa: best_sequence.count(aa) for aa in set(best_sequence)},
                'hydrophobicity': self.analyzer.calculate_hydrophobicity(best_sequence),
                'charge_distribution': self.analyzer.calculate_charge_distribution(best_sequence),
                'co2_motifs': self.analyzer.find_co2_binding_motifs(best_sequence),
                'secondary_structure': self.analyzer.calculate_secondary_structure_propensity(best_sequence)
            },
            'optimization_metrics': {
                'final_fitness': results['best_fitness'],
                'convergence_generation': self._find_convergence_point(results['generation_data']),
                'improvement_rate': self._calculate_improvement_rate(results['generation_data']),
                'final_diversity': results['generation_data'][-1]['population_diversity'] if results['generation_data'] else 0
            },
            'comparison_with_target': {
                'similarity': self._calculate_sequence_similarity(best_sequence, target_sequence),
                'length_ratio': len(best_sequence) / len(target_sequence),
                'conserved_motifs': self._find_conserved_motifs(best_sequence, target_sequence)
            }
        }
        
        return analysis
    
    def _find_convergence_point(self, generation_data: List[Dict]) -> int:
        """Find the generation where the algorithm converged"""
        if len(generation_data) < 10:
            return len(generation_data)
        
        fitness_values = [d['best_fitness'][0] for d in generation_data]
        for i in range(10, len(fitness_values)):
            recent_variance = np.var(fitness_values[i-10:i])
            if recent_variance < 0.001:  # Very small variance indicates convergence
                return i
        
        return len(generation_data)
    
    def _calculate_improvement_rate(self, generation_data: List[Dict]) -> float:
        """Calculate the rate of improvement over generations"""
        if len(generation_data) < 2:
            return 0.0
        
        initial_fitness = generation_data[0]['best_fitness'][0]
        final_fitness = generation_data[-1]['best_fitness'][0]
        
        return (final_fitness - initial_fitness) / len(generation_data)
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity"""
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        return matches / min_len
    
    def _find_conserved_motifs(self, seq1: str, seq2: str) -> List[str]:
        """Find conserved motifs between sequences"""
        conserved = []
        min_len = min(len(seq1), len(seq2))
        
        for length in [3, 4, 5]:  # Check for 3-5 AA motifs
            for i in range(min_len - length + 1):
                motif1 = seq1[i:i+length]
                motif2 = seq2[i:i+length]
                if motif1 == motif2 and motif1 not in conserved:
                    conserved.append(motif1)
        
        return conserved
    
    def _save_trial_data(self, trial_data: Dict):
        """Save trial data to files"""
        trial_name = trial_data['trial_name']
        
        # Create trial directory
        trial_dir = self.results_path / trial_name
        trial_dir.mkdir(exist_ok=True)
        
        # Save complete trial data as JSON
        with open(trial_dir / "complete_data.json", 'w') as f:
            # Convert non-serializable objects for JSON
            serializable_data = self._make_json_serializable(trial_data)
            json.dump(serializable_data, f, indent=2)
        
        # Save optimization table (similar to your existing format)
        results_df = pd.DataFrame([{
            'Accession': trial_data['accession'],
            'Status': 'Processed',
            'Best Binder': trial_data['results']['best_sequence'],
            'CO2 Affinity': trial_data['results']['best_fitness'][0],
            'Stability': trial_data['results']['best_fitness'][1],
            'Expression': trial_data['results']['best_fitness'][2],
            'Catalytic Efficiency': trial_data['results']['best_fitness'][3],
            'Trial Name': trial_name
        }])
        
        results_df.to_csv(trial_dir / f"optimization_results_{trial_name}.csv", index=False)
        
        print(f"Trial data saved to: {trial_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_trial_visualizations(self, trial_data: Dict):
        """Generate visualizations for a trial"""
        trial_name = trial_data['trial_name']
        trial_dir = self.results_path / trial_name
        
        # Evolution progress plot
        self.visualizer.plot_evolution_progress(
            trial_data['results']['generation_data'],
            save_path=str(trial_dir / "evolution_progress.png")
        )
        
        # 3D fitness landscape
        self.visualizer.plot_3d_fitness_landscape(
            trial_data['results']['generation_data'],
            save_path=str(trial_dir / "fitness_landscape_3d.png")
        )
        
        # Create protein analysis plots
        self._create_protein_analysis_plots(trial_data, trial_dir)
    
    def _create_protein_analysis_plots(self, trial_data: Dict, save_dir: Path):
        """Create detailed protein analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Protein Analysis - {trial_data['trial_name']}", fontsize=16)
        
        analysis = trial_data['analysis']
        best_sequence = trial_data['results']['best_sequence']
        
        # Amino acid composition
        composition = analysis['sequence_analysis']['composition']
        aa_names = list(composition.keys())
        aa_counts = list(composition.values())
        
        axes[0, 0].bar(aa_names, aa_counts, color='steelblue')
        axes[0, 0].set_title('Amino Acid Composition')
        axes[0, 0].set_xlabel('Amino Acid')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Secondary structure propensities
        ss_props = analysis['sequence_analysis']['secondary_structure']
        ss_names = list(ss_props.keys())
        ss_values = list(ss_props.values())
        
        axes[0, 1].pie(ss_values, labels=ss_names, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Secondary Structure Propensities')
        
        # Fitness evolution
        gen_data = trial_data['results']['generation_data']
        generations = [d['generation'] for d in gen_data]
        fitness_values = [sum(d['best_fitness']) for d in gen_data]  # Total fitness
        
        axes[1, 0].plot(generations, fitness_values, 'g-', linewidth=2)
        axes[1, 0].set_title('Total Fitness Evolution')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Total Fitness')
        axes[1, 0].grid(True)
        
        # Property analysis
        properties = ['hydrophobic', 'size', 'charge', 'polar']
        prop_values = []
        for prop in properties:
            values = [AA_PROPERTIES.get(aa, {}).get(prop, 0) for aa in best_sequence]
            prop_values.append(np.mean(values))
        
        axes[1, 1].bar(properties, prop_values, color=['red', 'blue', 'green', 'orange'])
        axes[1, 1].set_title('Average Physicochemical Properties')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / "protein_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_viewer(self, trial_name: str) -> py3Dmol.view:
        """Create interactive 3D viewer for optimization results"""
        if trial_name not in self.optimization_results:
            print(f"Trial {trial_name} not found")
            return None
        
        trial_data = self.optimization_results[trial_name]
        accession = trial_data['accession']
        best_sequence = trial_data['results']['best_sequence']
        
        # Fetch structure data if not already available
        if 'structure_data' not in trial_data:
            structure_data = self.fetch_protein_structure_data(accession)
            trial_data['structure_data'] = structure_data
        
        pdb_url = trial_data['structure_data'].get('alphafold', {}).get('pdb_url', '')
        
        return self.visualizer.create_interactive_protein_viewer(
            protein_sequence=trial_data['target_protein']['sequence'],
            pdb_url=pdb_url,
            binder_sequence=best_sequence
        )
    
    def compare_trials(self, trial_names: List[str]) -> pd.DataFrame:
        """Compare multiple optimization trials"""
        comparison_data = []
        
        for trial_name in trial_names:
            if trial_name not in self.optimization_results:
                continue
                
            trial_data = self.optimization_results[trial_name]
            analysis = trial_data['analysis']
            results = trial_data['results']
            
            comparison_data.append({
                'Trial': trial_name,
                'Accession': trial_data['accession'],
                'Best Sequence': results['best_sequence'],
                'CO2 Affinity': results['best_fitness'][0],
                'Stability': results['best_fitness'][1],
                'Expression': results['best_fitness'][2],
                'Catalytic Efficiency': results['best_fitness'][3],
                'Total Fitness': sum(results['best_fitness']),
                'Convergence Gen': analysis['optimization_metrics']['convergence_generation'],
                'Improvement Rate': analysis['optimization_metrics']['improvement_rate'],
                'Hydrophobicity': analysis['sequence_analysis']['hydrophobicity'],
                'CO2 Motifs': analysis['sequence_analysis']['co2_motifs']
            })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def generate_comprehensive_report(self, trial_name: str) -> str:
        """Generate a comprehensive text report for a trial"""
        if trial_name not in self.optimization_results:
            return f"Trial {trial_name} not found"
        
        trial_data = self.optimization_results[trial_name]
        analysis = trial_data['analysis']
        results = trial_data['results']
        
        report = f"""
ALGAE PROTEIN OPTIMIZATION REPORT
Trial: {trial_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

TARGET PROTEIN INFORMATION:
- Accession: {trial_data['accession']}
- Description: {trial_data['target_protein']['description']}
- Length: {trial_data['target_protein']['length']} amino acids

OPTIMIZATION CONFIGURATION:
- Population Size: {trial_data['config']['population_size']}
- Generations: {trial_data['config']['generations']}
- Sequence Length: {trial_data['config']['sequence_length']}
- Mutation Rate: {trial_data['config']['mutation_rate']}
- Crossover Rate: {trial_data['config']['crossover_rate']}

OPTIMIZATION RESULTS:
- Best Sequence: {results['best_sequence']}
- Fitness Scores:
  * CO2 Affinity: {results['best_fitness'][0]:.4f}
  * Stability: {results['best_fitness'][1]:.4f}
  * Expression Level: {results['best_fitness'][2]:.4f}
  * Catalytic Efficiency: {results['best_fitness'][3]:.4f}
  * Total Fitness: {sum(results['best_fitness']):.4f}

SEQUENCE ANALYSIS:
- Length: {analysis['sequence_analysis']['length']}
- Hydrophobicity: {analysis['sequence_analysis']['hydrophobicity']:.4f}
- Charge Distribution: {analysis['sequence_analysis']['charge_distribution']:.4f}
- CO2 Binding Motifs: {analysis['sequence_analysis']['co2_motifs']}
- Secondary Structure:
  * Helix Propensity: {analysis['sequence_analysis']['secondary_structure']['helix']:.4f}
  * Sheet Propensity: {analysis['sequence_analysis']['secondary_structure']['sheet']:.4f}
  * Turn Propensity: {analysis['sequence_analysis']['secondary_structure']['turn']:.4f}

OPTIMIZATION METRICS:
- Convergence Generation: {analysis['optimization_metrics']['convergence_generation']}
- Improvement Rate: {analysis['optimization_metrics']['improvement_rate']:.6f}
- Final Diversity: {analysis['optimization_metrics']['final_diversity']:.4f}

TARGET COMPARISON:
- Similarity to Target: {analysis['comparison_with_target']['similarity']:.4f}
- Length Ratio: {analysis['comparison_with_target']['length_ratio']:.4f}
- Conserved Motifs: {', '.join(analysis['comparison_with_target']['conserved_motifs'])}

RECOMMENDATIONS:
"""
        
        # Add recommendations based on analysis
        if results['best_fitness'][0] > 0.7:
            report += "✓ Excellent CO2 binding affinity achieved\\n"
        elif results['best_fitness'][0] > 0.5:
            report += "• Good CO2 binding affinity, consider further optimization\\n"
        else:
            report += "⚠ Low CO2 binding affinity, recommend parameter adjustment\\n"
            
        if analysis['sequence_analysis']['co2_motifs'] >= 2:
            report += "✓ Good presence of CO2 binding motifs\\n"
        else:
            report += "• Consider increasing CO2 binding motif presence\\n"
            
        if analysis['optimization_metrics']['convergence_generation'] < trial_data['config']['generations'] * 0.8:
            report += "✓ Algorithm converged efficiently\\n"
        else:
            report += "• Consider increasing generations for better convergence\\n"
        
        return report

def main():
    """Main application entry point"""
    print("🧬 Algae Protein Optimization System")
    print("Optimizing proteins for enhanced CO2 absorption")
    print("=" * 60)
    
    # Initialize the optimizer
    workspace_path = r"c:\\Users\\Gunjan Ghimire\\Downloads\\Testting code\\Algae-Bioniformatics-Binding-Protein"
    optimizer = AlgaeProteinOptimizer(workspace_path)
    
    return optimizer

if __name__ == "__main__":
    optimizer = main()