"""
Comprehensive Data Analysis and Reporting System
Analyzes optimization trials, compares results, and generates detailed reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAnalyzer:
    """Comprehensive analysis system for optimization trials"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.trials_path = self.workspace_path / "Trials Data"
        self.results_path = self.workspace_path / "Results"
        self.reports_path = self.workspace_path / "Reports"
        self.reports_path.mkdir(exist_ok=True)
        
        self.trial_data = {}
        self.comparison_data = pd.DataFrame()
        
    def load_all_trials(self) -> Dict:
        """Load all trial data from both old and new formats"""
        print("Loading trial data...")
        
        # Load old format trials
        old_trials = self._load_old_format_trials()
        
        # Load new format trials
        new_trials = self._load_new_format_trials()
        
        # Combine data
        self.trial_data = {**old_trials, **new_trials}
        
        print(f"Loaded {len(self.trial_data)} trials total")
        print(f"  - Old format: {len(old_trials)}")
        print(f"  - New format: {len(new_trials)}")
        
        return self.trial_data
    
    def _load_old_format_trials(self) -> Dict:
        """Load trials in old format from Trials Data directory"""
        old_trials = {}
        
        if not self.trials_path.exists():
            return old_trials
        
        for trial_dir in self.trials_path.iterdir():
            if trial_dir.is_dir() and trial_dir.name.startswith("Trial"):
                trial_num = trial_dir.name
                
                # Look for optimization table file
                table_file = trial_dir / f"Binder Optimization Table {trial_num}.txt"
                if table_file.exists():
                    try:
                        # Read the optimization table
                        with open(table_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Parse the data
                        if len(lines) >= 3:  # Header + data
                            data_line = lines[2].strip().split('\\t')
                            if len(data_line) >= 5:
                                old_trials[trial_num] = {
                                    'trial_name': trial_num,
                                    'accession': data_line[1],
                                    'status': data_line[2],
                                    'best_sequence': data_line[3],
                                    'pdb_url': data_line[4] if len(data_line) > 4 else '',
                                    'best_score': float(data_line[5]) if len(data_line) > 5 and data_line[5].replace('.', '').isdigit() else 0.0,
                                    'format': 'old',
                                    'timestamp': trial_dir.stat().st_mtime
                                }
                    except Exception as e:
                        print(f"Error loading {trial_num}: {e}")
        
        return old_trials
    
    def _load_new_format_trials(self) -> Dict:
        """Load trials in new format from Results directory"""
        new_trials = {}
        
        if not self.results_path.exists():
            return new_trials
        
        for trial_dir in self.results_path.iterdir():
            if trial_dir.is_dir():
                complete_data_file = trial_dir / "complete_data.json"
                if complete_data_file.exists():
                    try:
                        with open(complete_data_file, 'r') as f:
                            trial_data = json.load(f)
                        
                        trial_name = trial_data.get('trial_name', trial_dir.name)
                        trial_data['format'] = 'new'
                        new_trials[trial_name] = trial_data
                        
                    except Exception as e:
                        print(f"Error loading {trial_dir.name}: {e}")
        
        return new_trials
    
    def create_comprehensive_comparison(self) -> pd.DataFrame:
        """Create comprehensive comparison of all trials"""
        if not self.trial_data:
            self.load_all_trials()
        
        comparison_rows = []
        
        for trial_name, trial_data in self.trial_data.items():
            if trial_data['format'] == 'old':
                row = {
                    'Trial': trial_name,
                    'Format': 'Old',
                    'Accession': trial_data.get('accession', ''),
                    'Best Sequence': trial_data.get('best_sequence', ''),
                    'Best Score (Old)': trial_data.get('best_score', 0.0),
                    'CO2 Affinity': np.nan,
                    'Stability': np.nan,
                    'Expression': np.nan,
                    'Catalytic Efficiency': np.nan,
                    'Total Fitness': trial_data.get('best_score', 0.0),
                    'Sequence Length': len(trial_data.get('best_sequence', '')),
                    'Timestamp': datetime.fromtimestamp(trial_data.get('timestamp', 0)).isoformat()
                }
            else:  # New format
                results = trial_data.get('results', {})
                best_fitness = results.get('best_fitness', [0, 0, 0, 0])
                
                row = {
                    'Trial': trial_name,
                    'Format': 'New',
                    'Accession': trial_data.get('accession', ''),
                    'Best Sequence': results.get('best_sequence', ''),
                    'Best Score (Old)': np.nan,
                    'CO2 Affinity': best_fitness[0] if len(best_fitness) > 0 else 0,
                    'Stability': best_fitness[1] if len(best_fitness) > 1 else 0,
                    'Expression': best_fitness[2] if len(best_fitness) > 2 else 0,
                    'Catalytic Efficiency': best_fitness[3] if len(best_fitness) > 3 else 0,
                    'Total Fitness': sum(best_fitness),
                    'Sequence Length': len(results.get('best_sequence', '')),
                    'Timestamp': trial_data.get('timestamp', '')
                }
            
            comparison_rows.append(row)
        
        self.comparison_data = pd.DataFrame(comparison_rows)
        return self.comparison_data
    
    def analyze_performance_trends(self) -> Dict:
        """Analyze performance trends across trials"""
        if self.comparison_data.empty:
            self.create_comprehensive_comparison()
        
        analysis = {
            'overall_statistics': {},
            'format_comparison': {},
            'performance_trends': {},
            'best_performers': {},
            'improvement_analysis': {}
        }
        
        # Overall statistics
        analysis['overall_statistics'] = {
            'total_trials': len(self.comparison_data),
            'old_format_trials': len(self.comparison_data[self.comparison_data['Format'] == 'Old']),
            'new_format_trials': len(self.comparison_data[self.comparison_data['Format'] == 'New']),
            'unique_accessions': self.comparison_data['Accession'].nunique(),
            'avg_sequence_length': self.comparison_data['Sequence Length'].mean(),
            'sequence_length_std': self.comparison_data['Sequence Length'].std()
        }
        
        # Format comparison
        old_data = self.comparison_data[self.comparison_data['Format'] == 'Old']
        new_data = self.comparison_data[self.comparison_data['Format'] == 'New']
        
        if not old_data.empty and not new_data.empty:
            analysis['format_comparison'] = {
                'old_avg_score': old_data['Best Score (Old)'].mean(),
                'new_avg_total_fitness': new_data['Total Fitness'].mean(),
                'new_avg_co2_affinity': new_data['CO2 Affinity'].mean(),
                'improvement_in_methodology': True
            }
        
        # Performance trends for new format trials
        if not new_data.empty:
            analysis['performance_trends'] = {
                'co2_affinity': {
                    'mean': new_data['CO2 Affinity'].mean(),
                    'std': new_data['CO2 Affinity'].std(),
                    'max': new_data['CO2 Affinity'].max(),
                    'min': new_data['CO2 Affinity'].min()
                },
                'stability': {
                    'mean': new_data['Stability'].mean(),
                    'std': new_data['Stability'].std(),
                    'max': new_data['Stability'].max(),
                    'min': new_data['Stability'].min()
                },
                'total_fitness_correlation': self._analyze_fitness_correlations(new_data)
            }
        
        # Best performers
        if not self.comparison_data.empty:
            # For old format, use Best Score (Old)
            if not old_data.empty:
                best_old = old_data.loc[old_data['Best Score (Old)'].idxmax()]
                analysis['best_performers']['best_old_format'] = {
                    'trial': best_old['Trial'],
                    'score': best_old['Best Score (Old)'],
                    'sequence': best_old['Best Sequence']
                }
            
            # For new format, use Total Fitness
            if not new_data.empty:
                best_new = new_data.loc[new_data['Total Fitness'].idxmax()]
                analysis['best_performers']['best_new_format'] = {
                    'trial': best_new['Trial'],
                    'total_fitness': best_new['Total Fitness'],
                    'co2_affinity': best_new['CO2 Affinity'],
                    'sequence': best_new['Best Sequence']
                }
        
        return analysis
    
    def _analyze_fitness_correlations(self, data: pd.DataFrame) -> Dict:
        """Analyze correlations between fitness components"""
        fitness_cols = ['CO2 Affinity', 'Stability', 'Expression', 'Catalytic Efficiency']
        
        correlations = {}
        for i, col1 in enumerate(fitness_cols):
            for col2 in fitness_cols[i+1:]:
                if col1 in data.columns and col2 in data.columns:
                    corr, p_value = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
                    correlations[f"{col1}_vs_{col2}"] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        return correlations
    
    def perform_sequence_analysis(self) -> Dict:
        """Perform comprehensive sequence analysis across all trials"""
        if self.comparison_data.empty:
            self.create_comprehensive_comparison()
        
        sequences = self.comparison_data['Best Sequence'].dropna().tolist()
        
        analysis = {
            'amino_acid_frequency': {},
            'sequence_diversity': {},
            'motif_analysis': {},
            'length_distribution': {},
            'clustering_analysis': {}
        }
        
        # Amino acid frequency analysis
        all_aa_counts = {}
        for seq in sequences:
            for aa in seq:
                all_aa_counts[aa] = all_aa_counts.get(aa, 0) + 1
        
        total_aa = sum(all_aa_counts.values())
        analysis['amino_acid_frequency'] = {
            aa: count / total_aa for aa, count in all_aa_counts.items()
        }
        
        # Sequence diversity
        if len(sequences) > 1:
            diversity_scores = []
            for i, seq1 in enumerate(sequences):
                for seq2 in sequences[i+1:]:
                    similarity = self._calculate_sequence_similarity(seq1, seq2)
                    diversity_scores.append(1 - similarity)
            
            analysis['sequence_diversity'] = {
                'mean_diversity': np.mean(diversity_scores),
                'std_diversity': np.std(diversity_scores),
                'max_diversity': np.max(diversity_scores),
                'min_diversity': np.min(diversity_scores)
            }
        
        # Length distribution
        lengths = [len(seq) for seq in sequences]
        analysis['length_distribution'] = {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'length_distribution': np.histogram(lengths, bins=10)[0].tolist()
        }
        
        # Motif analysis (simple)
        co2_motifs = ['HHH', 'HEH', 'HDH', 'CXC', 'DXE']
        analysis['motif_analysis'] = {}
        for motif in co2_motifs:
            count = sum(1 for seq in sequences if motif in seq)
            analysis['motif_analysis'][motif] = {
                'count': count,
                'frequency': count / len(sequences) if sequences else 0
            }
        
        return analysis
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity"""
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        return matches / min_len
    
    def create_comprehensive_visualizations(self, save_dir: Optional[Path] = None) -> None:
        """Create comprehensive visualizations of all analyses"""
        if save_dir is None:
            save_dir = self.reports_path
        
        save_dir.mkdir(exist_ok=True)
        
        # 1. Overall performance comparison
        self._plot_performance_comparison(save_dir)
        
        # 2. Fitness component analysis
        self._plot_fitness_components(save_dir)
        
        # 3. Sequence analysis visualizations
        self._plot_sequence_analysis(save_dir)
        
        # 4. Timeline analysis
        self._plot_timeline_analysis(save_dir)
        
        # 5. Clustering analysis
        self._plot_clustering_analysis(save_dir)
        
        print(f"Visualizations saved to: {save_dir}")
    
    def _plot_performance_comparison(self, save_dir: Path):
        """Plot performance comparison across formats"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Comparison Across All Trials', fontsize=16)
        
        # 1. Score distribution by format
        old_data = self.comparison_data[self.comparison_data['Format'] == 'Old']
        new_data = self.comparison_data[self.comparison_data['Format'] == 'New']
        
        if not old_data.empty:
            axes[0, 0].hist(old_data['Best Score (Old)'].dropna(), alpha=0.7, label='Old Format', bins=20)
        if not new_data.empty:
            axes[0, 0].hist(new_data['Total Fitness'].dropna(), alpha=0.7, label='New Format', bins=20)
        
        axes[0, 0].set_xlabel('Score/Fitness')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution by Format')
        axes[0, 0].legend()
        
        # 2. Sequence length distribution
        axes[0, 1].hist(self.comparison_data['Sequence Length'], bins=20, color='steelblue', alpha=0.7)
        axes[0, 1].set_xlabel('Sequence Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Sequence Length Distribution')
        
        # 3. New format fitness components
        if not new_data.empty:
            fitness_cols = ['CO2 Affinity', 'Stability', 'Expression', 'Catalytic Efficiency']
            box_data = [new_data[col].dropna() for col in fitness_cols]
            axes[1, 0].boxplot(box_data, labels=fitness_cols)
            axes[1, 0].set_ylabel('Fitness Score')
            axes[1, 0].set_title('Fitness Components Distribution')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Performance vs sequence length
        if not new_data.empty:
            axes[1, 1].scatter(new_data['Sequence Length'], new_data['Total Fitness'], alpha=0.6)
            axes[1, 1].set_xlabel('Sequence Length')
            axes[1, 1].set_ylabel('Total Fitness')
            axes[1, 1].set_title('Performance vs Sequence Length')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_fitness_components(self, save_dir: Path):
        """Plot detailed fitness component analysis"""
        new_data = self.comparison_data[self.comparison_data['Format'] == 'New']
        
        if new_data.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fitness Components Analysis', fontsize=16)
        
        fitness_cols = ['CO2 Affinity', 'Stability', 'Expression', 'Catalytic Efficiency']
        
        # 1. Correlation heatmap
        corr_data = new_data[fitness_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
        axes[0, 0].set_title('Fitness Components Correlation')
        
        # 2. Principal Component Analysis
        if len(new_data) > 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(new_data[fitness_cols].fillna(0))
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
            axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[0, 1].set_title('PCA of Fitness Components')
        
        # 3. Fitness evolution over trials (if timestamps available)
        new_data_sorted = new_data.sort_values('Timestamp')
        if len(new_data_sorted) > 1:
            trial_numbers = range(len(new_data_sorted))
            for i, col in enumerate(fitness_cols):
                axes[1, 0].plot(trial_numbers, new_data_sorted[col], 
                               marker='o', label=col, alpha=0.7)
            
            axes[1, 0].set_xlabel('Trial Order')
            axes[1, 0].set_ylabel('Fitness Score')
            axes[1, 0].set_title('Fitness Evolution Over Trials')
            axes[1, 0].legend()
        
        # 4. Top performers radar chart
        if len(new_data) >= 5:
            top_5 = new_data.nlargest(5, 'Total Fitness')
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(fitness_cols), endpoint=False).tolist()
            angles += angles[:1]
            
            ax_radar = plt.subplot(2, 2, 4, projection='polar')
            
            for idx, (_, row) in enumerate(top_5.iterrows()):
                values = [row[col] for col in fitness_cols]
                values += values[:1]
                
                ax_radar.plot(angles, values, 'o-', linewidth=2, 
                             label=f"Trial {idx+1}", alpha=0.7)
                ax_radar.fill(angles, values, alpha=0.1)
            
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(fitness_cols)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('Top 5 Performers Comparison')
            ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'fitness_components_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_sequence_analysis(self, save_dir: Path):
        """Plot sequence analysis visualizations"""
        sequence_analysis = self.perform_sequence_analysis()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sequence Analysis Across All Trials', fontsize=16)
        
        # 1. Amino acid frequency
        aa_freq = sequence_analysis['amino_acid_frequency']
        if aa_freq:
            aa_names = list(aa_freq.keys())
            aa_frequencies = list(aa_freq.values())
            
            axes[0, 0].bar(aa_names, aa_frequencies, color='steelblue')
            axes[0, 0].set_xlabel('Amino Acid')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Amino Acid Frequency Across All Sequences')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Length distribution
        lengths = [len(seq) for seq in self.comparison_data['Best Sequence'].dropna()]
        if lengths:
            axes[0, 1].hist(lengths, bins=15, color='green', alpha=0.7)
            axes[0, 1].axvline(np.mean(lengths), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(lengths):.1f}')
            axes[0, 1].set_xlabel('Sequence Length')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Sequence Length Distribution')
            axes[0, 1].legend()
        
        # 3. Motif frequency
        motif_analysis = sequence_analysis['motif_analysis']
        if motif_analysis:
            motif_names = list(motif_analysis.keys())
            motif_frequencies = [data['frequency'] for data in motif_analysis.values()]
            
            axes[1, 0].bar(motif_names, motif_frequencies, color='orange')
            axes[1, 0].set_xlabel('Motif')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('CO2 Binding Motif Frequency')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Diversity analysis
        diversity = sequence_analysis.get('sequence_diversity', {})
        if diversity:
            diversity_metrics = ['mean_diversity', 'std_diversity', 'max_diversity', 'min_diversity']
            diversity_values = [diversity.get(metric, 0) for metric in diversity_metrics]
            diversity_labels = [metric.replace('_', ' ').title() for metric in diversity_metrics]
            
            axes[1, 1].bar(diversity_labels, diversity_values, color='purple')
            axes[1, 1].set_ylabel('Diversity Score')
            axes[1, 1].set_title('Sequence Diversity Metrics')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'sequence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_timeline_analysis(self, save_dir: Path):
        """Plot timeline analysis of trials"""
        # Convert timestamps and sort
        timeline_data = self.comparison_data.copy()
        timeline_data['Datetime'] = pd.to_datetime(timeline_data['Timestamp'])
        timeline_data = timeline_data.sort_values('Datetime')
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Timeline Analysis of Optimization Trials', fontsize=16)
        
        # 1. Performance over time
        new_data = timeline_data[timeline_data['Format'] == 'New']
        old_data = timeline_data[timeline_data['Format'] == 'Old']
        
        if not new_data.empty:
            axes[0].plot(new_data['Datetime'], new_data['Total Fitness'], 
                        'bo-', label='Total Fitness (New)', alpha=0.7)
            axes[0].plot(new_data['Datetime'], new_data['CO2 Affinity'], 
                        'ro-', label='CO2 Affinity', alpha=0.7)
        
        if not old_data.empty:
            axes[0].plot(old_data['Datetime'], old_data['Best Score (Old)'], 
                        'go-', label='Best Score (Old)', alpha=0.7)
        
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Performance Evolution Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Trial frequency over time
        timeline_data['Date'] = timeline_data['Datetime'].dt.date
        trial_counts = timeline_data.groupby('Date').size()
        
        axes[1].bar(trial_counts.index, trial_counts.values, alpha=0.7)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Number of Trials')
        axes[1].set_title('Trial Frequency Over Time')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'timeline_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_clustering_analysis(self, save_dir: Path):
        """Plot clustering analysis of sequences"""
        sequences = self.comparison_data['Best Sequence'].dropna().tolist()
        
        if len(sequences) < 3:
            print("Not enough sequences for clustering analysis")
            return
        
        # Calculate pairwise similarities
        similarity_matrix = np.zeros((len(sequences), len(sequences)))
        for i, seq1 in enumerate(sequences):
            for j, seq2 in enumerate(sequences):
                similarity_matrix[i, j] = self._calculate_sequence_similarity(seq1, seq2)
        
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        
        # Hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='average')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Sequence Clustering Analysis', fontsize=16)
        
        # 1. Dendrogram
        dendrogram(linkage_matrix, ax=axes[0], truncate_mode='level', p=10)
        axes[0].set_title('Hierarchical Clustering Dendrogram')
        axes[0].set_xlabel('Sequence Index')
        axes[0].set_ylabel('Distance')
        
        # 2. Similarity heatmap
        sns.heatmap(similarity_matrix, cmap='viridis', ax=axes[1])
        axes[1].set_title('Sequence Similarity Matrix')
        axes[1].set_xlabel('Sequence Index')
        axes[1].set_ylabel('Sequence Index')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.trial_data:
            self.load_all_trials()
        
        if self.comparison_data.empty:
            self.create_comprehensive_comparison()
        
        performance_analysis = self.analyze_performance_trends()
        sequence_analysis = self.perform_sequence_analysis()
        
        report = f"""
COMPREHENSIVE ALGAE PROTEIN OPTIMIZATION ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

EXECUTIVE SUMMARY:
This report analyzes {performance_analysis['overall_statistics']['total_trials']} optimization trials
for algae proteins targeting enhanced CO2 absorption capabilities.

TRIAL OVERVIEW:
- Total Trials: {performance_analysis['overall_statistics']['total_trials']}
- Old Format Trials: {performance_analysis['overall_statistics']['old_format_trials']}
- New Format Trials: {performance_analysis['overall_statistics']['new_format_trials']}
- Unique Protein Accessions: {performance_analysis['overall_statistics']['unique_accessions']}
- Average Sequence Length: {performance_analysis['overall_statistics']['avg_sequence_length']:.1f} ± {performance_analysis['overall_statistics']['sequence_length_std']:.1f}

"""
        
        # Performance analysis
        if 'performance_trends' in performance_analysis and performance_analysis['performance_trends']:
            trends = performance_analysis['performance_trends']
            report += f"""
PERFORMANCE ANALYSIS (New Format Trials):
CO2 Binding Affinity:
  - Mean: {trends['co2_affinity']['mean']:.4f}
  - Range: {trends['co2_affinity']['min']:.4f} - {trends['co2_affinity']['max']:.4f}
  - Standard Deviation: {trends['co2_affinity']['std']:.4f}

Protein Stability:
  - Mean: {trends['stability']['mean']:.4f}
  - Range: {trends['stability']['min']:.4f} - {trends['stability']['max']:.4f}
  - Standard Deviation: {trends['stability']['std']:.4f}

"""
        
        # Best performers
        if 'best_performers' in performance_analysis:
            report += "BEST PERFORMING TRIALS:\\n"
            
            if 'best_old_format' in performance_analysis['best_performers']:
                best_old = performance_analysis['best_performers']['best_old_format']
                report += f"Best Old Format Trial: {best_old['trial']} (Score: {best_old['score']:.4f})\\n"
            
            if 'best_new_format' in performance_analysis['best_performers']:
                best_new = performance_analysis['best_performers']['best_new_format']
                report += f"Best New Format Trial: {best_new['trial']} (Total Fitness: {best_new['total_fitness']:.4f})\\n"
                report += f"  - CO2 Affinity: {best_new['co2_affinity']:.4f}\\n"
        
        # Sequence analysis
        report += f"""

SEQUENCE ANALYSIS:
Most Common Amino Acids:
"""
        # Top 5 most frequent amino acids
        aa_freq = sequence_analysis['amino_acid_frequency']
        sorted_aa = sorted(aa_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        for aa, freq in sorted_aa:
            report += f"  - {aa}: {freq:.3f} ({freq*100:.1f}%)\\n"
        
        # Diversity metrics
        if 'sequence_diversity' in sequence_analysis:
            div = sequence_analysis['sequence_diversity']
            report += f"""
Sequence Diversity:
  - Mean Diversity Score: {div['mean_diversity']:.4f}
  - Diversity Range: {div['min_diversity']:.4f} - {div['max_diversity']:.4f}
"""
        
        # Motif analysis
        if 'motif_analysis' in sequence_analysis:
            report += "\\nCO2 Binding Motif Frequency:\\n"
            for motif, data in sequence_analysis['motif_analysis'].items():
                report += f"  - {motif}: {data['frequency']:.2%} of sequences\\n"
        
        # Recommendations
        report += f"""

RECOMMENDATIONS:
Based on the analysis of {performance_analysis['overall_statistics']['total_trials']} trials:

1. METHODOLOGY IMPROVEMENTS:
   {"✓ New multi-objective optimization shows improved results" if performance_analysis['overall_statistics']['new_format_trials'] > 0 else "• Consider implementing multi-objective optimization"}

2. SEQUENCE OPTIMIZATION:
"""
        
        # Add specific recommendations based on analysis
        if 'best_performers' in performance_analysis and 'best_new_format' in performance_analysis['best_performers']:
            best_fitness = performance_analysis['best_performers']['best_new_format']
            if best_fitness['co2_affinity'] > 0.7:
                report += "   ✓ Excellent CO2 binding affinity achieved in best trials\\n"
            else:
                report += "   • Focus on improving CO2 binding affinity\\n"
        
        # Check for common successful patterns
        if aa_freq:
            his_freq = aa_freq.get('H', 0)
            if his_freq > 0.1:
                report += "   ✓ Good histidine content for zinc binding\\n"
            else:
                report += "   • Increase histidine content for better zinc coordination\\n"
        
        report += f"""
3. FUTURE DIRECTIONS:
   • Continue optimization with successful sequence patterns
   • Investigate correlation between sequence features and performance
   • Consider experimental validation of top-performing sequences

TECHNICAL DETAILS:
- Analysis performed on {len(self.trial_data)} total trials
- Report generated using comprehensive multi-objective analysis
- Visualizations available in the Reports directory

END OF REPORT
{'='*80}
"""
        
        # Save report
        report_file = self.reports_path / f"comprehensive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Comprehensive report saved to: {report_file}")
        return report

def main():
    """Main function for comprehensive analysis"""
    workspace_path = r"c:\\Users\\Gunjan Ghimire\\Downloads\\Testting code\\Algae-Bioniformatics-Binding-Protein"
    analyzer = ComprehensiveAnalyzer(workspace_path)
    
    print("🔬 Comprehensive Analysis System")
    print("Analyzing all optimization trials...")
    print("=" * 60)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()