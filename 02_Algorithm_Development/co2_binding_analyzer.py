"""
Specialized CO2 Binding Analysis for Carbonic Anhydrase Optimization
Advanced analysis functions for predicting CO2 binding affinity and catalytic efficiency
"""

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

class CO2BindingAnalyzer:
    """Specialized analyzer for CO2 binding in carbonic anhydrase proteins"""
    
    def __init__(self):
        # Zinc-binding residues (essential for carbonic anhydrase activity)
        self.zinc_binding_residues = ['H']  # Histidine
        
        # Key residues for CO2 binding and catalysis
        self.catalytic_residues = {
            'proton_shuttle': ['H', 'D', 'E'],  # Histidine, Aspartic acid, Glutamic acid
            'zinc_coordination': ['H', 'C'],    # Histidine, Cysteine
            'substrate_binding': ['T', 'Y', 'W', 'F'],  # Threonine, Tyrosine, Tryptophan, Phenylalanine
            'structural_support': ['C', 'P'],   # Cysteine (disulfide), Proline (turns)
        }
        
        # Known carbonic anhydrase active site motifs
        self.ca_motifs = {
            'zinc_binding_core': r'H[A-Z]{1,3}H[A-Z]{1,3}H',  # Three histidines for zinc
            'catalytic_triad': r'[HDE][A-Z]{1,5}[HDE][A-Z]{1,5}[HDE]',
            'co2_binding_pocket': r'[TYWF][A-Z]{1,3}[HDE]',
            'proton_transfer': r'H[A-Z]{1,2}[ST][A-Z]{1,2}[DE]',
            'disulfide_bridge': r'C[A-Z]{2,10}C',
        }
        
        # Amino acid properties for CO2 binding prediction
        self.aa_co2_affinity = {
            'H': 0.9,  # High - zinc binding and proton transfer
            'D': 0.8,  # High - catalytic activity
            'E': 0.8,  # High - catalytic activity
            'C': 0.7,  # Good - structural support
            'T': 0.6,  # Good - substrate positioning
            'S': 0.6,  # Good - hydrogen bonding
            'Y': 0.7,  # Good - aromatic interactions
            'W': 0.6,  # Moderate - substrate binding
            'F': 0.5,  # Moderate - hydrophobic interactions
            'R': 0.5,  # Moderate - electrostatic interactions
            'K': 0.4,  # Low-moderate - electrostatic
            'N': 0.4,  # Low-moderate - hydrogen bonding
            'Q': 0.4,  # Low-moderate - hydrogen bonding
            'A': 0.3,  # Low - small and flexible
            'G': 0.3,  # Low - flexible
            'V': 0.2,  # Low - hydrophobic
            'L': 0.2,  # Low - hydrophobic
            'I': 0.2,  # Low - hydrophobic
            'M': 0.3,  # Low - hydrophobic
            'P': 0.4,  # Low-moderate - structural
        }
    
    def predict_co2_binding_affinity(self, sequence: str) -> Dict[str, float]:
        """Predict CO2 binding affinity based on sequence features"""
        
        # 1. Zinc binding capacity
        zinc_score = self._calculate_zinc_binding_score(sequence)
        
        # 2. Catalytic residue presence
        catalytic_score = self._calculate_catalytic_score(sequence)
        
        # 3. Active site motif presence
        motif_score = self._calculate_motif_score(sequence)
        
        # 4. Overall sequence affinity
        sequence_affinity = self._calculate_sequence_affinity(sequence)
        
        # 5. Spatial arrangement score (simplified)
        spatial_score = self._calculate_spatial_arrangement_score(sequence)
        
        # Combined score with weights
        weights = {
            'zinc_binding': 0.25,
            'catalytic': 0.25,
            'motifs': 0.20,
            'sequence_affinity': 0.15,
            'spatial': 0.15
        }
        
        overall_affinity = (
            zinc_score * weights['zinc_binding'] +
            catalytic_score * weights['catalytic'] +
            motif_score * weights['motifs'] +
            sequence_affinity * weights['sequence_affinity'] +
            spatial_score * weights['spatial']
        )
        
        return {
            'overall_affinity': overall_affinity,
            'zinc_binding_score': zinc_score,
            'catalytic_score': catalytic_score,
            'motif_score': motif_score,
            'sequence_affinity': sequence_affinity,
            'spatial_score': spatial_score,
            'detailed_analysis': self._detailed_sequence_analysis(sequence)
        }
    
    def _calculate_zinc_binding_score(self, sequence: str) -> float:
        """Calculate zinc binding capacity"""
        his_count = sequence.count('H')
        cys_count = sequence.count('C')
        
        # Ideal zinc binding: 3-4 histidines
        his_score = min(his_count / 3.0, 1.0) if his_count > 0 else 0.0
        
        # Additional coordination from cysteine
        cys_bonus = min(cys_count / 2.0, 0.2)
        
        return min(his_score + cys_bonus, 1.0)
    
    def _calculate_catalytic_score(self, sequence: str) -> float:
        """Calculate catalytic residue score"""
        scores = []
        
        for category, residues in self.catalytic_residues.items():
            count = sum(sequence.count(res) for res in residues)
            total_length = len(sequence)
            
            if category == 'proton_shuttle':
                # Critical for catalysis
                score = min(count / (total_length * 0.15), 1.0)  # ~15% ideal
            elif category == 'zinc_coordination':
                score = min(count / (total_length * 0.12), 1.0)  # ~12% ideal
            elif category == 'substrate_binding':
                score = min(count / (total_length * 0.20), 1.0)  # ~20% ideal
            else:
                score = min(count / (total_length * 0.10), 1.0)  # ~10% ideal
            
            scores.append(score)
        
        return np.mean(scores)
    
    def _calculate_motif_score(self, sequence: str) -> float:
        """Calculate active site motif score"""
        motif_scores = []
        
        for motif_name, pattern in self.ca_motifs.items():
            matches = len(re.findall(pattern, sequence))
            
            if motif_name == 'zinc_binding_core':
                # Most critical motif
                score = min(matches / 1.0, 1.0) * 2.0  # Weight heavily
            elif motif_name == 'catalytic_triad':
                score = min(matches / 1.0, 1.0) * 1.5
            else:
                score = min(matches / 1.0, 1.0)
            
            motif_scores.append(score)
        
        return min(np.mean(motif_scores), 1.0)
    
    def _calculate_sequence_affinity(self, sequence: str) -> float:
        """Calculate overall sequence affinity for CO2"""
        if not sequence:
            return 0.0
        
        total_affinity = sum(self.aa_co2_affinity.get(aa, 0.1) for aa in sequence)
        return total_affinity / len(sequence)
    
    def _calculate_spatial_arrangement_score(self, sequence: str) -> float:
        """Calculate spatial arrangement score (simplified)"""
        if len(sequence) < 5:
            return 0.0
        
        # Look for clustering of important residues
        important_residues = set(['H', 'D', 'E', 'C', 'T', 'Y'])
        
        cluster_scores = []
        window_size = 5
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            important_count = sum(1 for aa in window if aa in important_residues)
            cluster_scores.append(important_count / window_size)
        
        return max(cluster_scores) if cluster_scores else 0.0
    
    def _detailed_sequence_analysis(self, sequence: str) -> Dict:
        """Perform detailed sequence analysis"""
        analysis = {
            'length': len(sequence),
            'composition': {},
            'motif_positions': {},
            'critical_residue_positions': {},
            'hydrophobic_regions': [],
            'charged_regions': []
        }
        
        # Composition analysis
        for aa in set(sequence):
            analysis['composition'][aa] = sequence.count(aa)
        
        # Motif positions
        for motif_name, pattern in self.ca_motifs.items():
            matches = [(m.start(), m.end(), m.group()) for m in re.finditer(pattern, sequence)]
            analysis['motif_positions'][motif_name] = matches
        
        # Critical residue positions
        critical_residues = ['H', 'D', 'E', 'C']
        for residue in critical_residues:
            positions = [i for i, aa in enumerate(sequence) if aa == residue]
            analysis['critical_residue_positions'][residue] = positions
        
        # Hydrophobic and charged regions
        analysis['hydrophobic_regions'] = self._find_regions(sequence, ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'Y'])
        analysis['charged_regions'] = self._find_regions(sequence, ['D', 'E', 'K', 'R', 'H'])
        
        return analysis
    
    def _find_regions(self, sequence: str, target_residues: List[str], min_length: int = 3) -> List[Tuple[int, int]]:
        """Find regions rich in specific residues"""
        regions = []
        current_start = None
        
        for i, aa in enumerate(sequence):
            if aa in target_residues:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None and i - current_start >= min_length:
                    regions.append((current_start, i - 1))
                current_start = None
        
        # Handle region at end of sequence
        if current_start is not None and len(sequence) - current_start >= min_length:
            regions.append((current_start, len(sequence) - 1))
        
        return regions
    
    def calculate_catalytic_efficiency(self, sequence: str) -> Dict[str, float]:
        """Calculate predicted catalytic efficiency"""
        
        # 1. Active site accessibility
        accessibility = self._calculate_active_site_accessibility(sequence)
        
        # 2. Proton transfer efficiency
        proton_transfer = self._calculate_proton_transfer_efficiency(sequence)
        
        # 3. Substrate positioning
        substrate_positioning = self._calculate_substrate_positioning(sequence)
        
        # 4. Cofactor binding (zinc)
        cofactor_binding = self._calculate_zinc_binding_score(sequence)
        
        # 5. Structural stability
        structural_stability = self._calculate_structural_stability(sequence)
        
        # Overall efficiency
        efficiency_components = {
            'accessibility': accessibility,
            'proton_transfer': proton_transfer,
            'substrate_positioning': substrate_positioning,
            'cofactor_binding': cofactor_binding,
            'structural_stability': structural_stability
        }
        
        weights = [0.25, 0.25, 0.20, 0.20, 0.10]
        overall_efficiency = sum(score * weight for score, weight in zip(efficiency_components.values(), weights))
        
        return {
            'overall_efficiency': overall_efficiency,
            **efficiency_components
        }
    
    def _calculate_active_site_accessibility(self, sequence: str) -> float:
        """Calculate active site accessibility score"""
        # Look for flexible regions near critical residues
        flexible_residues = set(['G', 'A', 'S', 'T'])
        critical_residues = set(['H', 'D', 'E'])
        
        accessibility_score = 0.0
        count = 0
        
        for i, aa in enumerate(sequence):
            if aa in critical_residues:
                # Check surrounding residues for flexibility
                start = max(0, i - 2)
                end = min(len(sequence), i + 3)
                surrounding = sequence[start:end]
                
                flexible_count = sum(1 for res in surrounding if res in flexible_residues)
                accessibility_score += flexible_count / len(surrounding)
                count += 1
        
        return accessibility_score / count if count > 0 else 0.0
    
    def _calculate_proton_transfer_efficiency(self, sequence: str) -> float:
        """Calculate proton transfer pathway efficiency"""
        # Look for proton transfer networks
        proton_residues = ['H', 'D', 'E', 'S', 'T', 'Y']
        
        # Find clusters of proton transfer residues
        max_cluster_density = 0.0
        window_size = 7
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            proton_count = sum(1 for aa in window if aa in proton_residues)
            density = proton_count / window_size
            max_cluster_density = max(max_cluster_density, density)
        
        return max_cluster_density
    
    def _calculate_substrate_positioning(self, sequence: str) -> float:
        """Calculate substrate positioning score"""
        # Residues important for substrate binding and positioning
        positioning_residues = ['T', 'Y', 'W', 'F', 'H', 'N', 'Q']
        
        count = sum(1 for aa in sequence if aa in positioning_residues)
        return min(count / (len(sequence) * 0.3), 1.0)  # Optimal ~30%
    
    def _calculate_structural_stability(self, sequence: str) -> float:
        """Calculate structural stability score"""
        # Factors contributing to stability
        disulfide_potential = sequence.count('C') / 2  # Pairs for disulfide bonds
        proline_content = sequence.count('P') / len(sequence)  # Structural rigidity
        hydrophobic_core = sum(1 for aa in sequence if aa in ['A', 'V', 'L', 'I', 'M', 'F', 'W'])
        
        # Normalize scores
        disulfide_score = min(disulfide_potential / 2.0, 1.0)  # Up to 2 disulfide bonds ideal
        proline_score = min(proline_content / 0.05, 1.0)  # ~5% proline ideal
        hydrophobic_score = min(hydrophobic_core / (len(sequence) * 0.4), 1.0)  # ~40% hydrophobic
        
        return (disulfide_score + proline_score + hydrophobic_score) / 3.0
    
    def compare_with_known_ca(self, sequence: str, reference_ca_sequences: List[str]) -> Dict:
        """Compare sequence with known carbonic anhydrase sequences"""
        if not reference_ca_sequences:
            return {'average_similarity': 0.0, 'best_match': 0.0, 'similarities': []}
        
        similarities = []
        for ref_seq in reference_ca_sequences:
            similarity = self._calculate_sequence_similarity(sequence, ref_seq)
            similarities.append(similarity)
        
        return {
            'average_similarity': np.mean(similarities),
            'best_match': max(similarities),
            'worst_match': min(similarities),
            'similarities': similarities,
            'similarity_std': np.std(similarities)
        }
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity"""
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0
        
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        return matches / min_len
    
    def generate_optimization_suggestions(self, sequence: str) -> List[str]:
        """Generate suggestions for optimizing CO2 binding"""
        suggestions = []
        
        # Analyze current sequence
        affinity_analysis = self.predict_co2_binding_affinity(sequence)
        catalytic_analysis = self.calculate_catalytic_efficiency(sequence)
        
        # Check zinc binding
        if affinity_analysis['zinc_binding_score'] < 0.7:
            his_count = sequence.count('H')
            if his_count < 3:
                suggestions.append(f"Increase histidine content for zinc binding (current: {his_count}, recommended: 3-4)")
        
        # Check catalytic residues
        if affinity_analysis['catalytic_score'] < 0.6:
            de_count = sequence.count('D') + sequence.count('E')
            if de_count < 3:
                suggestions.append(f"Add more catalytic residues (D/E) (current: {de_count}, recommended: 3-5)")
        
        # Check motifs
        if affinity_analysis['motif_score'] < 0.5:
            suggestions.append("Consider incorporating known carbonic anhydrase motifs (HxxHxxH for zinc binding)")
        
        # Check structural elements
        if catalytic_analysis['structural_stability'] < 0.6:
            cys_count = sequence.count('C')
            if cys_count < 2:
                suggestions.append(f"Add cysteine residues for disulfide bonds (current: {cys_count}, recommended: 2-4)")
        
        # Check proton transfer
        if catalytic_analysis['proton_transfer'] < 0.5:
            suggestions.append("Improve proton transfer pathway with more H/D/E/S/T residues in proximity")
        
        return suggestions
    
    def visualize_sequence_analysis(self, sequence: str, save_path: str = None):
        """Create comprehensive visualization of sequence analysis"""
        
        # Perform analyses
        affinity_data = self.predict_co2_binding_affinity(sequence)
        catalytic_data = self.calculate_catalytic_efficiency(sequence)
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'CO2 Binding Analysis - Sequence Length: {len(sequence)}', fontsize=16)
        
        # 1. Affinity components radar chart
        categories = ['Zinc Binding', 'Catalytic', 'Motifs', 'Sequence Affinity', 'Spatial']
        values = [
            affinity_data['zinc_binding_score'],
            affinity_data['catalytic_score'],
            affinity_data['motif_score'],
            affinity_data['sequence_affinity'],
            affinity_data['spatial_score']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 3, 1, projection='polar')
        ax_radar.plot(angles, values, 'o-', linewidth=2, label='Current Sequence')
        ax_radar.fill(angles, values, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('CO2 Binding Affinity Components')
        
        # 2. Catalytic efficiency components
        cat_categories = list(catalytic_data.keys())[1:]  # Exclude overall
        cat_values = [catalytic_data[key] for key in cat_categories]
        
        axes[0, 1].bar(range(len(cat_categories)), cat_values, color='steelblue')
        axes[0, 1].set_xticks(range(len(cat_categories)))
        axes[0, 1].set_xticklabels([cat.replace('_', ' ').title() for cat in cat_categories], rotation=45)
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Catalytic Efficiency Components')
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Amino acid composition pie chart
        composition = {aa: sequence.count(aa) for aa in set(sequence)}
        important_aas = {aa: count for aa, count in composition.items() if aa in ['H', 'D', 'E', 'C', 'T', 'Y']}
        other_count = sum(count for aa, count in composition.items() if aa not in important_aas)
        
        if other_count > 0:
            important_aas['Others'] = other_count
        
        axes[0, 2].pie(important_aas.values(), labels=important_aas.keys(), autopct='%1.1f%%', startangle=90)
        axes[0, 2].set_title('Key Amino Acid Composition')
        
        # 4. Position-wise CO2 affinity
        position_affinities = [self.aa_co2_affinity.get(aa, 0.1) for aa in sequence]
        axes[1, 0].plot(range(len(sequence)), position_affinities, 'g-', alpha=0.7)
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('CO2 Affinity')
        axes[1, 0].set_title('Position-wise CO2 Binding Affinity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Critical residue distribution
        critical_positions = {'H': [], 'D': [], 'E': [], 'C': []}
        for i, aa in enumerate(sequence):
            if aa in critical_positions:
                critical_positions[aa].append(i)
        
        colors = {'H': 'red', 'D': 'blue', 'E': 'green', 'C': 'orange'}
        for aa, positions in critical_positions.items():
            if positions:
                axes[1, 1].scatter(positions, [aa] * len(positions), 
                                 c=colors[aa], s=50, alpha=0.7, label=f'{aa} ({len(positions)})')
        
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Residue Type')
        axes[1, 1].set_title('Critical Residue Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Overall scores summary
        scores = {
            'Overall CO2 Affinity': affinity_data['overall_affinity'],
            'Overall Catalytic Efficiency': catalytic_data['overall_efficiency'],
            'Zinc Binding': affinity_data['zinc_binding_score'],
            'Structural Stability': catalytic_data['structural_stability']
        }
        
        score_names = list(scores.keys())
        score_values = list(scores.values())
        colors_bar = ['red', 'blue', 'green', 'orange']
        
        bars = axes[1, 2].bar(range(len(score_names)), score_values, color=colors_bar)
        axes[1, 2].set_xticks(range(len(score_names)))
        axes[1, 2].set_xticklabels([name.replace(' ', '\\n') for name in score_names])
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Overall Performance Scores')
        axes[1, 2].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, score_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return affinity_data, catalytic_data

def main():
    """Example usage of CO2 binding analyzer"""
    analyzer = CO2BindingAnalyzer()
    
    # Example sequence (from your carbonic anhydrase)
    example_sequence = "MRVAAALLALAVCANACSHVYFADSDLHAHGRRLTAAEGPTWNYNKGGSDWPGTCASGNK"
    
    print("CO2 Binding Analysis Example")
    print("=" * 50)
    print(f"Sequence: {example_sequence}")
    print()
    
    # Perform analysis
    affinity_results = analyzer.predict_co2_binding_affinity(example_sequence)
    catalytic_results = analyzer.calculate_catalytic_efficiency(example_sequence)
    suggestions = analyzer.generate_optimization_suggestions(example_sequence)
    
    print("CO2 Binding Affinity Analysis:")
    for key, value in affinity_results.items():
        if key != 'detailed_analysis':
            print(f"  {key}: {value:.4f}")
    
    print("\\nCatalytic Efficiency Analysis:")
    for key, value in catalytic_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\\nOptimization Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()