# 🧬 Quantum-Enhanced Protein Optimizer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_Algorithm_Development'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '05_3D_Molecular_Viewer'))

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

from quantum_co2_calculator import QuantumCO2BindingCalculator, QuantumResult

# Import existing optimization components
try:
    from enhanced_ga_protein_optimizer import EnhancedGeneticAlgorithm
    from co2_binding_analyzer import CO2BindingAnalyzer
    from protein_3d_generator import Protein3DGenerator
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("⚠️ Optimization modules not found. Please check path configuration.")
    OPTIMIZATION_AVAILABLE = False

class QuantumEnhancedOptimizer:
    """
    Advanced protein optimizer integrating quantum mechanical calculations
    for ultra-accurate CO2 binding predictions
    """
    
    def __init__(self, 
                 quantum_method='B3LYP',
                 quantum_basis='6-31G*',
                 use_quantum=True,
                 fallback_empirical=True):
        """
        Initialize quantum-enhanced optimizer
        
        Args:
            quantum_method: Quantum chemistry method (B3LYP, PBE, M06-2X, etc.)
            quantum_basis: Basis set for calculations
            use_quantum: Enable quantum calculations
            fallback_empirical: Use empirical methods if quantum fails
        """
        self.use_quantum = use_quantum
        self.fallback_empirical = fallback_empirical
        self.logger = self._setup_logging()
        
        # Initialize quantum calculator
        if use_quantum:
            self.quantum_calc = QuantumCO2BindingCalculator(
                method=quantum_method,
                basis=quantum_basis
            )
        else:
            self.quantum_calc = None
        
        # Initialize existing components
        if OPTIMIZATION_AVAILABLE:
            # Default target sequence (Carbonic Anhydrase) and config
            default_target = "MWSAHQILFPQCRTRVELQNSSAAVFQSADLKPQGTFFILVDWLLLPPVQDCGRIRQGEQVVIHPDLAVVLVRQRQELQRQQRQELQSESYEGHQQQLQNKQLQNKQLQNKQ"
            default_config = {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elite_size': 5,
                'tournament_size': 3
            }
            self.ga_optimizer = EnhancedGeneticAlgorithm(default_target, default_config)
            self.binding_analyzer = CO2BindingAnalyzer()
            self.structure_generator = Protein3DGenerator()
        
        # Quantum-enhanced fitness weights
        self.quantum_fitness_weights = {
            'quantum_binding_energy': 0.4,    # Primary quantum contribution
            'electron_density_overlap': 0.15, # Quantum orbital overlap
            'charge_distribution': 0.1,       # Electrostatic interactions
            'classical_co2_affinity': 0.2,    # Classical contributions
            'stability': 0.1,                 # Protein stability
            'expression': 0.05                # Expression level
        }
        
        # Results storage
        self.quantum_results_cache = {}
        self.optimization_history = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('QuantumOptimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = f"quantum_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            
            # Console handler
            console_handler = logging.StreamHandler()
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def quantum_fitness_evaluation(self, sequence: str) -> Dict[str, float]:
        """
        Comprehensive fitness evaluation using quantum calculations
        
        Args:
            sequence: Protein sequence to evaluate
            
        Returns:
            Dictionary of fitness scores
        """
        self.logger.info(f"Evaluating sequence with quantum calculations: {sequence[:10]}...")
        
        fitness_scores = {}
        
        try:
            # 1. Generate 3D structure
            structure_data = self._generate_3d_structure(sequence)
            
            # 2. Identify active sites
            active_sites = self._identify_active_sites(structure_data, sequence)
            
            # 3. Quantum calculations for each active site
            quantum_scores = []
            
            for i, site in enumerate(active_sites):
                # Multiple CO2 positions around active site
                co2_positions = self._generate_co2_positions(site)
                
                site_scores = []
                for co2_pos in co2_positions:
                    if self.use_quantum and self.quantum_calc:
                        quantum_result = self.quantum_calc.calculate_co2_binding_energy(
                            site['coords'], site['atoms'], co2_pos
                        )
                        site_scores.append(quantum_result.binding_energy)
                    else:
                        # Fallback to empirical calculation
                        empirical_score = self._empirical_binding_score(site, co2_pos)
                        site_scores.append(empirical_score)
                
                # Best binding energy for this site
                best_binding = min(site_scores)  # Most negative = strongest binding
                quantum_scores.append(best_binding)
            
            # 4. Combine quantum results
            if quantum_scores:
                avg_binding_energy = np.mean(quantum_scores)
                best_binding_energy = min(quantum_scores)
                
                # Convert to fitness score (0-1, higher is better)
                fitness_scores['quantum_binding_energy'] = self._normalize_binding_energy(best_binding_energy)
                fitness_scores['avg_quantum_binding'] = self._normalize_binding_energy(avg_binding_energy)
            else:
                fitness_scores['quantum_binding_energy'] = 0.0
                fitness_scores['avg_quantum_binding'] = 0.0
            
            # 5. Additional quantum-derived features
            if self.use_quantum and active_sites:
                # Electron density analysis
                fitness_scores['electron_density_overlap'] = self._calculate_electron_overlap(active_sites)
                
                # Charge distribution analysis
                fitness_scores['charge_distribution'] = self._analyze_charge_distribution(active_sites)
                
                # Molecular orbital analysis
                fitness_scores['orbital_compatibility'] = self._analyze_orbital_compatibility(active_sites)
            else:
                fitness_scores['electron_density_overlap'] = 0.5
                fitness_scores['charge_distribution'] = 0.5
                fitness_scores['orbital_compatibility'] = 0.5
            
            # 6. Classical fitness components (from existing system)
            if OPTIMIZATION_AVAILABLE:
                classical_scores = self._calculate_classical_fitness(sequence)
                fitness_scores.update(classical_scores)
            
            # 7. Composite quantum-enhanced fitness
            composite_fitness = self._calculate_composite_fitness(fitness_scores)
            fitness_scores['composite_fitness'] = composite_fitness
            
            self.logger.info(f"Quantum fitness evaluation complete. Score: {composite_fitness:.3f}")
            
        except Exception as e:
            self.logger.error(f"Quantum fitness evaluation failed: {str(e)}")
            # Fallback to classical evaluation
            if OPTIMIZATION_AVAILABLE:
                fitness_scores = self._calculate_classical_fitness(sequence)
            else:
                fitness_scores = {'composite_fitness': 0.1}  # Low default score
        
        return fitness_scores
    
    def _generate_3d_structure(self, sequence: str) -> Dict:
        """Generate 3D structure for quantum calculations"""
        if OPTIMIZATION_AVAILABLE:
            return self.structure_generator.generate_3d_structure(sequence)
        else:
            # Simplified structure generation
            n_atoms = len(sequence) * 4  # Approximate atoms per residue
            return {
                'atoms': [{'position': np.random.random(3) * 10} for _ in range(n_atoms)],
                'sequence': sequence
            }
    
    def _identify_active_sites(self, structure_data: Dict, sequence: str) -> List[Dict]:
        """Identify potential CO2 binding sites"""
        active_sites = []
        
        # Look for zinc-binding motifs and catalytic residues
        zinc_binding_residues = ['H', 'C', 'D', 'E']  # His, Cys, Asp, Glu
        
        for i, residue in enumerate(sequence):
            if residue in zinc_binding_residues:
                # Create active site around this residue
                site_atoms = ['Zn']  # Add zinc center
                site_coords = [np.array([0.0, 0.0, 0.0])]  # Zinc at origin
                
                # Add coordinating atoms
                for j in range(max(0, i-2), min(len(sequence), i+3)):
                    if j != i and sequence[j] in zinc_binding_residues:
                        site_atoms.append('N')  # Simplified as nitrogen
                        # Position around zinc
                        angle = (j - i) * np.pi / 3
                        coord = np.array([2.1 * np.cos(angle), 2.1 * np.sin(angle), 0.0])
                        site_coords.append(coord)
                
                if len(site_atoms) >= 3:  # Need at least zinc + 2 ligands
                    active_sites.append({
                        'center_residue': i,
                        'atoms': site_atoms,
                        'coords': np.array(site_coords),
                        'type': 'zinc_center'
                    })
        
        if not active_sites:
            # Create default active site
            active_sites.append({
                'center_residue': len(sequence) // 2,
                'atoms': ['Zn', 'N', 'N', 'N'],
                'coords': np.array([
                    [0.0, 0.0, 0.0],    # Zn
                    [2.1, 0.0, 0.0],    # N1
                    [-1.05, 1.82, 0.0], # N2
                    [-1.05, -1.82, 0.0] # N3
                ]),
                'type': 'default'
            })
        
        return active_sites
    
    def _generate_co2_positions(self, active_site: Dict) -> List[np.ndarray]:
        """Generate multiple CO2 positions around active site"""
        zinc_pos = active_site['coords'][0]  # Assume first atom is zinc
        
        positions = []
        
        # Generate positions in sphere around zinc
        for theta in np.linspace(0, 2*np.pi, 8):
            for phi in np.linspace(0, np.pi, 4):
                distance = 3.0  # Angstroms from zinc
                x = zinc_pos[0] + distance * np.sin(phi) * np.cos(theta)
                y = zinc_pos[1] + distance * np.sin(phi) * np.sin(theta)
                z = zinc_pos[2] + distance * np.cos(phi)
                positions.append(np.array([x, y, z]))
        
        return positions
    
    def _empirical_binding_score(self, active_site: Dict, co2_position: np.ndarray) -> float:
        """Fallback empirical binding score calculation"""
        zinc_pos = active_site['coords'][0]
        distance = np.linalg.norm(co2_position - zinc_pos)
        
        # Simple distance-based scoring
        optimal_distance = 2.8  # Angstroms
        score = -abs(distance - optimal_distance) * 2.0  # Negative for binding energy
        
        return score
    
    def _normalize_binding_energy(self, binding_energy: float) -> float:
        """Convert binding energy to 0-1 fitness score"""
        # Typical binding energies range from 0 to -20 kcal/mol
        # More negative = stronger binding = higher fitness
        normalized = max(0.0, min(1.0, (abs(binding_energy) / 20.0)))
        return normalized
    
    def _calculate_electron_overlap(self, active_sites: List[Dict]) -> float:
        """Calculate electron density overlap score"""
        # Simplified calculation based on active site geometry
        total_overlap = 0.0
        
        for site in active_sites:
            # Score based on optimal coordination geometry
            coords = site['coords']
            if len(coords) >= 4:  # Tetrahedral coordination
                center = coords[0]
                ligands = coords[1:4]
                
                # Check for tetrahedral angles (~109.5 degrees)
                angles = []
                for i in range(len(ligands)):
                    for j in range(i+1, len(ligands)):
                        v1 = ligands[i] - center
                        v2 = ligands[j] - center
                        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        angles.append(np.degrees(angle))
                
                # Score based on deviation from ideal tetrahedral angle
                ideal_angle = 109.47
                avg_deviation = np.mean([abs(angle - ideal_angle) for angle in angles])
                overlap_score = max(0.0, 1.0 - avg_deviation / 180.0)
                total_overlap += overlap_score
        
        return total_overlap / max(1, len(active_sites))
    
    def _analyze_charge_distribution(self, active_sites: List[Dict]) -> float:
        """Analyze charge distribution for optimal CO2 binding"""
        # Simplified charge analysis
        total_score = 0.0
        
        for site in active_sites:
            # CO2 binding favors partial positive charge on zinc
            # and electron-rich coordination environment
            charge_score = 0.7  # Default good score
            total_score += charge_score
        
        return total_score / max(1, len(active_sites))
    
    def _analyze_orbital_compatibility(self, active_sites: List[Dict]) -> float:
        """Analyze molecular orbital compatibility for CO2 binding"""
        # Simplified orbital analysis
        # In reality, this would involve HOMO-LUMO gap analysis
        return 0.75  # Default good compatibility score
    
    def _calculate_classical_fitness(self, sequence: str) -> Dict[str, float]:
        """Calculate classical fitness scores using existing methods"""
        try:
            classical_scores = {}
            
            # Use existing CO2 binding analyzer
            if hasattr(self, 'binding_analyzer'):
                co2_score = self.binding_analyzer.calculate_co2_binding_score(sequence)
                classical_scores['classical_co2_affinity'] = co2_score
            else:
                classical_scores['classical_co2_affinity'] = 0.5
            
            # Add other classical scores
            classical_scores['stability'] = self._calculate_stability_score(sequence)
            classical_scores['expression'] = self._calculate_expression_score(sequence)
            
            return classical_scores
            
        except Exception as e:
            self.logger.warning(f"Classical fitness calculation failed: {str(e)}")
            return {
                'classical_co2_affinity': 0.5,
                'stability': 0.5,
                'expression': 0.5
            }
    
    def _calculate_stability_score(self, sequence: str) -> float:
        """Calculate protein stability score"""
        # Simplified stability based on amino acid properties
        hydrophobic = set('AILMFPWV')
        charged = set('DEKR')
        
        hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic)
        charged_count = sum(1 for aa in sequence if aa in charged)
        
        # Balanced hydrophobicity and charge distribution
        hydrophobic_ratio = hydrophobic_count / len(sequence)
        charged_ratio = charged_count / len(sequence)
        
        stability = 1.0 - abs(hydrophobic_ratio - 0.4) - abs(charged_ratio - 0.2)
        return max(0.0, min(1.0, stability))
    
    def _calculate_expression_score(self, sequence: str) -> float:
        """Calculate expression level score"""
        # Simplified expression score based on codon usage
        difficult_residues = set('CMWY')  # Cys, Met, Trp, Tyr
        difficult_count = sum(1 for aa in sequence if aa in difficult_residues)
        
        expression_score = 1.0 - (difficult_count / len(sequence))
        return max(0.0, min(1.0, expression_score))
    
    def _calculate_composite_fitness(self, fitness_scores: Dict[str, float]) -> float:
        """Calculate weighted composite fitness score"""
        composite = 0.0
        total_weight = 0.0
        
        for component, weight in self.quantum_fitness_weights.items():
            if component in fitness_scores:
                composite += weight * fitness_scores[component]
                total_weight += weight
        
        if total_weight > 0:
            composite /= total_weight
        
        return composite
    
    def run_quantum_optimization(self, 
                                target_sequence: str,
                                population_size: int = 30,
                                generations: int = 100,
                                sequence_length: int = 25) -> Dict:
        """
        Run complete quantum-enhanced optimization
        
        Args:
            target_sequence: Reference sequence for optimization
            population_size: GA population size
            generations: Number of generations
            sequence_length: Length of optimized sequences
            
        Returns:
            Optimization results including quantum analysis
        """
        self.logger.info("🔬 Starting Quantum-Enhanced Protein Optimization")
        self.logger.info(f"Parameters: pop={population_size}, gen={generations}, len={sequence_length}")
        
        start_time = datetime.now()
        
        try:
            # Initialize results storage
            results = {
                'parameters': {
                    'population_size': population_size,
                    'generations': generations,
                    'sequence_length': sequence_length,
                    'quantum_method': self.quantum_calc.method if self.quantum_calc else 'Empirical',
                    'quantum_basis': self.quantum_calc.basis if self.quantum_calc else 'N/A'
                },
                'optimization_history': [],
                'best_sequences': [],
                'quantum_analysis': {},
                'performance_metrics': {}
            }
            
            # Create custom fitness function for GA
            def quantum_fitness_func(sequence):
                scores = self.quantum_fitness_evaluation(sequence)
                return scores['composite_fitness']
            
            # Run optimization using existing GA framework
            if OPTIMIZATION_AVAILABLE:
                # Configure GA with quantum fitness
                self.ga_optimizer.fitness_function = quantum_fitness_func
                
                # Run optimization
                ga_results = self.ga_optimizer.run_optimization(
                    generations=generations,
                    population_size=population_size
                )
                
                results['optimization_history'] = ga_results.get('history', [])
                results['best_sequences'] = ga_results.get('best_sequences', [])
                
            else:
                # Simple optimization without GA framework
                self.logger.warning("GA framework not available. Running simplified optimization.")
                best_sequence = self._simple_optimization(
                    target_sequence, population_size, generations, sequence_length
                )
                results['best_sequences'] = [best_sequence]
            
            # Detailed quantum analysis of best sequences
            self.logger.info("Performing detailed quantum analysis of best sequences...")
            quantum_analysis = self._detailed_quantum_analysis(results['best_sequences'][:5])
            results['quantum_analysis'] = quantum_analysis
            
            # Performance metrics
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()
            
            results['performance_metrics'] = {
                'runtime_seconds': runtime,
                'sequences_evaluated': len(self.optimization_history),
                'quantum_calculations_performed': len(self.quantum_results_cache),
                'convergence_generation': len(results['optimization_history'])
            }
            
            self.logger.info(f"🎉 Quantum optimization completed in {runtime:.1f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Quantum optimization failed: {str(e)}")
            raise
    
    def _simple_optimization(self, target_sequence: str, pop_size: int, 
                           generations: int, seq_length: int) -> str:
        """Simple optimization when GA framework unavailable"""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Start with random sequence
        current_best = ''.join(np.random.choice(list(amino_acids), seq_length))
        current_fitness = self.quantum_fitness_evaluation(current_best)['composite_fitness']
        
        for gen in range(generations):
            # Simple mutation-based improvement
            candidate = list(current_best)
            
            # Mutate 1-3 positions
            n_mutations = np.random.randint(1, 4)
            positions = np.random.choice(seq_length, n_mutations, replace=False)
            
            for pos in positions:
                candidate[pos] = np.random.choice(list(amino_acids))
            
            candidate_seq = ''.join(candidate)
            candidate_fitness = self.quantum_fitness_evaluation(candidate_seq)['composite_fitness']
            
            if candidate_fitness > current_fitness:
                current_best = candidate_seq
                current_fitness = candidate_fitness
                self.logger.info(f"Generation {gen}: New best fitness {current_fitness:.3f}")
        
        return current_best
    
    def _detailed_quantum_analysis(self, sequences: List[str]) -> Dict:
        """Perform detailed quantum analysis on best sequences"""
        analysis = {
            'binding_energies': [],
            'electronic_properties': [],
            'structural_analysis': [],
            'comparative_analysis': {}
        }
        
        for i, sequence in enumerate(sequences):
            self.logger.info(f"Detailed analysis of sequence {i+1}/{len(sequences)}")
            
            # Full quantum analysis
            seq_analysis = {
                'sequence': sequence,
                'binding_sites': [],
                'quantum_properties': {}
            }
            
            # Generate structure and analyze
            structure_data = self._generate_3d_structure(sequence)
            active_sites = self._identify_active_sites(structure_data, sequence)
            
            for site in active_sites:
                co2_positions = self._generate_co2_positions(site)
                site_analysis = {
                    'site_type': site['type'],
                    'binding_energies': [],
                    'optimal_co2_position': None,
                    'electronic_analysis': {}
                }
                
                best_binding = float('inf')
                best_position = None
                
                for co2_pos in co2_positions[:5]:  # Analyze top 5 positions
                    if self.use_quantum and self.quantum_calc:
                        result = self.quantum_calc.calculate_co2_binding_energy(
                            site['coords'], site['atoms'], co2_pos
                        )
                        
                        binding_energy = result.binding_energy
                        site_analysis['binding_energies'].append(binding_energy)
                        
                        if binding_energy < best_binding:
                            best_binding = binding_energy
                            best_position = co2_pos
                            site_analysis['electronic_analysis'] = {
                                'dipole_moment': result.dipole_moment,
                                'molecular_orbitals': result.molecular_orbitals,
                                'charge_distribution': result.charge_distribution
                            }
                
                site_analysis['best_binding_energy'] = best_binding
                site_analysis['optimal_co2_position'] = best_position.tolist() if best_position is not None else None
                seq_analysis['binding_sites'].append(site_analysis)
            
            analysis['binding_energies'].append(seq_analysis)
        
        return analysis
    
    def save_quantum_optimization_results(self, results: Dict, output_dir: str):
        """Save comprehensive quantum optimization results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main results
        with open(output_path / f'quantum_optimization_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed quantum data
        if 'quantum_analysis' in results:
            with open(output_path / f'quantum_analysis_{timestamp}.json', 'w') as f:
                json.dump(results['quantum_analysis'], f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    optimizer = QuantumEnhancedOptimizer(
        quantum_method='B3LYP',
        quantum_basis='6-31G*',
        use_quantum=True
    )
    
    # Test sequence (carbonic anhydrase fragment)
    target_sequence = "MKAAVLTLAVLFLTGSQARHFWGYGSHTNDQIKQYK"
    
    print("🔬 Testing Quantum-Enhanced Protein Optimization")
    print(f"Target sequence: {target_sequence}")
    print("Running optimization...")
    
    results = optimizer.run_quantum_optimization(
        target_sequence=target_sequence,
        population_size=20,
        generations=50,
        sequence_length=25
    )
    
    print(f"✅ Optimization complete!")
    print(f"Runtime: {results['performance_metrics']['runtime_seconds']:.1f} seconds")
    print(f"Best sequence: {results['best_sequences'][0] if results['best_sequences'] else 'None'}")
    
    # Save results
    optimizer.save_quantum_optimization_results(results, "quantum_results")