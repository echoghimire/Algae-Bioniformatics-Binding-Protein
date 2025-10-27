# 📚 API Reference & Code Documentation

## System Architecture Overview

The Algae Protein Optimization System follows a modular architecture with clear separation of concerns across five major development phases.

## Phase 1: Original Research (`01_Original_Research/`)

### `runfile.py`
**Purpose:** Initial genetic algorithm implementation for protein optimization

#### Core Functions:
```python
def genetic_algorithm(target_sequence, population_size=50, generations=100):
    """
    Basic genetic algorithm for protein sequence optimization
    
    Args:
        target_sequence (str): Target protein sequence
        population_size (int): Size of the population
        generations (int): Number of generations to run
        
    Returns:
        tuple: (best_sequence, fitness_score)
    """

def fitness_function(sequence, target):
    """
    Evaluate fitness of a protein sequence
    
    Args:
        sequence (str): Protein sequence to evaluate
        target (str): Target sequence for comparison
        
    Returns:
        float: Fitness score (0-1)
    """

def mutate_sequence(sequence, mutation_rate=0.1):
    """
    Apply mutations to a protein sequence
    
    Args:
        sequence (str): Input protein sequence
        mutation_rate (float): Probability of mutation per residue
        
    Returns:
        str: Mutated sequence
    """
```

## Phase 2: Algorithm Development (`02_Algorithm_Development/`)

### `enhanced_ga_protein_optimizer.py`
**Purpose:** Multi-objective genetic algorithm using NSGA-II

#### Main Classes:
```python
class EnhancedGeneticAlgorithm:
    """
    Advanced genetic algorithm with multi-objective optimization
    """
    
    def __init__(self, sequence_length=25, population_size=50):
        """
        Initialize the genetic algorithm
        
        Args:
            sequence_length (int): Length of protein sequences to optimize
            population_size (int): Size of the population
        """
    
    def create_fitness_functions(self):
        """
        Create multi-objective fitness functions
        
        Returns:
            list: List of fitness function objects
        """
    
    def run_optimization(self, generations=150):
        """
        Run the complete optimization process
        
        Args:
            generations (int): Number of generations to run
            
        Returns:
            dict: Optimization results and statistics
        """
```

#### Key Functions:
```python
def evaluate_co2_affinity(sequence):
    """
    Evaluate CO2 binding affinity of a protein sequence
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: CO2 affinity score (0-1)
    """

def evaluate_protein_stability(sequence):
    """
    Assess protein stability based on physicochemical properties
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Stability score (0-1)
    """

def evaluate_expression_level(sequence):
    """
    Predict protein expression level in biological systems
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Expression level score (0-1)
    """

def evaluate_catalytic_efficiency(sequence):
    """
    Evaluate catalytic efficiency for CO2 conversion
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Catalytic efficiency score (0-1)
    """
```

### `co2_binding_analyzer.py`
**Purpose:** Specialized CO2 binding site analysis

#### Core Functions:
```python
def analyze_zinc_binding_sites(sequence):
    """
    Identify and analyze zinc binding sites in protein sequence
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        dict: Zinc binding analysis results
    """

def find_catalytic_motifs(sequence):
    """
    Find known catalytic motifs for CO2 binding
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        list: Found motifs with positions and scores
    """

def calculate_co2_binding_score(sequence):
    """
    Calculate comprehensive CO2 binding score
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        float: Overall CO2 binding score
    """
```

## Phase 3: Visualization Systems (`03_Visualization_Systems/`)

### `comprehensive_analyzer.py`
**Purpose:** Statistical analysis and comprehensive reporting

#### Main Class:
```python
class ComprehensiveAnalyzer:
    """
    Complete analysis and reporting system for optimization results
    """
    
    def __init__(self, workspace_path):
        """
        Initialize the analyzer
        
        Args:
            workspace_path (str): Path to workspace directory
        """
    
    def load_all_trials(self):
        """
        Load all trial data from workspace
        
        Returns:
            pandas.DataFrame: Combined trial data
        """
    
    def create_comprehensive_comparison(self):
        """
        Create detailed comparison of all trials
        
        Returns:
            pandas.DataFrame: Comparison statistics
        """
    
    def generate_comprehensive_report(self):
        """
        Generate complete HTML analysis report
        
        Returns:
            str: Path to generated report
        """
```

### `advanced_visualizer.py`
**Purpose:** Professional visualization suite

#### Visualization Functions:
```python
def plot_evolution_progress(generation_data, save_path=None):
    """
    Create evolution progress plots
    
    Args:
        generation_data (list): Generation-by-generation fitness data
        save_path (str): Optional path to save plot
        
    Returns:
        matplotlib.figure.Figure: Generated plot
    """

def plot_3d_fitness_landscape(fitness_data, save_path=None):
    """
    Create 3D fitness landscape visualization
    
    Args:
        fitness_data (numpy.array): Multi-dimensional fitness data
        save_path (str): Optional path to save plot
        
    Returns:
        matplotlib.figure.Figure: 3D landscape plot
    """

def plot_protein_analysis(sequence_data, save_path=None):
    """
    Create comprehensive protein analysis plots
    
    Args:
        sequence_data (dict): Protein sequence analysis data
        save_path (str): Optional path to save plot
        
    Returns:
        matplotlib.figure.Figure: Analysis plots
    """
```

## Phase 4: Dashboard Development (`04_Dashboard_Development/`)

### `main_optimization_demo.py`
**Purpose:** Complete workflow automation and demonstration

#### Main Function:
```python
def run_complete_workflow():
    """
    Execute the complete optimization workflow
    
    Returns:
        dict: Complete workflow results
    """

def setup_optimization_environment():
    """
    Set up the optimization environment and parameters
    
    Returns:
        dict: Configuration parameters
    """

def execute_optimization_pipeline():
    """
    Run the optimization pipeline with all components
    
    Returns:
        dict: Pipeline execution results
    """
```

### `dashboard_server.py`
**Purpose:** Flask web server for interactive dashboard

#### Flask Routes:
```python
@app.route('/')
def dashboard_home():
    """
    Main dashboard page
    
    Returns:
        str: Rendered HTML template
    """

@app.route('/run_optimization', methods=['POST'])
def run_optimization():
    """
    Execute optimization with user parameters
    
    Returns:
        json: Optimization results
    """

@app.route('/view_results/<run_id>')
def view_results(run_id):
    """
    View results for a specific optimization run
    
    Args:
        run_id (str): Unique run identifier
        
    Returns:
        str: Rendered results template
    """
```

## Phase 5: 3D Molecular Viewer (`05_3D_Molecular_Viewer/`)

### `enhanced_3d_dashboard.py`
**Purpose:** Final dashboard with integrated 3D molecular visualization

#### Main Class:
```python
class Enhanced3DDashboard:
    """
    Advanced dashboard with 3D molecular viewer integration
    """
    
    def __init__(self, port=8000):
        """
        Initialize the enhanced dashboard
        
        Args:
            port (int): Port for web server
        """
    
    def start_server(self):
        """
        Start the dashboard web server
        """
    
    def generate_3d_structures(self):
        """
        Generate 3D structures for visualization
        
        Returns:
            dict: Generated structure data
        """
```

### `protein_3d_generator.py`
**Purpose:** 3D protein structure generation from sequences

#### Core Functions:
```python
def generate_3d_structure(sequence, structure_type='carbonic_anhydrase'):
    """
    Generate 3D protein structure from amino acid sequence
    
    Args:
        sequence (str): Protein sequence
        structure_type (str): Type of protein structure to generate
        
    Returns:
        dict: 3D structure data with coordinates
    """

def generate_backbone_coordinates(sequence):
    """
    Generate backbone coordinates for protein structure
    
    Args:
        sequence (str): Protein sequence
        
    Returns:
        numpy.array: 3D coordinates for backbone atoms
    """

def add_side_chains(backbone_coords, sequence):
    """
    Add side chain atoms to backbone structure
    
    Args:
        backbone_coords (numpy.array): Backbone coordinates
        sequence (str): Protein sequence
        
    Returns:
        dict: Complete structure with side chains
    """

def position_co2_molecules(structure, binding_sites):
    """
    Position CO2 molecules at predicted binding sites
    
    Args:
        structure (dict): Protein structure data
        binding_sites (list): Predicted binding site locations
        
    Returns:
        dict: Structure with positioned CO2 molecules
    """
```

### Three.js Molecular Viewer (`dashboard/templates/3d_viewer.html`)

#### JavaScript Classes:
```javascript
class Protein3DViewer {
    /**
     * Interactive 3D protein structure viewer using Three.js
     * 
     * @param {string} containerId - HTML container element ID
     */
    constructor(containerId) {
        this.containerId = containerId;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera();
        this.renderer = new THREE.WebGLRenderer();
    }
    
    /**
     * Load and display protein structure from JSON data
     * 
     * @param {string} jsonPath - Path to structure JSON file
     */
    loadStructure(jsonPath) {
        // Implementation for loading and rendering protein structure
    }
    
    /**
     * Set up mouse controls for interaction
     */
    setupMouseControls() {
        // Implementation for mouse rotation, zoom, and pan
    }
    
    /**
     * Highlight specific atoms or residues
     * 
     * @param {Array} atomIds - Array of atom IDs to highlight
     */
    highlightAtoms(atomIds) {
        // Implementation for highlighting selected atoms
    }
}
```

## Configuration & Data Structures

### Configuration Parameters
```python
DEFAULT_CONFIG = {
    'optimization': {
        'population_size': 50,
        'generations': 150,
        'sequence_length': 25,
        'mutation_rate': 0.15,
        'crossover_rate': 0.7,
        'elite_count': 5
    },
    'fitness_weights': {
        'co2_affinity': 0.3,
        'stability': 0.25,
        'expression': 0.25,
        'catalytic_efficiency': 0.2
    },
    'visualization': {
        'plot_dpi': 300,
        'figure_size': (12, 8),
        'color_scheme': 'viridis',
        'save_format': 'png'
    },
    'dashboard': {
        'port': 8000,
        'host': 'localhost',
        'debug': False
    }
}
```

### Data Structures

#### Optimization Results
```python
OptimizationResult = {
    'best_sequence': str,           # Best optimized sequence
    'best_fitness': list,           # Multi-objective fitness scores
    'generation_data': list,        # Generation-by-generation progress
    'population_diversity': list,   # Genetic diversity over time
    'convergence_generation': int,  # Generation where algorithm converged
    'runtime_seconds': float,       # Total optimization runtime
    'parameters': dict              # Optimization parameters used
}
```

#### 3D Structure Data
```python
ProteinStructure = {
    'atoms': [
        {
            'id': int,
            'element': str,         # Chemical element (C, N, O, etc.)
            'position': [x, y, z],  # 3D coordinates
            'residue_id': int,      # Residue number
            'atom_type': str        # Atom type (CA, CB, etc.)
        }
    ],
    'bonds': [
        {
            'atom1': int,           # First atom ID
            'atom2': int,           # Second atom ID
            'bond_type': str        # Bond type (single, double, etc.)
        }
    ],
    'metadata': {
        'sequence': str,            # Protein sequence
        'structure_type': str,      # Type of structure
        'generation_method': str    # How structure was generated
    }
}
```

## Error Handling & Logging

### Common Exceptions
```python
class OptimizationError(Exception):
    """Raised when optimization fails"""
    pass

class StructureGenerationError(Exception):
    """Raised when 3D structure generation fails"""
    pass

class VisualizationError(Exception):
    """Raised when visualization creation fails"""
    pass
```

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization.log'),
        logging.StreamHandler()
    ]
)
```

## Performance Considerations

### Memory Management
- **Large Populations:** Use generators for memory-efficient processing
- **Visualization Data:** Clear matplotlib figures after saving
- **3D Structures:** Optimize Three.js geometry for performance

### Computational Optimization
- **Parallel Processing:** Use multiprocessing for fitness evaluation
- **Caching:** Cache expensive calculations (structure generation)
- **Batch Operations:** Process multiple sequences simultaneously

### Web Performance
- **JSON Structure Size:** Compress large molecular structures
- **WebGL Optimization:** Use appropriate levels of detail
- **Caching:** Implement browser caching for static assets

This API reference provides a comprehensive overview of the system architecture and can be used as a guide for extending or modifying the codebase.