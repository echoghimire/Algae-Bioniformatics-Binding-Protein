"""
Interactive Dashboard Server for Algae Protein Optimization
Provides a user-friendly web interface to understand and visualize genetic algorithm results
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import webbrowser
from flask import Flask, render_template, jsonify, request, send_from_directory
import threading
import time

# Import our existing optimization modules
try:
    from enhanced_ga_protein_optimizer import EnhancedGeneticAlgorithm, ProteinAnalyzer
    # Import from the scientific visualizer instead of the mock version
    # from advanced_visualizer import ProteinOptimizationVisualizer
    from scientific_dashboard import ScientificDashboardServer
    from co2_binding_analyzer import CO2BindingAnalyzer
    from algae_protein_optimizer import AlgaeProteinOptimizer
except ImportError as e:
    print(f"Warning: Could not import optimization modules: {e}")

class DashboardManager:
    """Manages the web dashboard for protein optimization results"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent
        
        self.base_dir = Path(base_dir)
        self.dashboard_dir = self.base_dir / "dashboard"
        self.runs_dir = self.base_dir / "optimization_runs"
        self.static_dir = self.dashboard_dir / "static"
        self.templates_dir = self.dashboard_dir / "templates"
        
        # Ensure directories exist
        self.dashboard_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder=str(self.templates_dir),
                        static_folder=str(self.static_dir))
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/runs')
        def get_runs():
            """Get list of available optimization runs"""
            runs = self.get_optimization_runs()
            return jsonify({'runs': runs})
        
        @self.app.route('/api/run/<run_id>')
        def get_run_data(run_id):
            """Get detailed data for a specific run"""
            run_data = self.load_run_data(run_id)
            if run_data:
                return jsonify(run_data)
            else:
                return jsonify({'error': 'Run not found'}), 404
        
        @self.app.route('/api/start_optimization', methods=['POST'])
        def start_optimization():
            """Start a new optimization run"""
            try:
                params = request.json or {}
                run_id = self.run_optimization(params)
                return jsonify({'run_id': run_id, 'status': 'started'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/visualizations/<path:filename>')
        def serve_visualization(filename):
            """Serve visualization files"""
            # Find the file in any run directory
            for run_dir in self.runs_dir.iterdir():
                if run_dir.is_dir():
                    file_path = run_dir / filename
                    if file_path.exists():
                        return send_from_directory(str(run_dir), filename)
            return "File not found", 404
    
    def get_optimization_runs(self):
        """Get list of all optimization runs with metadata"""
        runs = []
        
        if not self.runs_dir.exists():
            return runs
        
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and run_dir.name.startswith('run_'):
                metadata_file = run_dir / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        runs.append(metadata)
                    except Exception as e:
                        print(f"Error loading metadata for {run_dir}: {e}")
        
        return runs
    
    def load_run_data(self, run_id):
        """Load complete data for a specific run"""
        run_dir = self.runs_dir / run_id
        metadata_file = run_dir / 'metadata.json'
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            # Add visualization file paths
            visualizations = []
            viz_files = [
                ('evolution_progress.png', 'Evolution Progress', 'Fitness improvement over generations'),
                ('interactive_evolution.html', 'Interactive Dashboard', 'Interactive visualization'),
                ('3d_fitness_landscape.png', '3D Fitness Landscape', 'Three-dimensional optimization space'),
                ('original_sequence_analysis.png', 'Original Sequence', 'Starting protein analysis'),
                ('optimized_sequence_analysis.png', 'Optimized Sequence', 'Final protein analysis'),
                ('co2_binding_analysis.png', 'CO₂ Binding Analysis', 'CO₂ processing capabilities'),
                ('sequence_comparison.png', 'Sequence Comparison', 'Before vs after comparison'),
                ('evolution_animation.gif', 'Evolution Animation', 'Animated optimization process')
            ]
            
            for filename, title, description in viz_files:
                file_path = run_dir / filename
                if file_path.exists():
                    visualizations.append({
                        'title': title,
                        'path': f'/visualizations/{filename}',
                        'description': description,
                        'filename': filename
                    })
            
            data['visualizations'] = visualizations
            return data
            
        except Exception as e:
            print(f"Error loading run data: {e}")
            return None
    
    def create_run_directory(self):
        """Create a new timestamped run directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"run_{timestamp}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        return run_id, run_dir
    
    def run_optimization(self, params=None):
        """Run a new optimization and save results"""
        if params is None:
            params = {}
        
        run_id, run_dir = self.create_run_directory()
        
        try:
            print(f"Starting optimization run: {run_id}")
            
            # Default parameters
            default_params = {
                'population_size': 100,
                'generations': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            }
            default_params.update(params)
            
            # Initialize optimizer
            fasta_file = self.base_dir / "Carbonic Anhydrase FASTA sequence.fasta"
            optimizer = AlgaeProteinOptimizer(str(fasta_file))
            
            # Run optimization
            results = optimizer.run_optimization(
                population_size=default_params['population_size'],
                generations=default_params['generations'],
                mutation_rate=default_params['mutation_rate'],
                crossover_rate=default_params['crossover_rate']
            )
            
            # Create visualizations in the run directory
            visualizer = ProteinOptimizationVisualizer()
            
            # Generate all visualizations
            viz_files = visualizer.create_comprehensive_dashboard(
                results['optimization_data'],
                results['original_sequence'],
                results['optimized_sequence'],
                str(run_dir)
            )
            
            # Calculate metrics
            original_fitness = results['optimization_data']['fitness_history'][0]
            final_fitness = results['optimization_data']['fitness_history'][-1]
            
            # Create metadata
            metadata = {
                'id': run_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': f"Multi-objective optimization ({default_params['generations']} generations)",
                'parameters': default_params,
                'metrics': {
                    'initial_co2_affinity': float(original_fitness[0]),
                    'final_co2_affinity': float(final_fitness[0]),
                    'improvement_percentage': float(((final_fitness[0] - original_fitness[0]) / original_fitness[0]) * 100),
                    'final_stability': float(final_fitness[1]),
                    'final_expression': float(final_fitness[2]),
                    'final_catalytic': float(final_fitness[3]),
                    'generations': default_params['generations'],
                    'population_size': default_params['population_size']
                },
                'sequences': {
                    'original': results['original_sequence'],
                    'optimized': results['optimized_sequence']
                },
                'files': {
                    'visualizations': viz_files
                }
            }
            
            # Save metadata
            with open(run_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save detailed results
            with open(run_dir / 'results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Optimization completed: {run_id}")
            return run_id
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            # Clean up failed run directory
            if run_dir.exists():
                shutil.rmtree(run_dir)
            raise e
    
    def start_server(self, host='localhost', port=5000, debug=False, open_browser=True):
        """Start the Flask development server"""
        print(f"Starting dashboard server at http://{host}:{port}")
        
        if open_browser:
            # Open browser after a short delay
            def open_browser_delayed():
                time.sleep(1.5)
                webbrowser.open(f'http://{host}:{port}')
            
            threading.Thread(target=open_browser_delayed, daemon=True).start()
        
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)
    
    def generate_sample_run(self):
        """Generate a sample run for demonstration"""
        print("Generating sample optimization run...")
        
        try:
            run_id = self.run_optimization({
                'population_size': 50,
                'generations': 30,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            })
            print(f"Sample run completed: {run_id}")
            return run_id
        except Exception as e:
            print(f"Error generating sample run: {e}")
            return None

def main():
    """Main function to start the dashboard"""
    dashboard = DashboardManager()
    
    print("🧬 Algae Protein Optimization Dashboard")
    print("=" * 50)
    
    # Check if we have any existing runs
    runs = dashboard.get_optimization_runs()
    
    if not runs:
        print("No optimization runs found. Generating sample run...")
        sample_run = dashboard.generate_sample_run()
        if sample_run:
            print(f"✅ Sample run generated: {sample_run}")
        else:
            print("❌ Failed to generate sample run")
    else:
        print(f"Found {len(runs)} existing optimization runs")
    
    print("\n🚀 Starting web dashboard...")
    print("📊 Features:")
    print("  - Interactive tutorial for beginners")
    print("  - Comprehensive visualizations")
    print("  - Run comparison and analysis")
    print("  - Scientific glossary")
    print("\n💡 The dashboard will open automatically in your browser")
    print("🔄 Use Ctrl+C to stop the server")
    
    try:
        dashboard.start_server(debug=False, open_browser=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard server stopped")

if __name__ == "__main__":
    main()