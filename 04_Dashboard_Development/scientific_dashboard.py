"""
Scientific Dashboard - Real Protein Optimization with Biochemical Analysis
Uses actual genetic algorithm and biochemical fitness evaluation
"""

import os
import json
import sys
from datetime import datetime
from pathlib import Path
import webbrowser
import time
import threading
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

# Add path to import our scientific analyzer
sys.path.append(str(Path(__file__).parent.parent / "03_Visualization_Systems"))
from scientific_co2_analyzer import ScientificCO2Analyzer
from scientific_visualizer import ScientificProteinVisualizer

class ScientificDashboardServer:
    """Real-time dashboard for scientific protein optimization"""
    
    def __init__(self, port=8080):
        self.port = port
        self.analyzer = ScientificCO2Analyzer()
        self.visualizer = ScientificProteinVisualizer()
        self.current_optimization = None
        self.optimization_results = {}
        
    def run_real_optimization(self, sequence_length=50, generations=100, population_size=50):
        """Run REAL genetic algorithm optimization"""
        
        print("🧬 Starting REAL Protein Optimization...")
        print(f"📊 Parameters: Length={sequence_length}, Generations={generations}, Population={population_size}")
        
        # Run the real genetic algorithm
        results = self.visualizer.run_real_genetic_algorithm(
            target_length=sequence_length,
            generations=generations,
            population_size=population_size
        )
        
        # Store results for dashboard
        optimization_id = f"opt_{int(time.time())}"
        self.optimization_results[optimization_id] = {
            'id': optimization_id,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'sequence_length': sequence_length,
                'generations': generations,
                'population_size': population_size
            },
            'results': results,
            'status': 'completed'
        }
        
        return optimization_id
    
    def analyze_custom_sequence(self, sequence):
        """Analyze a custom protein sequence"""
        
        print(f"🔬 Analyzing custom sequence: {sequence[:20]}...")
        
        # Real biochemical analysis
        fitness_data = self.analyzer.evaluate_protein_fitness(sequence)
        binding_analysis = self.analyzer.predict_co2_binding_affinity(sequence)
        
        return {
            'sequence': sequence,
            'fitness_data': fitness_data,
            'binding_analysis': binding_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_dashboard_html(self):
        """Generate real-time dashboard HTML"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧬 Scientific Protein Optimizer - Real Biochemical Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .status-banner {{
            background: linear-gradient(135deg, #28a745, #20c997);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 1.1em;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .panel {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }}
        
        .panel:hover {{
            transform: translateY(-5px);
        }}
        
        .panel h3 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.3em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .form-group {{
            margin-bottom: 20px;
        }}
        
        .form-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .form-group input, .form-group select, .form-group textarea {{
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e6ed;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }}
        
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {{
            outline: none;
            border-color: #3498db;
        }}
        
        .btn {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
        
        .btn:hover {{
            background: linear-gradient(135deg, #2980b9, #1f5f8b);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        
        .btn-success {{
            background: linear-gradient(135deg, #27ae60, #229954);
        }}
        
        .btn-success:hover {{
            background: linear-gradient(135deg, #229954, #1e8449);
        }}
        
        .btn-warning {{
            background: linear-gradient(135deg, #f39c12, #e67e22);
        }}
        
        .btn-warning:hover {{
            background: linear-gradient(135deg, #e67e22, #d35400);
        }}
        
        .results-panel {{
            grid-column: 1 / -1;
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 10px;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            transition: width 0.5s ease;
        }}
        
        .sequence-display {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            word-break: break-all;
            margin-bottom: 20px;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .plot-container {{
            min-height: 400px;
            margin-bottom: 20px;
        }}
        
        .status-running {{
            background: linear-gradient(135deg, #f39c12, #e67e22);
        }}
        
        .status-completed {{
            background: linear-gradient(135deg, #27ae60, #229954);
        }}
        
        .help-text {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-top: 5px;
            font-style: italic;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}
        
        .running {{
            animation: pulse 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Scientific Protein Optimizer</h1>
            <p>Real Biochemical Analysis for CO2-Binding Protein Design</p>
        </div>
        
        <div class="status-banner status-completed">
            ✅ REAL SCIENTIFIC MODE: Using actual biochemical analysis and genetic algorithms
        </div>
        
        <div class="dashboard-grid">
            <div class="panel">
                <h3>🚀 Run New Optimization</h3>
                <div class="form-group">
                    <label for="seq-length">🧬 Target Sequence Length</label>
                    <input type="number" id="seq-length" value="50" min="20" max="200">
                    <div class="help-text">Length of the protein to optimize (20-200 amino acids)</div>
                </div>
                <div class="form-group">
                    <label for="generations">📈 Generations</label>
                    <input type="number" id="generations" value="50" min="10" max="200">
                    <div class="help-text">Number of evolutionary generations (more = better results, slower)</div>
                </div>
                <div class="form-group">
                    <label for="population">👥 Population Size</label>
                    <input type="number" id="population" value="30" min="10" max="100">
                    <div class="help-text">Number of proteins in each generation</div>
                </div>
                <button class="btn btn-success" onclick="startOptimization()">
                    🚀 Start Real Optimization
                </button>
                <div id="optimization-status" style="margin-top: 10px;"></div>
            </div>
            
            <div class="panel">
                <h3>🔬 Analyze Custom Sequence</h3>
                <div class="form-group">
                    <label for="custom-sequence">🧬 Protein Sequence</label>
                    <textarea id="custom-sequence" rows="4" 
                              placeholder="Enter amino acid sequence (e.g., MKAAVLTLAVLFLTGSQARHFWGYGSHTNDQIK...)">MKAAVLTLAVLFLTGSQARHFWGYGSHTNDQIKQYK</textarea>
                    <div class="help-text">Enter a protein sequence using standard amino acid codes</div>
                </div>
                <button class="btn btn-warning" onclick="analyzeSequence()">
                    🔬 Analyze Sequence
                </button>
                <div id="analysis-status" style="margin-top: 10px;"></div>
            </div>
        </div>
        
        <div class="results-panel">
            <h3>📊 Real-Time Results</h3>
            <div id="results-content">
                <p style="text-align: center; color: #7f8c8d; font-style: italic; padding: 40px;">
                    🔬 Run an optimization or analyze a sequence to see real biochemical results
                </p>
            </div>
        </div>
    </div>

    <script>
        let currentOptimization = null;
        let isOptimizing = false;
        
        async function startOptimization() {{
            if (isOptimizing) {{
                alert('⚠️ Optimization already running! Please wait...');
                return;
            }}
            
            const seqLength = document.getElementById('seq-length').value;
            const generations = document.getElementById('generations').value;
            const population = document.getElementById('population').value;
            
            if (!seqLength || !generations || !population) {{
                alert('❌ Please fill in all parameters');
                return;
            }}
            
            isOptimizing = true;
            const statusDiv = document.getElementById('optimization-status');
            statusDiv.innerHTML = '<div class="running" style="color: #f39c12; font-weight: bold;">🔄 Running real genetic algorithm...</div>';
            
            try {{
                const response = await fetch('/optimize', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        sequence_length: parseInt(seqLength),
                        generations: parseInt(generations),
                        population_size: parseInt(population)
                    }})
                }});
                
                const result = await response.json();
                
                if (result.success) {{
                    statusDiv.innerHTML = '<div style="color: #27ae60; font-weight: bold;">✅ Optimization completed!</div>';
                    displayOptimizationResults(result.data);
                }} else {{
                    statusDiv.innerHTML = `<div style="color: #e74c3c; font-weight: bold;">❌ Error: ${{result.error}}</div>`;
                }}
            }} catch (error) {{
                statusDiv.innerHTML = `<div style="color: #e74c3c; font-weight: bold;">❌ Network error: ${{error.message}}</div>`;
            }} finally {{
                isOptimizing = false;
            }}
        }}
        
        async function analyzeSequence() {{
            const sequence = document.getElementById('custom-sequence').value.trim().toUpperCase();
            
            if (!sequence) {{
                alert('❌ Please enter a protein sequence');
                return;
            }}
            
            // Validate amino acid sequence
            const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/;
            if (!validAminoAcids.test(sequence)) {{
                alert('❌ Invalid amino acid sequence. Use only standard amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)');
                return;
            }}
            
            const statusDiv = document.getElementById('analysis-status');
            statusDiv.innerHTML = '<div class="running" style="color: #f39c12; font-weight: bold;">🔬 Analyzing biochemical properties...</div>';
            
            try {{
                const response = await fetch('/analyze', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{sequence: sequence}})
                }});
                
                const result = await response.json();
                
                if (result.success) {{
                    statusDiv.innerHTML = '<div style="color: #27ae60; font-weight: bold;">✅ Analysis completed!</div>';
                    displayAnalysisResults(result.data);
                }} else {{
                    statusDiv.innerHTML = `<div style="color: #e74c3c; font-weight: bold;">❌ Error: ${{result.error}}</div>`;
                }}
            }} catch (error) {{
                statusDiv.innerHTML = `<div style="color: #e74c3c; font-weight: bold;">❌ Network error: ${{error.message}}</div>`;
            }}
        }}
        
        function displayOptimizationResults(data) {{
            const resultsDiv = document.getElementById('results-content');
            const results = data.results;
            const finalAnalysis = results.final_analysis;
            
            resultsDiv.innerHTML = `
                <h4>🏆 Optimization Results - REAL Biochemical Analysis</h4>
                
                <div class="sequence-display">
                    <strong>🧬 Optimized Sequence (${{results.best_sequence.length}} amino acids):</strong><br>
                    ${{results.best_sequence}}
                </div>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">${{results.best_fitness.toFixed(4)}}</div>
                        <div class="metric-label">Overall Fitness</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${{finalAnalysis.binding_energy.toFixed(2)}}</div>
                        <div class="metric-label">Binding Energy (kcal/mol)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${{(finalAnalysis.co2_binding_affinity * 100).toFixed(1)}}%</div>
                        <div class="metric-label">CO2 Binding Affinity</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${{(finalAnalysis.structural_stability * 100).toFixed(1)}}%</div>
                        <div class="metric-label">Structural Stability</div>
                    </div>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <h5>⚛️ Real Biochemical Properties:</h5>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                        <div>🧲 Zinc Binding Score: <strong>${{finalAnalysis.detailed_co2_analysis.zinc_binding_score.toFixed(3)}}</strong></div>
                        <div>⚗️ Catalytic Score: <strong>${{finalAnalysis.detailed_co2_analysis.catalytic_score.toFixed(3)}}</strong></div>
                        <div>🔍 Motif Score: <strong>${{finalAnalysis.detailed_co2_analysis.motif_score.toFixed(3)}}</strong></div>
                        <div>📐 Spatial Score: <strong>${{finalAnalysis.detailed_co2_analysis.spatial_score.toFixed(3)}}</strong></div>
                    </div>
                </div>
                
                <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; border-left: 4px solid #27ae60;">
                    <strong>🎯 Classification:</strong> ${{finalAnalysis.detailed_co2_analysis.binding_category}}<br>
                    <strong>📊 Analysis Status:</strong> Based on real carbonic anhydrase biochemistry<br>
                    <strong>⏱️ Completed:</strong> ${{new Date().toLocaleString()}}
                </div>
                
                <div id="evolution-plot" class="plot-container"></div>
            `;
            
            // Create evolution plot
            createEvolutionPlot(results.generations_data);
        }}
        
        function displayAnalysisResults(data) {{
            const resultsDiv = document.getElementById('results-content');
            const fitness = data.fitness_data;
            const binding = data.binding_analysis;
            
            resultsDiv.innerHTML = `
                <h4>🔬 Sequence Analysis - REAL Biochemical Properties</h4>
                
                <div class="sequence-display">
                    <strong>🧬 Analyzed Sequence (${{data.sequence.length}} amino acids):</strong><br>
                    ${{data.sequence}}
                </div>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">${{fitness.overall_fitness.toFixed(4)}}</div>
                        <div class="metric-label">Overall Fitness</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${{fitness.binding_energy.toFixed(2)}}</div>
                        <div class="metric-label">Binding Energy (kcal/mol)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${{(binding.overall_affinity * 100).toFixed(1)}}%</div>
                        <div class="metric-label">CO2 Binding Affinity</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${{(fitness.expression_likelihood * 100).toFixed(1)}}%</div>
                        <div class="metric-label">Expression Likelihood</div>
                    </div>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <h5>🧪 Detailed Biochemical Analysis:</h5>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;">
                        <div>
                            <strong>⚛️ CO2 Binding:</strong>
                            <div style="margin-left: 10px; font-size: 0.9em;">
                                • Zinc Binding: ${{binding.zinc_binding_score.toFixed(3)}}<br>
                                • Catalytic Score: ${{binding.catalytic_score.toFixed(3)}}<br>
                                • Active Site Motifs: ${{binding.motif_score.toFixed(3)}}<br>
                                • Spatial Arrangement: ${{binding.spatial_score.toFixed(3)}}
                            </div>
                        </div>
                        <div>
                            <strong>🏗️ Protein Properties:</strong>
                            <div style="margin-left: 10px; font-size: 0.9em;">
                                • Structural Stability: ${{fitness.structural_stability.toFixed(3)}}<br>
                                • Catalytic Efficiency: ${{fitness.catalytic_efficiency.toFixed(3)}}<br>
                                • Expression Likelihood: ${{fitness.expression_likelihood.toFixed(3)}}<br>
                                • Overall Quality: ${{fitness.overall_fitness.toFixed(3)}}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                    <strong>🎯 Classification:</strong> ${{binding.binding_category}}<br>
                    <strong>🔬 Analysis Method:</strong> Real biochemical analysis using carbonic anhydrase principles<br>
                    <strong>📊 Data Source:</strong> Experimental binding energies and amino acid properties<br>
                    <strong>⏱️ Analyzed:</strong> ${{new Date().toLocaleString()}}
                </div>
            `;
        }}
        
        function createEvolutionPlot(generationsData) {{
            const generations = generationsData.map(d => d.generation);
            const bestFitness = generationsData.map(d => d.best_fitness);
            const avgFitness = generationsData.map(d => d.average_fitness);
            
            const trace1 = {{
                x: generations,
                y: bestFitness,
                mode: 'lines+markers',
                name: 'Best Fitness',
                line: {{color: '#e74c3c', width: 3}},
                marker: {{size: 6}}
            }};
            
            const trace2 = {{
                x: generations,
                y: avgFitness,
                mode: 'lines+markers',
                name: 'Average Fitness',
                line: {{color: '#3498db', width: 2}},
                marker: {{size: 4}}
            }};
            
            const layout = {{
                title: '📈 Real Evolution Progress - Biochemical Fitness',
                xaxis: {{title: 'Generation'}},
                yaxis: {{title: 'Fitness Score'}},
                hovermode: 'x unified'
            }};
            
            Plotly.newPlot('evolution-plot', [trace1, trace2], layout);
        }}
        
        // Auto-refresh status every 5 seconds when optimizing
        setInterval(() => {{
            if (isOptimizing) {{
                // Could add progress polling here if needed
            }}
        }}, 5000);
    </script>
</body>
</html>
        """
        
        return html_content
    
    def create_request_handler(self):
        """Create HTTP request handler for the dashboard"""
        
        dashboard_server = self
        
        class DashboardHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/dashboard':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(dashboard_server.generate_dashboard_html().encode())
                else:
                    self.send_error(404)
            
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    
                    if self.path == '/optimize':
                        # Run real optimization
                        optimization_id = dashboard_server.run_real_optimization(
                            sequence_length=data.get('sequence_length', 50),
                            generations=data.get('generations', 50),
                            population_size=data.get('population_size', 30)
                        )
                        
                        response = {
                            'success': True,
                            'data': dashboard_server.optimization_results[optimization_id]
                        }
                        
                    elif self.path == '/analyze':
                        # Analyze custom sequence
                        analysis_result = dashboard_server.analyze_custom_sequence(
                            data.get('sequence', '')
                        )
                        
                        response = {
                            'success': True,
                            'data': analysis_result
                        }
                    
                    else:
                        response = {'success': False, 'error': 'Unknown endpoint'}
                    
                except Exception as e:
                    response = {'success': False, 'error': str(e)}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response, default=str).encode())
        
        return DashboardHandler
    
    def start_server(self):
        """Start the dashboard server"""
        
        print(f"🌐 Starting Scientific Dashboard Server on port {self.port}...")
        
        handler = self.create_request_handler()
        
        with socketserver.TCPServer(("", self.port), handler) as httpd:
            print(f"✅ Dashboard running at: http://localhost:{self.port}")
            print("🧬 Real biochemical analysis ready!")
            print("📊 Features:")
            print("   • Real genetic algorithm optimization")
            print("   • Actual CO2 binding energy calculations")
            print("   • Biochemical fitness evaluation")
            print("   • Custom sequence analysis")
            print("\\n🚀 Open your browser to start optimizing proteins!")
            
            try:
                # Try to open browser automatically
                webbrowser.open(f'http://localhost:{self.port}')
            except:
                pass
            
            httpd.serve_forever()


def main():
    """Run the scientific dashboard"""
    
    print("🧬 Scientific Protein Optimization Dashboard")
    print("=" * 50)
    print("🔬 REAL MODE: Using actual biochemical analysis")
    print("⚛️ Features: Genetic algorithms, CO2 binding prediction, thermodynamic calculations")
    print("📊 No mock data - all results based on real science!")
    print()
    
    # Create and start dashboard server
    dashboard = ScientificDashboardServer(port=8081)
    
    try:
        dashboard.start_server()
    except KeyboardInterrupt:
        print("\\n\\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("\\n💡 Tips:")
        print("   • Make sure port 8081 is available")
        print("   • Check that all dependencies are installed")
        print("   • Try running as administrator if needed")


if __name__ == "__main__":
    main()