"""
Scientific 3D Molecular Viewer
Real-time visualization of scientifically predicted protein structures
"""

import http.server
import socketserver
import json
import sys
from pathlib import Path
import webbrowser
import threading
import time

# Import our scientific structure predictor
from scientific_structure_predictor import ScientificProteinStructurePredictor

class Scientific3DViewer:
    """Real-time 3D molecular viewer with scientific structure prediction"""
    
    def __init__(self, port=8082):
        self.port = port
        self.predictor = ScientificProteinStructurePredictor()
        self.cached_structures = {}
    
    def generate_viewer_html(self):
        """Generate HTML for 3D molecular viewer"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧬 Scientific 3D Molecular Viewer</title>
    <script src="https://unpkg.com/3dmol@latest/build/3Dmol-min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2c3e50, #3498db);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{
            max-width: 1600px;
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
        
        .scientific-banner {{
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .viewer-grid {{
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
            height: 80vh;
        }}
        
        .viewer-panel {{
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            position: relative;
        }}
        
        .controls-panel {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            overflow-y: auto;
        }}
        
        #viewer3d {{
            width: 100%;
            height: 100%;
            min-height: 600px;
        }}
        
        .control-section {{
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .control-section:last-child {{
            border-bottom: none;
        }}
        
        .control-section h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .form-group {{
            margin-bottom: 15px;
        }}
        
        .form-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #2c3e50;
            font-size: 0.9em;
        }}
        
        .form-group input, .form-group textarea, .form-group select {{
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e6ed;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }}
        
        .form-group input:focus, .form-group textarea:focus, .form-group select:focus {{
            outline: none;
            border-color: #3498db;
        }}
        
        .btn {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            margin-bottom: 8px;
        }}
        
        .btn:hover {{
            background: linear-gradient(135deg, #2980b9, #1f5f8b);
            transform: translateY(-1px);
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
        
        .btn-small {{
            padding: 6px 12px;
            font-size: 12px;
            width: auto;
            margin-right: 5px;
        }}
        
        .analysis-result {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 0.9em;
        }}
        
        .metric-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 5px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .metric-row:last-child {{
            border-bottom: none;
            margin-bottom: 0;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-weight: 500;
        }}
        
        .metric-value {{
            color: #2c3e50;
            font-weight: 600;
        }}
        
        .binding-sites-list {{
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .binding-site {{
            background: #e8f5e8;
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 5px;
            font-size: 0.85em;
            border-left: 3px solid #27ae60;
        }}
        
        .status-indicator {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        
        .status-working {{
            background: rgba(243, 156, 18, 0.9);
            animation: pulse 2s infinite;
        }}
        
        .status-ready {{
            background: rgba(39, 174, 96, 0.9);
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}
        
        .help-text {{
            font-size: 0.8em;
            color: #7f8c8d;
            margin-top: 3px;
            font-style: italic;
        }}
        
        .visualization-controls {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Scientific 3D Molecular Viewer</h1>
            <p>Real-time protein structure prediction and CO2 binding analysis</p>
        </div>
        
        <div class="scientific-banner">
            ✅ SCIENTIFIC MODE: Using Chou-Fasman prediction, energy minimization, and real biochemical analysis
        </div>
        
        <div class="viewer-grid">
            <div class="viewer-panel">
                <div id="viewer3d"></div>
                <div id="status" class="status-indicator status-ready">Ready</div>
            </div>
            
            <div class="controls-panel">
                <div class="control-section">
                    <h3>🧬 Protein Input</h3>
                    <div class="form-group">
                        <label for="protein-sequence">Amino Acid Sequence</label>
                        <textarea id="protein-sequence" rows="4" 
                                placeholder="Enter protein sequence (e.g., MKAAVLTLAVLFLTGSQARHFWGYGSHTNDQIK...)">MKAAVLTLAVLFLTGSQARHFWGYGSHTNDQIKQYKHHDHETHWGQNDFTGQIYDLYNIQK</textarea>
                        <div class="help-text">Standard amino acid codes (A-Y, 20-200 residues recommended)</div>
                    </div>
                    <button class="btn btn-success" onclick="predictStructure()">
                        🔮 Predict Structure
                    </button>
                </div>
                
                <div class="control-section">
                    <h3>🎨 Visualization</h3>
                    <div class="visualization-controls">
                        <button class="btn btn-small" onclick="showCartoon()">Cartoon</button>
                        <button class="btn btn-small" onclick="showSpheres()">Spheres</button>
                        <button class="btn btn-small" onclick="showSticks()">Sticks</button>
                        <button class="btn btn-small" onclick="showSurface()">Surface</button>
                        <button class="btn btn-small" onclick="showBindingSites()">Sites</button>
                        <button class="btn btn-small" onclick="showCO2()">CO2</button>
                    </div>
                    <div class="form-group" style="margin-top: 10px;">
                        <label for="color-scheme">Color Scheme</label>
                        <select id="color-scheme" onchange="updateColors()">
                            <option value="amino">Amino Acid Type</option>
                            <option value="hydrophobicity">Hydrophobicity</option>
                            <option value="charge">Charge</option>
                            <option value="secondary">Secondary Structure</option>
                            <option value="binding">CO2 Binding Affinity</option>
                        </select>
                    </div>
                </div>
                
                <div class="control-section">
                    <h3>📊 Structure Analysis</h3>
                    <div id="analysis-results">
                        <p style="color: #7f8c8d; font-style: italic; text-align: center;">
                            Predict a structure to see analysis
                        </p>
                    </div>
                </div>
                
                <div class="control-section">
                    <h3>🎯 Binding Sites</h3>
                    <div id="binding-sites" class="binding-sites-list">
                        <p style="color: #7f8c8d; font-style: italic; text-align: center;">
                            No binding sites identified
                        </p>
                    </div>
                </div>
                
                <div class="control-section">
                    <h3>💾 Export</h3>
                    <button class="btn" onclick="downloadPDB()">📁 Download PDB</button>
                    <button class="btn" onclick="saveImage()">📸 Save Image</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let viewer = null;
        let currentStructure = null;
        let isPredicting = false;
        
        // Initialize 3D viewer
        function initViewer() {{
            const element = document.getElementById('viewer3d');
            const config = {{
                backgroundColor: 'white',
                antialias: true,
                alpha: true
            }};
            viewer = $3Dmol.createViewer(element, config);
            
            // Add some initial content
            viewer.addLabel("🧬 Scientific Protein Viewer\\n\\nPredict a structure to begin", 
                          {{position: {{x: 0, y: 0, z: 0}}, 
                            backgroundColor: 'rgba(52, 152, 219, 0.8)',
                            fontColor: 'white',
                            fontSize: 16}});
            viewer.render();
        }}
        
        async function predictStructure() {{
            if (isPredicting) {{
                alert('⚠️ Structure prediction in progress...');
                return;
            }}
            
            const sequence = document.getElementById('protein-sequence').value.trim().toUpperCase();
            
            if (!sequence) {{
                alert('❌ Please enter a protein sequence');
                return;
            }}
            
            // Validate sequence
            const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/;
            if (!validAminoAcids.test(sequence)) {{
                alert('❌ Invalid sequence. Use only standard amino acid codes.');
                return;
            }}
            
            if (sequence.length < 10 || sequence.length > 500) {{
                alert('❌ Sequence length should be between 10-500 amino acids');
                return;
            }}
            
            isPredicting = true;
            updateStatus('🔄 Predicting structure...', 'working');
            
            try {{
                const response = await fetch('/predict', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{sequence: sequence}})
                }});
                
                const result = await response.json();
                
                if (result.success) {{
                    currentStructure = result.data;
                    displayStructure(currentStructure);
                    displayAnalysis(currentStructure);
                    displayBindingSites(currentStructure.binding_sites);
                    updateStatus('✅ Structure ready', 'ready');
                }} else {{
                    alert(`❌ Prediction failed: ${{result.error}}`);
                    updateStatus('❌ Prediction failed', 'ready');
                }}
            }} catch (error) {{
                alert(`❌ Network error: ${{error.message}}`);
                updateStatus('❌ Network error', 'ready');
            }} finally {{
                isPredicting = false;
            }}
        }}
        
        function displayStructure(structure) {{
            viewer.clear();
            
            // Add protein structure from PDB
            viewer.addModel(structure.pdb_content, 'pdb');
            
            // Set initial style
            showCartoon();
            
            // Center and zoom
            viewer.zoomTo();
            viewer.render();
        }}
        
        function displayAnalysis(structure) {{
            const analysis = structure.binding_analysis;
            const quality = structure.structure_quality;
            
            const analysisDiv = document.getElementById('analysis-results');
            analysisDiv.innerHTML = `
                <div class="metric-row">
                    <span class="metric-label">🧬 Length</span>
                    <span class="metric-value">${{structure.length}} AA</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">⚛️ Binding Energy</span>
                    <span class="metric-value">${{quality.binding_energy.toFixed(2)}} kcal/mol</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">🎯 Overall Quality</span>
                    <span class="metric-value">${{(quality.overall_quality * 100).toFixed(1)}}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">🧲 Zinc Binding</span>
                    <span class="metric-value">${{analysis.zinc_binding_score.toFixed(3)}}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">⚗️ Catalytic Score</span>
                    <span class="metric-value">${{analysis.catalytic_score.toFixed(3)}}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">🔍 Active Sites</span>
                    <span class="metric-value">${{quality.binding_sites_count}}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">💨 CO2 Molecules</span>
                    <span class="metric-value">${{quality.co2_molecules_bound}}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">🎯 Classification</span>
                    <span class="metric-value">${{analysis.binding_category}}</span>
                </div>
            `;
        }}
        
        function displayBindingSites(bindingSites) {{
            const bindingDiv = document.getElementById('binding-sites');
            
            if (bindingSites.length === 0) {{
                bindingDiv.innerHTML = '<p style="color: #7f8c8d; font-style: italic; text-align: center;">No binding sites found</p>';
                return;
            }}
            
            bindingDiv.innerHTML = bindingSites.map((site, index) => `
                <div class="binding-site">
                    <strong>${{site.site_type}}</strong><br>
                    Sequence: ${{site.sequence}}<br>
                    Strength: ${{site.binding_strength.toFixed(3)}}<br>
                    Position: ${{site.residue_range[0]}}-${{site.residue_range[1]}}
                </div>
            `).join('');
        }}
        
        function showCartoon() {{
            if (!viewer || !currentStructure) return;
            viewer.removeAllLabels();
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            viewer.render();
        }}
        
        function showSpheres() {{
            if (!viewer || !currentStructure) return;
            viewer.removeAllLabels();
            viewer.setStyle({{}}, {{sphere: {{colorscheme: 'amino', radius: 1.0}}}});
            viewer.render();
        }}
        
        function showSticks() {{
            if (!viewer || !currentStructure) return;
            viewer.removeAllLabels();
            viewer.setStyle({{}}, {{stick: {{colorscheme: 'amino', radius: 0.3}}}});
            viewer.render();
        }}
        
        function showSurface() {{
            if (!viewer || !currentStructure) return;
            viewer.removeAllLabels();
            viewer.addSurface($3Dmol.SurfaceType.VDW, {{opacity: 0.7, colorscheme: 'hydrophobicity'}});
            viewer.render();
        }}
        
        function showBindingSites() {{
            if (!viewer || !currentStructure) return;
            viewer.removeAllLabels();
            
            // Highlight binding sites
            viewer.setStyle({{}}, {{cartoon: {{color: 'lightgray'}}}});
            
            currentStructure.binding_sites.forEach((site, index) => {{
                const start = site.residue_range[0];
                const end = site.residue_range[1];
                
                viewer.setStyle({{resi: `${{start}}-${{end}}`}}, 
                              {{cartoon: {{color: 'red'}}, stick: {{colorscheme: 'amino'}}}});
                
                // Add label
                const pos = site.position;
                viewer.addLabel(`${{site.site_type}}\\n${{site.sequence}}`, 
                              {{position: {{x: pos[0], y: pos[1], z: pos[2]}},
                                backgroundColor: 'rgba(231, 76, 60, 0.8)',
                                fontColor: 'white',
                                fontSize: 10}});
            }});
            
            viewer.render();
        }}
        
        function showCO2() {{
            if (!viewer || !currentStructure) return;
            viewer.removeAllLabels();
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            
            // Highlight CO2 molecules
            viewer.setStyle({{resn: 'CO2'}}, {{sphere: {{color: 'red', radius: 1.5}}}});
            
            // Add CO2 labels
            currentStructure.co2_molecules.forEach((co2, index) => {{
                const pos = co2.carbon_position;
                viewer.addLabel(`CO2\\n${{co2.binding_energy.toFixed(1)}} kcal/mol`, 
                              {{position: {{x: pos[0], y: pos[1], z: pos[2]}},
                                backgroundColor: 'rgba(46, 204, 113, 0.8)',
                                fontColor: 'white',
                                fontSize: 10}});
            }});
            
            viewer.render();
        }}
        
        function updateColors() {{
            const colorScheme = document.getElementById('color-scheme').value;
            
            if (!viewer || !currentStructure) return;
            
            viewer.removeAllLabels();
            
            switch(colorScheme) {{
                case 'amino':
                    viewer.setStyle({{}}, {{cartoon: {{colorscheme: 'amino'}}}});
                    break;
                case 'hydrophobicity':
                    viewer.setStyle({{}}, {{cartoon: {{colorscheme: 'hydrophobicity'}}}});
                    break;
                case 'charge':
                    viewer.setStyle({{}}, {{cartoon: {{colorscheme: 'charge'}}}});
                    break;
                case 'secondary':
                    viewer.setStyle({{}}, {{cartoon: {{colorscheme: 'secondary'}}}});
                    break;
                case 'binding':
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                    // Color by CO2 binding affinity (custom)
                    break;
            }}
            
            viewer.render();
        }}
        
        function downloadPDB() {{
            if (!currentStructure) {{
                alert('❌ No structure to download');
                return;
            }}
            
            const blob = new Blob([currentStructure.pdb_content], {{type: 'text/plain'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `protein_structure_${{currentStructure.length}}aa.pdb`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
        
        function saveImage() {{
            if (!viewer) {{
                alert('❌ No structure to save');
                return;
            }}
            
            viewer.pngURI(function(uri) {{
                const a = document.createElement('a');
                a.href = uri;
                a.download = 'protein_structure.png';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }});
        }}
        
        function updateStatus(message, type) {{
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = `status-indicator status-${{type}}`;
        }}
        
        // Initialize viewer when page loads
        window.addEventListener('load', function() {{
            initViewer();
        }});
    </script>
</body>
</html>
        """
        
        return html_content
    
    def create_request_handler(self):
        """Create HTTP request handler for the 3D viewer"""
        
        viewer_server = self
        
        class ViewerHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/viewer':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(viewer_server.generate_viewer_html().encode())
                else:
                    self.send_error(404)
            
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    
                    if self.path == '/predict':
                        sequence = data.get('sequence', '')
                        
                        # Check cache first
                        if sequence in viewer_server.cached_structures:
                            response = {
                                'success': True,
                                'data': viewer_server.cached_structures[sequence]
                            }
                        else:
                            # Predict new structure
                            structure_prediction = viewer_server.predictor.predict_full_structure(sequence)
                            viewer_server.cached_structures[sequence] = structure_prediction
                            
                            response = {
                                'success': True,
                                'data': structure_prediction
                            }
                    else:
                        response = {'success': False, 'error': 'Unknown endpoint'}
                    
                except Exception as e:
                    response = {'success': False, 'error': str(e)}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response, default=str).encode())
        
        return ViewerHandler
    
    def start_server(self):
        """Start the 3D viewer server"""
        
        print(f"🌐 Starting Scientific 3D Molecular Viewer on port {self.port}...")
        
        handler = self.create_request_handler()
        
        with socketserver.TCPServer(("", self.port), handler) as httpd:
            print(f"✅ 3D Viewer running at: http://localhost:{self.port}")
            print("🧬 Scientific structure prediction ready!")
            print("📊 Features:")
            print("   • Chou-Fasman secondary structure prediction")
            print("   • Energy minimization")
            print("   • Real CO2 binding site identification")
            print("   • Interactive 3D visualization")
            print("   • PDB export")
            print("\\n🚀 Open your browser to start viewing structures!")
            
            try:
                webbrowser.open(f'http://localhost:{self.port}')
            except:
                pass
            
            httpd.serve_forever()


def main():
    """Run the scientific 3D viewer"""
    
    print("🧬 Scientific 3D Molecular Viewer")
    print("=" * 50)
    print("🔬 REAL MODE: Scientific structure prediction")
    print("⚛️ Features:")
    print("   • Chou-Fasman secondary structure prediction")
    print("   • Physics-based coordinate generation") 
    print("   • Energy minimization")
    print("   • Real CO2 binding analysis")
    print("   • Interactive 3D visualization")
    print("   • PDB format export")
    print("📊 No mock coordinates - all based on real structural biology!")
    print()
    
    viewer = Scientific3DViewer(port=8082)
    
    try:
        viewer.start_server()
    except KeyboardInterrupt:
        print("\\n\\n🛑 3D Viewer stopped by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        print("\\n💡 Tips:")
        print("   • Make sure port 8082 is available")
        print("   • Check that all dependencies are installed")


if __name__ == "__main__":
    main()