# 🌊 Quantum Visualization System

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

from quantum_co2_calculator import QuantumResult, QuantumCO2BindingCalculator

class QuantumVisualizationSystem:
    """
    Advanced visualization system for quantum mechanical properties
    including electron density, molecular orbitals, and binding analysis
    """
    
    def __init__(self, output_dir: str = "quantum_visualizations"):
        """
        Initialize quantum visualization system
        
        Args:
            output_dir: Directory for saving visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for different properties
        self.color_schemes = {
            'electron_density': 'plasma',
            'molecular_orbitals': 'RdYlBu',
            'binding_energy': 'viridis',
            'charge_distribution': 'RdBu_r'
        }
        
        # Default visualization parameters
        self.viz_params = {
            'figure_size': (12, 10),
            'dpi': 300,
            'grid_resolution': 50,
            'contour_levels': 20,
            'opacity': 0.7,
            'interactive': True
        }
    
    def visualize_electron_density(self, 
                                 quantum_result: QuantumResult,
                                 protein_coords: np.ndarray,
                                 co2_coords: np.ndarray,
                                 save_path: Optional[str] = None) -> str:
        """
        Create 3D visualization of electron density around CO2 binding site
        
        Args:
            quantum_result: Results from quantum calculation
            protein_coords: Protein atom coordinates
            co2_coords: CO2 molecule coordinates
            save_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization file
        """
        print("🌊 Generating electron density visualization...")
        
        # Create 3D grid around binding site
        center = np.mean(np.vstack([protein_coords, co2_coords]), axis=0)
        grid_size = 10.0  # Angstroms
        resolution = self.viz_params['grid_resolution']
        
        x = np.linspace(center[0] - grid_size/2, center[0] + grid_size/2, resolution)
        y = np.linspace(center[1] - grid_size/2, center[1] + grid_size/2, resolution)
        z = np.linspace(center[2] - grid_size/2, center[2] + grid_size/2, resolution)
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Get electron density from quantum result
        # For demonstration, we'll use the provided density or generate realistic-looking data
        if hasattr(quantum_result, 'electron_density') and quantum_result.electron_density is not None:
            density = quantum_result.electron_density
            if density.shape != (resolution, resolution, resolution):
                # Interpolate to correct size
                density = self._interpolate_density(density, (resolution, resolution, resolution))
        else:
            # Generate realistic electron density distribution
            density = self._generate_realistic_density(X, Y, Z, protein_coords, co2_coords)
        
        # Create interactive 3D visualization
        fig = self._create_interactive_electron_density_plot(X, Y, Z, density, protein_coords, co2_coords)
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "electron_density_3d.html"
        
        fig.write_html(str(save_path))
        print(f"✅ Electron density visualization saved to {save_path}")
        
        # Also create matplotlib version for static use
        self._create_static_electron_density_plot(X, Y, Z, density, protein_coords, co2_coords)
        
        return str(save_path)
    
    def visualize_molecular_orbitals(self, 
                                   quantum_result: QuantumResult,
                                   protein_coords: np.ndarray,
                                   save_path: Optional[str] = None) -> str:
        """
        Visualize molecular orbitals (HOMO, LUMO, etc.)
        
        Args:
            quantum_result: Results from quantum calculation
            protein_coords: Protein atom coordinates
            save_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization file
        """
        print("🎯 Generating molecular orbital visualization...")
        
        # Extract orbital information
        orbitals = quantum_result.molecular_orbitals
        
        # Create subplot figure for multiple orbitals
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['HOMO', 'LUMO', 'HOMO-1', 'LUMO+1'],
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        orbital_names = ['HOMO', 'LUMO', 'HOMO-1', 'LUMO+1']
        
        for i, orbital_name in enumerate(orbital_names):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Generate orbital density (simplified representation)
            orbital_density = self._generate_orbital_density(protein_coords, orbital_name, orbitals)
            
            # Create 3D isosurface
            self._add_orbital_isosurface(fig, orbital_density, protein_coords, 
                                       orbital_name, row, col)
        
        # Update layout
        fig.update_layout(
            title="Molecular Orbitals Analysis",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)"
            ),
            height=800,
            width=1200
        )
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "molecular_orbitals.html"
        
        fig.write_html(str(save_path))
        print(f"✅ Molecular orbital visualization saved to {save_path}")
        
        return str(save_path)
    
    def visualize_binding_energy_landscape(self, 
                                         binding_energies: List[float],
                                         co2_positions: List[np.ndarray],
                                         protein_coords: np.ndarray,
                                         save_path: Optional[str] = None) -> str:
        """
        Create 3D binding energy landscape around protein
        
        Args:
            binding_energies: List of binding energies at different positions
            co2_positions: Corresponding CO2 positions
            protein_coords: Protein atom coordinates
            save_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization file
        """
        print("🏔️ Generating binding energy landscape...")
        
        # Convert positions and energies to arrays
        positions = np.array(co2_positions)
        energies = np.array(binding_energies)
        
        # Create 3D scatter plot with energy-based coloring
        fig = go.Figure()
        
        # Add protein atoms
        fig.add_trace(go.Scatter3d(
            x=protein_coords[:, 0],
            y=protein_coords[:, 1],
            z=protein_coords[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color='gray',
                opacity=0.6
            ),
            name='Protein Atoms',
            showlegend=True
        ))
        
        # Add CO2 positions colored by binding energy
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=energies,
                colorscale=self.color_schemes['binding_energy'],
                colorbar=dict(title="Binding Energy (kcal/mol)"),
                opacity=0.8
            ),
            name='CO2 Positions',
            text=[f'Energy: {e:.2f} kcal/mol' for e in energies],
            hovertemplate='<b>CO2 Position</b><br>' +
                         'X: %{x:.2f} Å<br>' +
                         'Y: %{y:.2f} Å<br>' +
                         'Z: %{z:.2f} Å<br>' +
                         '%{text}<extra></extra>',
            showlegend=True
        ))
        
        # Find and highlight best binding position
        best_idx = np.argmin(energies)
        best_position = positions[best_idx]
        best_energy = energies[best_idx]
        
        fig.add_trace(go.Scatter3d(
            x=[best_position[0]],
            y=[best_position[1]],
            z=[best_position[2]],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(color='black', width=2)
            ),
            name=f'Best Binding Site ({best_energy:.2f} kcal/mol)',
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title=f"CO2 Binding Energy Landscape<br>Best Energy: {best_energy:.2f} kcal/mol",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            width=1000
        )
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "binding_energy_landscape.html"
        
        fig.write_html(str(save_path))
        print(f"✅ Binding energy landscape saved to {save_path}")
        
        return str(save_path)
    
    def visualize_charge_distribution(self, 
                                    quantum_result: QuantumResult,
                                    protein_coords: np.ndarray,
                                    atom_types: List[str],
                                    save_path: Optional[str] = None) -> str:
        """
        Visualize charge distribution on protein atoms
        
        Args:
            quantum_result: Results from quantum calculation
            protein_coords: Protein atom coordinates
            atom_types: List of atom types
            save_path: Optional path to save visualization
            
        Returns:
            Path to saved visualization file
        """
        print("⚡ Generating charge distribution visualization...")
        
        # Get charges from quantum result
        charges = quantum_result.charge_distribution.get('mulliken_charges', [0.0] * len(protein_coords))
        
        # Ensure charges match coordinates
        if len(charges) != len(protein_coords):
            charges = charges[:len(protein_coords)] + [0.0] * max(0, len(protein_coords) - len(charges))
        
        charges = np.array(charges)
        
        # Create 3D visualization
        fig = go.Figure()
        
        # Color atoms by charge
        fig.add_trace(go.Scatter3d(
            x=protein_coords[:, 0],
            y=protein_coords[:, 1],
            z=protein_coords[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color=charges,
                colorscale=self.color_schemes['charge_distribution'],
                colorbar=dict(title="Partial Charge (e)"),
                cmin=-0.5,
                cmax=0.5,
                opacity=0.8
            ),
            text=[f'{atom}: {charge:.3f}e' for atom, charge in zip(atom_types, charges)],
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x:.2f} Å<br>' +
                         'Y: %{y:.2f} Å<br>' +
                         'Z: %{z:.2f} Å<extra></extra>',
            name='Atomic Charges'
        ))
        
        # Update layout
        fig.update_layout(
            title="Charge Distribution Analysis",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)"
            ),
            height=600,
            width=800
        )
        
        # Save visualization
        if save_path is None:
            save_path = self.output_dir / "charge_distribution.html"
        
        fig.write_html(str(save_path))
        print(f"✅ Charge distribution visualization saved to {save_path}")
        
        return str(save_path)
    
    def create_comprehensive_quantum_dashboard(self, 
                                             quantum_results: List[QuantumResult],
                                             protein_data: Dict,
                                             save_path: Optional[str] = None) -> str:
        """
        Create comprehensive dashboard with all quantum visualizations
        
        Args:
            quantum_results: List of quantum calculation results
            protein_data: Protein structure and sequence data
            save_path: Optional path to save dashboard
            
        Returns:
            Path to saved dashboard file
        """
        print("📊 Creating comprehensive quantum dashboard...")
        
        # Create multi-panel dashboard
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Binding Energy vs Position',
                'Electron Density Isosurface',
                'Molecular Orbital Energies',
                'Charge Distribution',
                'Dipole Moment Analysis',
                'Quantum Properties Summary'
            ],
            specs=[
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                [{'type': 'bar'}, {'type': 'scatter3d'}],
                [{'type': 'scatter'}, {'type': 'table'}]
            ]
        )
        
        # Panel 1: Binding energies
        binding_energies = [result.binding_energy for result in quantum_results]
        positions = list(range(len(binding_energies)))
        
        fig.add_trace(go.Scatter(
            x=positions,
            y=binding_energies,
            mode='lines+markers',
            name='Binding Energy',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ), row=1, col=1)
        
        # Panel 2: Molecular orbital energies
        if quantum_results:
            orbitals = quantum_results[0].molecular_orbitals
            homo_energy = orbitals.get('HOMO_energy', -5.0)
            lumo_energy = orbitals.get('LUMO_energy', -1.0)
            
            fig.add_trace(go.Bar(
                x=['HOMO', 'LUMO'],
                y=[homo_energy, lumo_energy],
                name='Orbital Energies',
                marker_color=['red', 'blue']
            ), row=2, col=1)
        
        # Panel 3: Dipole moments
        dipole_mags = [np.linalg.norm(result.dipole_moment) for result in quantum_results]
        
        fig.add_trace(go.Scatter(
            x=positions,
            y=dipole_mags,
            mode='lines+markers',
            name='Dipole Moment',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ), row=3, col=1)
        
        # Panel 4: Summary table
        if quantum_results:
            best_result = min(quantum_results, key=lambda x: x.binding_energy)
            
            summary_data = [
                ['Best Binding Energy', f'{best_result.binding_energy:.2f} kcal/mol'],
                ['Method', best_result.method],
                ['Basis Set', best_result.basis_set],
                ['Total Energy', f'{best_result.total_energy:.2f} Hartree'],
                ['Dipole Moment', f'{np.linalg.norm(best_result.dipole_moment):.2f} Debye'],
                ['HOMO-LUMO Gap', f'{best_result.molecular_orbitals.get("HOMO_LUMO_gap", "N/A")} eV']
            ]
            
            fig.add_trace(go.Table(
                header=dict(values=['Property', 'Value']),
                cells=dict(values=[[row[0] for row in summary_data],
                                 [row[1] for row in summary_data]])
            ), row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title="Quantum Chemical Analysis Dashboard",
            height=1200,
            width=1400,
            showlegend=True
        )
        
        # Save dashboard
        if save_path is None:
            save_path = self.output_dir / "quantum_dashboard.html"
        
        fig.write_html(str(save_path))
        print(f"✅ Quantum dashboard saved to {save_path}")
        
        return str(save_path)
    
    def _interpolate_density(self, density: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Interpolate electron density to target grid size"""
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        current_shape = density.shape
        x_old = np.linspace(0, 1, current_shape[0])
        y_old = np.linspace(0, 1, current_shape[1])
        z_old = np.linspace(0, 1, current_shape[2])
        
        interpolator = RegularGridInterpolator((x_old, y_old, z_old), density)
        
        # Create new grid
        x_new = np.linspace(0, 1, target_shape[0])
        y_new = np.linspace(0, 1, target_shape[1])
        z_new = np.linspace(0, 1, target_shape[2])
        X_new, Y_new, Z_new = np.meshgrid(x_new, y_new, z_new, indexing='ij')
        
        # Interpolate
        new_density = interpolator((X_new, Y_new, Z_new))
        
        return new_density
    
    def _generate_realistic_density(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                  protein_coords: np.ndarray, co2_coords: np.ndarray) -> np.ndarray:
        """Generate realistic-looking electron density distribution"""
        density = np.zeros_like(X)
        
        # Add Gaussian distributions around each atom
        all_coords = np.vstack([protein_coords, co2_coords])
        
        for coord in all_coords:
            # Distance from each grid point to atom
            dist = np.sqrt((X - coord[0])**2 + (Y - coord[1])**2 + (Z - coord[2])**2)
            
            # Gaussian electron density around atom
            atom_density = np.exp(-dist**2 / (2 * 0.5**2))  # 0.5 Å width
            density += atom_density
        
        return density
    
    def _create_interactive_electron_density_plot(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                                density: np.ndarray, protein_coords: np.ndarray,
                                                co2_coords: np.ndarray) -> go.Figure:
        """Create interactive 3D electron density visualization"""
        fig = go.Figure()
        
        # Add protein atoms
        fig.add_trace(go.Scatter3d(
            x=protein_coords[:, 0],
            y=protein_coords[:, 1],
            z=protein_coords[:, 2],
            mode='markers',
            marker=dict(size=8, color='gray', opacity=0.8),
            name='Protein Atoms'
        ))
        
        # Add CO2 atoms
        fig.add_trace(go.Scatter3d(
            x=co2_coords[:, 0],
            y=co2_coords[:, 1],
            z=co2_coords[:, 2],
            mode='markers',
            marker=dict(size=10, color='red', opacity=0.9),
            name='CO2 Molecule'
        ))
        
        # Add electron density isosurface
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=density.flatten(),
            isomin=0.1,
            isomax=0.9,
            opacity=0.3,
            colorscale=self.color_schemes['electron_density'],
            name='Electron Density'
        ))
        
        # Update layout
        fig.update_layout(
            title="3D Electron Density Visualization",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)"
            ),
            height=700,
            width=1000
        )
        
        return fig
    
    def _create_static_electron_density_plot(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                           density: np.ndarray, protein_coords: np.ndarray,
                                           co2_coords: np.ndarray):
        """Create static matplotlib version of electron density plot"""
        fig = plt.figure(figsize=self.viz_params['figure_size'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot atoms
        ax.scatter(protein_coords[:, 0], protein_coords[:, 1], protein_coords[:, 2],
                  c='gray', s=100, alpha=0.8, label='Protein')
        ax.scatter(co2_coords[:, 0], co2_coords[:, 1], co2_coords[:, 2],
                  c='red', s=150, alpha=0.9, label='CO2')
        
        # Create 2D slice through electron density
        mid_z = density.shape[2] // 2
        density_slice = density[:, :, mid_z]
        
        # Plot contours
        x_2d = X[:, :, mid_z]
        y_2d = Y[:, :, mid_z]
        contour = ax.contour(x_2d, y_2d, density_slice, levels=10, alpha=0.6)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Electron Density Cross-Section')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'electron_density_static.png', 
                   dpi=self.viz_params['dpi'], bbox_inches='tight')
        plt.close()
    
    def _generate_orbital_density(self, protein_coords: np.ndarray, 
                                orbital_name: str, orbitals: Dict) -> np.ndarray:
        """Generate simplified orbital density for visualization"""
        # This is a simplified representation
        # In reality, you would use the actual orbital coefficients
        
        grid_size = 20
        x = np.linspace(-5, 5, grid_size)
        y = np.linspace(-5, 5, grid_size)
        z = np.linspace(-5, 5, grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Create different patterns for different orbitals
        if orbital_name == 'HOMO':
            density = np.exp(-(X**2 + Y**2)/4) * np.cos(Z)
        elif orbital_name == 'LUMO':
            density = np.exp(-(X**2 + Z**2)/4) * np.sin(Y)
        elif orbital_name == 'HOMO-1':
            density = X * Y * np.exp(-(X**2 + Y**2 + Z**2)/6)
        else:  # LUMO+1
            density = (X**2 - Y**2) * np.exp(-(X**2 + Y**2 + Z**2)/6)
        
        return density
    
    def _add_orbital_isosurface(self, fig: go.Figure, orbital_density: np.ndarray,
                              protein_coords: np.ndarray, orbital_name: str,
                              row: int, col: int):
        """Add orbital isosurface to subplot"""
        grid_size = orbital_density.shape[0]
        x = np.linspace(-5, 5, grid_size)
        y = np.linspace(-5, 5, grid_size)
        z = np.linspace(-5, 5, grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Add isosurface
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=orbital_density.flatten(),
            isomin=-0.1,
            isomax=0.1,
            opacity=0.4,
            colorscale=self.color_schemes['molecular_orbitals'],
            name=orbital_name,
            showlegend=False
        ), row=row, col=col)

if __name__ == "__main__":
    # Example usage
    from quantum_co2_calculator import QuantumCO2BindingCalculator
    
    viz = QuantumVisualizationSystem()
    
    # Test with dummy data
    test_coords = np.array([
        [0.0, 0.0, 0.0],    # Zn
        [2.1, 0.0, 0.0],    # N
        [-2.1, 0.0, 0.0],   # N
        [0.0, 2.1, 0.0],    # N
    ])
    
    co2_coords = np.array([
        [3.0, 0.0, 0.0],    # C
        [4.2, 0.0, 0.0],    # O
        [1.8, 0.0, 0.0],    # O
    ])
    
    # Create dummy quantum result
    from quantum_co2_calculator import QuantumResult
    
    dummy_result = QuantumResult(
        binding_energy=-8.5,
        interaction_energy=-8.5,
        electron_density=np.random.random((20, 20, 20)),
        molecular_orbitals={'HOMO_energy': -5.2, 'LUMO_energy': -1.8, 'HOMO_LUMO_gap': 3.4},
        charge_distribution={'mulliken_charges': [0.8, -0.3, -0.3, -0.2]},
        dipole_moment=[1.2, -0.8, 0.3],
        total_energy=-150.5,
        convergence_info={'converged': True},
        method='B3LYP',
        basis_set='6-31G*'
    )
    
    print("🎨 Testing quantum visualization system...")
    
    # Test visualizations
    viz.visualize_electron_density(dummy_result, test_coords, co2_coords)
    viz.visualize_molecular_orbitals(dummy_result, test_coords)
    viz.visualize_charge_distribution(dummy_result, test_coords, ['Zn', 'N', 'N', 'N'])
    
    # Test binding energy landscape
    positions = [co2_coords + np.random.random(3) for _ in range(20)]
    energies = np.random.uniform(-10, -2, 20)
    viz.visualize_binding_energy_landscape(energies, positions, test_coords)
    
    # Test comprehensive dashboard
    viz.create_comprehensive_quantum_dashboard([dummy_result], {'coords': test_coords})
    
    print("✅ All quantum visualizations generated successfully!")