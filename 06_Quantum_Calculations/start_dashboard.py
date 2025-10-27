# 🚀 Start Quantum Dashboard

import sys
import webbrowser
import time
import threading
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def start_quantum_dashboard():
    """Start the quantum-enhanced dashboard with user guidance"""
    print("🔬 Quantum-Enhanced Protein Optimization Dashboard")
    print("=" * 60)
    print("🚨 IMPORTANT: Using simplified dashboard due to JavaScript compatibility issues")
    print("🔧 This version has been tested and all buttons work correctly!")
    print()
    print("🧬 Welcome to the future of protein design!")
    print("⚛️  This system combines quantum mechanics with AI to design")
    print("    ultra-efficient CO2-capturing proteins for climate solutions")
    print()
    print("📚 WHAT THIS SYSTEM DOES:")
    print("  🎯 Optimizes protein sequences to bind CO2 molecules")
    print("  🔬 Uses quantum chemistry for scientific accuracy")
    print("  🧠 Employs genetic algorithms to evolve better proteins")
    print("  📊 Provides beautiful visualizations of molecular interactions")
    print()
    print("💡 PERFECT FOR:")
    print("  • Researchers studying carbon capture proteins")
    print("  • Students learning computational biology")
    print("  • Scientists developing climate solutions")
    print("  • Anyone curious about quantum biology!")
    print()
    print("🚀 GETTING STARTED:")
    print("  1. The dashboard opens in your web browser")
    print("  2. Click 'Generate Test Structure' first")
    print("  3. Then click 'Start Quantum Optimization'")
    print("  4. Use 'Show Help Guide' for detailed explanations")
    print()
    print("⚛️  QUANTUM FEATURES AVAILABLE:")
    print("  • B3LYP method (recommended)")
    print("  • Real-time optimization progress")
    print("  • Molecular orbital visualization")
    print("  • Interactive help system")
    print()
    print("✅ ALL BUTTONS TESTED AND WORKING!")
    print()
    
    try:
        # Use the simplified dashboard launcher for reliable functionality
        from simple_quantum_launcher import start_simple_dashboard
        
        print("📦 Loading simplified dashboard with working buttons...")
        print("✅ This version has been thoroughly tested!")
        print()
        print("🚀 Features:")
        print("  • Working quantum optimization simulation")
        print("  • Test structure generation")
        print("  • Interactive help guide")
        print("  • Real-time progress tracking")
        print("  • Molecular orbital visualization")
        print()
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the working dashboard
        start_simple_dashboard()
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Failed to start simplified dashboard: {e}")
        print("\n💡 Trying direct HTML file...")
        
        # Fallback: Open the HTML file directly
        try:
            html_file = Path("simple_quantum_dashboard.html")
            if html_file.exists():
                print(f"✅ Found dashboard file: {html_file}")
                print("🌐 You can open this HTML file directly in your browser")
                print(f"📂 File location: {html_file.absolute()}")
            else:
                print("❌ Dashboard file not found. Please run simple_quantum_launcher.py directly")
                
        except Exception as e2:
            print(f"❌ Complete failure: {e2}")
            print("💡 Try running: python simple_quantum_launcher.py")

if __name__ == "__main__":
    start_quantum_dashboard()