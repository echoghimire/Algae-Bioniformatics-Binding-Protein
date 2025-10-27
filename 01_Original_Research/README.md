# 🔬 Phase 1: Original Research

## Overview
This directory contains the foundational work that started the algae protein optimization project. It represents the initial exploration of genetic algorithms for protein sequence optimization.

## Files

### `runfile.py`
**Purpose:** Original genetic algorithm implementation  
**Status:** Historical baseline  
**Key Features:**
- Basic genetic algorithm for protein optimization
- Single-objective fitness function
- Simple mutation and crossover operations
- Target sequence comparison

### `Trials Data/`
**Purpose:** Original 100 trial results  
**Structure:**
```
Trials Data/
├── Carbonic Anhydrase FASTA sequence.fasta
└── Trial X/
    └── Binder Optimization Table Trial X.txt
```

**Contents:**
- 100 individual optimization trials (Trial 1 through Trial 100)
- Each trial contains optimization results and sequence data
- Original target sequence file for Carbonic Anhydrase

## Historical Significance

This phase established:
- ✅ Proof of concept for GA-based protein optimization
- ✅ Baseline performance metrics
- ✅ Initial trial data structure
- ✅ Foundation for future algorithm enhancements

## Limitations Identified

- ❌ Single-objective optimization only
- ❌ Limited fitness function sophistication
- ❌ No visualization or analysis tools
- ❌ Manual result interpretation required
- ❌ No automated workflow

## Evolution Path

This work led directly to:
1. **Phase 2:** Enhanced algorithm development with multi-objective optimization
2. **Phase 3:** Comprehensive visualization and analysis systems
3. **Phase 4:** User-friendly dashboard interfaces
4. **Phase 5:** Advanced 3D molecular visualization

## Running Original Code

```bash
cd 01_Original_Research
python runfile.py
```

**Note:** This code is preserved for historical reference and comparison with enhanced versions developed in later phases.