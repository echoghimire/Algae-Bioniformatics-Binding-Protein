# Install required libraries
!pip install biopython deap matplotlib pandas py3Dmol --quiet

# === Imports ===
import pandas as pd
import numpy as np
import requests
import random
from Bio import SeqIO
import matplotlib.pyplot as plt
from deap import base, creator, tools
from google.colab import files
import py3Dmol
import multiprocessing

# === Amino Acids List ===
AA = "ACDEFGHIKLMNPQRSTVWY"

# === DEAP Class Deletion & Creation ===
for cname in ['FitnessMulti', 'Individual']:
    if hasattr(creator, cname): delattr(creator, cname)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# === Utility Functions ===
def fetch_uniprot_details(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        resp = requests.get(url)
        if resp.status_code != 200: return {}
        js = resp.json()
        details = {
            "Protein Name": js.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""),
            "Organism": js.get("organism", {}).get("scientificName", ""),
            "Sequence Length": js.get("sequence", {}).get("length", ""),
            "Function": next((ref.get("value") for ref in js.get("comments", []) if ref.get('type') == 'FUNCTION'), ""),
            "UniProt ID": accession,
        }
        return details
    except Exception as e:
        return {"Error": str(e)}

def fetch_alphafold_structure(accession):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    try:
        response = requests.get(url)
        if response.status_code != 200: return None
        data = response.json()
        if not data or not isinstance(data, list): return None
        model_entry = None
        for entry in data:
            if entry.get("model", "") == "model_1": model_entry = entry; break
        if model_entry is None: model_entry = data[0]
        return {
            "pdb": model_entry.get("pdbUrl", ""),
            "sequence": model_entry.get("sequence", ""),
            "accession": model_entry.get("accession", accession),
            "description": model_entry.get("description", ""),
            "thumbnail_url": f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.png"
        }
    except Exception:
        return None

def random_seq(length):
    return [random.choice(AA) for _ in range(length)]

# === Genetic Algorithm Functions (GLOBAL SCOPE!) ===
def mutate_sequence(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(AA)
    return individual,

def mate_wrapper(ind1, ind2, crossover_rate):
    if random.random() < crossover_rate:
        tools.cxOnePoint(ind1, ind2)

def evaluate_binder(individual, target_sequence):
    binder = ''.join(individual)
    matches = sum(a == b for a, b in zip(binder, target_sequence))
    percent_id = matches / len(binder)
    return percent_id, len(binder)

# === Main Genetic Algorithm ===
def run_ga_binder(target_sequence, ngen, pop_size, binder_length, mutation_rate, crossover_rate, elite_count):
    toolbox = base.Toolbox()
    toolbox.register("attr_seq", lambda: random_seq(binder_length))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_seq)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate_wrapper, crossover_rate=crossover_rate)
    toolbox.register("mutate", mutate_sequence, mutation_rate=mutation_rate)
    toolbox.register("select", tools.selNSGA2)

    pop = [toolbox.individual() for _ in range(pop_size)]

    # Evaluate initial population
    fitnesses = [evaluate_binder(ind, target_sequence[:binder_length]) for ind in pop]
    for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit

    generations, best_pdockq, avg_pdockq = [], [], []

    for gen in range(ngen):
        pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
        elites = pop[:elite_count] if elite_count > 0 else []
        offspring = toolbox.select(pop, len(pop) - elite_count)
        offspring = list(map(toolbox.clone, offspring))

        # ADAPTIVE mutation/crossover rates
        base_cxpb = 0.5
        base_mutpb = 0.2
        min_cxpb = 0.3
        max_mutpb = 0.4
        if len(avg_pdockq) > 1:
          if abs(avg_pdockq[-1] - avg_pdockq[-2]) < 1e-3:
            mutpb = min(base_mutpb * 1.5, max_mutpb)
          else:
            mutpb = base_mutpb
            cxpb = max(base_cxpb * (0.99 ** gen), min_cxpb)
        else:
          mutpb = base_mutpb
          cxpb = base_cxpb


        # Offspring generation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                toolbox.mate(child1, child2)
            if np.random.random() < mutpb:
                toolbox.mutate(child1)
            if np.random.random() < mutpb:
                toolbox.mutate(child2)
            del child1.fitness.values
            del child2.fitness.values

        # Parallel fitness evaluation for offspring
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        with multiprocessing.Pool() as pool:
            fitnesses = pool.starmap(evaluate_binder,
                [(ind, target_sequence[:binder_length]) for ind in invalid_inds])
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

        # Elitism
        pop[:] = elites + offspring
        pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True)
        best = pop[0]
        best_pdockq.append(best.fitness.values[0])
        avg_pdockq.append(np.mean([ind.fitness.values[0] for ind in pop]))
        generations.append(gen)

    return generations, best_pdockq, avg_pdockq, ''.join(pop[0])

# === FASTA Processing & Visualization ===
def process_fasta(fasta_path, ngen=100, pop_size=30, binder_length=25,
                  mutation_rate=0.1, crossover_rate=0.5, elite_count=2, top_n=5):
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))[:top_n]
        if not records:
            print("No records found in FASTA")
            return None, None
        results_summary, details_table = [], []
        for idx, record in enumerate(records):
            if "|" in record.id:
                acc = record.id.split("|")[1]
            else:
                acc = record.id
            seq = str(record.seq)
            uniprot_details = fetch_uniprot_details(acc)
            af = fetch_alphafold_structure(acc)
            details_row = {
                "Accession": acc,
                "Protein Name": uniprot_details.get("Protein Name", ""),
                "Organism": uniprot_details.get("Organism", ""),
                "Sequence Length": uniprot_details.get("Sequence Length", ""),
                "Function": uniprot_details.get("Function", ""),
                "AlphaFold PDB": af["pdb"] if af and af.get("pdb") else "",
                "AlphaFold Thumbnail": af["thumbnail_url"] if af and af.get("thumbnail_url") else "",
                "AlphaFold Description": af["description"] if af and af.get("description") else ""
            }
            details_table.append(details_row)
            print(f"\nProcessing [{idx+1}/{len(records)}]: {acc} ({record.id})")
            if not af or not af.get("pdb"):
                print("  Structure not found in AlphaFold.")
                results_summary.append({
                    "Accession": acc,
                    "Status": "Structure not found",
                    "Best Binder": "",
                    "PDB": "",
                    "Best Score": ""
                })
                continue
            ga_gens, ga_best_scores, ga_avg_scores, best_binder = run_ga_binder(
                seq, ngen, pop_size, binder_length, mutation_rate, crossover_rate, elite_count)
            plt.figure(figsize=(8, 4))
            plt.plot(ga_gens, ga_best_scores, label="Best Score")
            plt.plot(ga_gens, ga_avg_scores, label="Avg Score", linestyle="--")
            plt.xlabel("Generation")
            plt.ylabel("Simulated pDockQ Score")
            plt.title(f"Binder Optimization - {acc}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            results_summary.append({
                "Accession": acc,
                "Status": "Processed",
                "Best Binder": best_binder,
                "PDB": af["pdb"],
                "Best Score": ga_best_scores[-1] if ga_best_scores else ""
            })
        results_df = pd.DataFrame(results_summary)
        details_df = pd.DataFrame(details_table)
        print("\nProtein Details:")
        display(details_df)
        print("\nBinder Optimization Results:")
        display(results_df)
        return details_df, results_df
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        print(traceback.format_exc())
        return None, None

# === Run: Upload & Process FASTA ===
uploaded = files.upload()
fasta_path = list(uploaded.keys())[0]
details_df, results_df = process_fasta(
    fasta_path,
    ngen=100, pop_size=30, binder_length=25, mutation_rate=0.1,
    crossover_rate=0.5, elite_count=2, top_n=5
)

# === Visualization ===
if details_df is not None and results_df is not None:
    for i in range(len(details_df)):
        print(f"\n=== Visualization for Entry {i+1}/{len(details_df)} ===")
        accession = details_df.iloc[i]["Accession"]
        protein_name = details_df.iloc[i]["Protein Name"]
        pdb_url = details_df.iloc[i]["AlphaFold PDB"]
        binder_seq = results_df.iloc[i]["Best Binder"]
        # Visualize protein if structure available
        if pdb_url and pdb_url.startswith("http"):
            pdb_string = requests.get(pdb_url).text
            view = py3Dmol.view(width=600, height=400)
            view.addModel(pdb_string, "pdb")
            view.setStyle({'cartoon': {'color':'spectrum'}})
            view.zoomTo()
            print(f"Showing structure for UniProt accession {accession}")
            view.show()
        else:
            print(f"No AlphaFold PDB structure available for {accession}.")
        # Visualize binder (always attempt)
        esmfold_url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        binder_pdb = requests.post(esmfold_url, data=binder_seq).text
        binder_view = py3Dmol.view(width=600, height=300)
        binder_view.addModel(binder_pdb, "pdb")
        binder_view.setStyle({'stick':{}})
        binder_view.zoomTo()
        print(f"Showing binder peptide structure for sequence: {binder_seq}")
        binder_view.show()
        # Combined view if protein available
        if pdb_url and pdb_url.startswith("http"):
            combined_view = py3Dmol.view(width=800, height=400)
            combined_view.addModel(pdb_string, "pdb")
            combined_view.setStyle({'cartoon': {'color':'spectrum'}})
            combined_view.addModel(binder_pdb, "pdb")
            combined_view.setStyle({'stick':{}}, model=2)
            combined_view.zoomTo()
            print("Showing protein and binder together (not docked).")
            combined_view.show()
else:
    print("No valid results to visualize.")
