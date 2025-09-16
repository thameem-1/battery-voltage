
import glob, json
import numpy as np, pandas as pd
from tqdm import tqdm
from pymatgen.core.structure import Structure
from concurrent.futures import ThreadPoolExecutor
import torch
from torch_geometric.data import Data

class GaussianDistance:
    def __init__(self, dmin, dmax, step, var=None):
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var or step

    def expand(self, distances):
        return np.exp(-(distances[..., None] - self.filter)**2 / self.var**2)

# CGCNN Featurizer
def build_cgcnn_config(cif_folder, config_path="cgcnn_node_config.json"):
    atomic_numbers = set()
    for cif in glob.glob(f"{cif_folder}/**/*.cif", recursive=True):
        structure = Structure.from_file(cif)
        atomic_numbers.update(structure.atomic_numbers)
    unique_z = sorted(atomic_numbers)
    vectors = np.eye(len(unique_z)).tolist()
    with open(config_path, 'w') as f:
        json.dump({'atomic_numbers': unique_z, 'node_vectors': vectors}, f)
    return config_path

def load_atom_init(path):
    with open(path) as f:
        config = json.load(f)
    return {z: np.array(v) for z, v in zip(config['atomic_numbers'], config['node_vectors'])}

def process_cif_cgcnn(file_path, atom_init, gdf, cutoff=8.0, max_nbr=12):
    struct = Structure.from_file(file_path)
    atom_fea = np.vstack([atom_init[site.specie.number] for site in struct])
    all_nbrs = [sorted(nbrs, key=lambda x: x[1])[:max_nbr]
                for nbrs in struct.get_all_neighbors(cutoff, True)]
    edge_index, edge_attr = [], []
    for i, nbrs in enumerate(all_nbrs):
        for nbr in nbrs:
            j, dist = nbr[2], nbr[1]
            edge_index.append([i, j])
            edge_attr.append(gdf.expand(dist))
    return Data(
        x=torch.tensor(atom_fea, dtype=torch.float32),
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.tensor(np.vstack(edge_attr), dtype=torch.float32)
    )

# Parallel Dataset & Split
def make_graph(row, atom_init, gdf, cif_dir):
    try:
        chg = glob.glob(f"{cif_dir}/**/{row['id_charge']}.cif", recursive=True)
        dis = glob.glob(f"{cif_dir}/**/{row['id_discharge']}.cif", recursive=True)
        if not (chg and dis): return None

        g1 = process_cif_cgcnn(dis[0], atom_init, gdf)
        g2 = process_cif_cgcnn(chg[0], atom_init, gdf)
        target = torch.tensor([row['average_voltage']], dtype=torch.float32)
        return (g1, g2, target, row)
    except Exception as e:
        print(f"{row['id_discharge']}/{row['id_charge']}: {e}")
        return None

def generate_dual_cgcnn_graph_dataset_parallel(cif_dir, csv_path, atom_init_path, max_workers=12):
    atom_init = load_atom_init(atom_init_path)
    gdf = GaussianDistance(0, 8, 0.2)
    df = pd.read_excel(csv_path).dropna(subset=['average_voltage'])
    df = df[(df['average_voltage'] <= 7) & (df['average_voltage'] >= -2)]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_graph, row, atom_init, gdf, cif_dir) for _, row in df.iterrows()]
        for future in tqdm(futures):
            result = future.result()
            if result:
                results.append(result)

    graphs = [(g1, g2, target) for g1, g2, target, _ in results]
    df_valid = pd.DataFrame([row for _, _, _, row in results])
    return graphs, df_valid



# Execution
cif_dir = "MP_CIFs_Battery" # path to cif folder
csv_path = "charge_discharge_pairs.xlsx" # path to excel
atom_init_path = build_cgcnn_config(cif_dir)
graphs, df_valid = generate_dual_cgcnn_graph_dataset_parallel(cif_dir, csv_path, atom_init_path)

import joblib

# Save both together in one file
joblib.dump((graphs, df_valid), "dual_graph_dataset_.pkl")
