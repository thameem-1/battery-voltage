import os
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.base import MultipleFeaturizer

cif_folder_path = "MP_CIFs_Battery" # path to cif folder
material_id_path = "charge_discharge_pairs.xlsx" # path to material id pairs and average voltage excel file

pair_df = pd.read_excel(material_id_path)

featurizer = MultipleFeaturizer([ElementProperty.from_preset("magpie", impute_nan=True)])

# Featurize all unique material_ids
unique_ids = pd.unique(pair_df[['id_discharge', 'id_charge']].values.ravel())
records = []

for mid in tqdm(unique_ids, desc="Featurizing"):
    cif_path = os.path.join(cif_folder_path, f"{mid}.cif")
    struct = Structure.from_file(cif_path)
    comp = struct.composition
    feats = featurizer.featurize(comp)
    record = {"material_id": mid}
    record.update(dict(zip(featurizer.feature_labels(), feats)))
    records.append(record)


desc_df = pd.DataFrame(records) # 132 features for each material id

