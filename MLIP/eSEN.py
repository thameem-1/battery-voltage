from huggingface_hub import hf_hub_download
from huggingface_hub import login
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator
from ase.io import read
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter

from torch.serialization import add_safe_globals
add_safe_globals([slice]) 

checkpoint_path = hf_hub_download(
    repo_id="facebook/UMA",
    filename="uma-s-1.pt",
    subfolder="checkpoints"
)

# Load prediction unit
predictor = load_predict_unit(checkpoint_path, inference_settings="default", device="cuda")

calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = read("path to the cif/poscar file")

atoms.calc = calc

# Relaxation
opt = LBFGS(FrechetCellFilter(atoms))
opt.run(fmax=0.02, steps=500)
