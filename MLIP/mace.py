from mace.calculators import mace_mp
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from ase.io import read


atoms = read("path to cif/poscar")
calc = mace_mp(
    model="medium-omat-0",
    device="cuda",         
    default_dtype="float64" # more accurate for relaxation
)
atoms.calc = calc
opt = LBFGS(FrechetCellFilter(atoms))
opt.run(fmax=0.05, steps=500)
