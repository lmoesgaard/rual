from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter

def get_structure(smi, name):
    mol = Chem.MolFromSmiles(smi)
    try:
        mol = Chem.AddHs(mol)
    except ValueError:
        print("get_structure: could not add hydrogens to the molecule '{}'".format(smi))
        return None
    except RuntimeError:
        print("get_structure: could not add hydrogens to the molecule '{}'".format(smi))
        raise
    new_mol = Chem.Mol(mol)
    try:
        AllChem.EmbedMolecule(new_mol)
    except ValueError:
        print("Error: embed_simple: '{}' could not convert to 3D".format(Chem.MolToSmiles(mol)))
        return None
    try:
        AllChem.MMFFOptimizeMolecule(new_mol)
    except ValueError:
        print("Error: embed_simple: '{}' could not optimize molecule with MMFF".format(Chem.MolToSmiles(mol)))
        return None
    new_mol.SetProp("_Name", name)
    return new_mol

def smi2sdfs(smis, filename):
    structures = [get_structure(smi, str(i)) for i, smi in enumerate(smis)]
    molecules = [mol for mol in structures if mol is not None]
    w = SDWriter(filename)
    for mol in molecules:
        w.write(mol)
    w.close()
