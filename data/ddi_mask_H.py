from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, Recap
import dill
import numpy as np

NDCList = dill.load(open('./idx2SMILES.pkl', 'rb'))
voc = dill.load(open('./voc_final.pkl', 'rb'))
med_voc = voc['med_voc']

fraction = []
fracSet_dict = {}
print(len(med_voc.idx2word))
i=0
for k, v in med_voc.idx2word.items():
    tempF = set()

    for SMILES in NDCList[v]:
        try:
            m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
            # m = Recap.RecapDecompose(m)
            for frac in m:
                try:
                    mol = Chem.AddHs(Chem.MolFromSmiles(frac))
                    # rw_mol = Chem.RWMol(mol)
                    
                    # for atom in reversed(list(rw_mol.GetAtoms())):
                    #     if atom.GetSymbol() == '*':
                    #         rw_mol.RemoveAtom(atom.GetIdx())
                    
                    # mol = rw_mol.GetMol()

                    # mol.UpdatePropertyCache(strict=False)
                    # Chem.GetSymmSSSR(mol)

                    status = AllChem.EmbedMolecule(mol)
                    AllChem.MMFFOptimizeMolecule(mol)
                    if status == 0:
                        tempF.add(frac)
                        fracSet_dict[frac] = mol
                except:
                    pass
        except:
            pass
    i += 1
    print(i)
    fraction.append(tempF)

fracSet = []
for i in fraction:
    fracSet += i
fracSet = list(set(fracSet))

ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))

for i, fracList in enumerate(fraction):
    for frac in fracList:
        ddi_matrix[i, fracSet.index(frac)] = 1

print(len(fracSet))
with Chem.SDWriter('substructure_smiles_3d.sdf') as writer:
    for mol in fracSet:
        writer.write(fracSet_dict[mol])
dill.dump(ddi_matrix, open('ddi_mask_H.pkl', 'wb'))
dill.dump(fracSet, open('substructure_smiles.pkl', 'wb'))
