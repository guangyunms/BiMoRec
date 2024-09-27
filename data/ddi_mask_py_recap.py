from rdkit import Chem
from rdkit.Chem import BRICS, Recap
import dill
import numpy as np

NDCList = dill.load(open('./idx2SMILES.pkl', 'rb'))
voc = dill.load(open('./voc_final.pkl', 'rb'))
med_voc = voc['med_voc']
flag=0
fraction = []
for k, v in med_voc.idx2word.items():
    tempF = set()

    for SMILES in NDCList[v]:
        try:
            hierarch = Recap.RecapDecompose(Chem.MolFromSmiles(SMILES))
            for frac in list(hierarch.GetAllChildren().keys()):
                tempF.add(frac)
        except:
            flag+=1
            pass

    fraction.append(tempF)

fracSet = []
for i in fraction:
    fracSet += i
fracSet = list(set(fracSet))

ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))

for i, fracList in enumerate(fraction):
    for frac in fracList:
        ddi_matrix[i, fracSet.index(frac)] = 1
print(flag)
dill.dump(ddi_matrix, open('ddi_mask_H_recap.pkl', 'wb'))
dill.dump(fracSet, open('substructure_smiles_recap.pkl', 'wb'))
