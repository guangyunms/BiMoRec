from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pubchempy as pcp
import dill

with open('./idx2SMILES.pkl', 'rb') as f:
    idx2SMILES = dill.load(f)

with open('./mimic4/idx2drug.pkl', 'rb') as f:
    idx2drug = dill.load(f)

SMILES_list = []
for value in idx2SMILES.values():
    for SMILES in value:
        SMILES_list.append(SMILES)
        

SMILES_list = list(set(SMILES_list))
print(len(SMILES_list))

cid_list = []
SMILES2CID = {}
a = 0
for SMILES in SMILES_list:
    compound = pcp.get_compounds(SMILES, 'smiles')
    if(len(compound) == 1 and compound[0].cid != None):
        cid_list.append(compound[0].cid)
        SMILES2CID[SMILES] = compound[0].cid
    
    a += 1
    print(a)
print(len(cid_list))

dill.dump(SMILES2CID, open('./SMILES2CID.pkl', 'wb'))


with open('./SMILES2CID.pkl', 'rb')  as f:
    SMILES2CID = dill.load(f)

with open('./substructure_smiles.pkl', 'rb') as f:
    substructure_smiles = dill.load(f)


print('substructure_smiles_list长度：{}'.format(len(substructure_smiles)))
sub_cid_list = []
SUBSMILES2CID = {}
a, b = 0, 0
for substructure in substructure_smiles:
    try:
        source = Chem.MolFromSmiles(substructure)
        Draw.MolToFile(source, 'source.png')
        mol = Chem.AddHs(source)
        Draw.MolToFile(mol, 'mol.png')
        AllChem.EmbedMolecule(mol)
        Draw.MolToFile(mol, 'embed.png')
        AllChem.MMFFOptimizeMolecule(mol)
        Draw.MolToFile(mol, 'mmff.png')
        sub_cid_list.append(mol)
        
        compound = pcp.get_compounds(substructure, 'smiles')
        if(len(compound) == 1 and compound[0].cid != None):
            sub_cid_list.append(compound[0].cid)
            SUBSMILES2CID[substructure] = compound[0].cid
        else:
            print('意外亚结构：{}'.format(substructure))
    except:
        print('失败亚结构：{}'.format(substructure))
        SUBSMILES2CID[substructure] = None
        b += 1
    a += 1
    print(a)
print('失败次数：{}'.format(b))
print('cid_list长度：{}'.format(len(sub_cid_list)))

dill.dump(SUBSMILES2CID, open('./SUBSMILES2CID.pkl', 'wb'))
