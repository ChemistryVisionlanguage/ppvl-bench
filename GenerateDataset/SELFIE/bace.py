#this is for training dataset
import warnings
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os
import datetime
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from tqdm import tqdm
import warnings
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image
from IPython.display import display
import selfies as sf

warnings.filterwarnings("ignore", category=UserWarning, module=".*rdkit.*")

def random_sample_examples(bace, sample_size):
    positive_examples = bace[bace["Class"] == 1].sample(int(sample_size / 2))
    negative_examples = bace[bace["Class"] == 0].sample(int(sample_size / 2))
    selfies_pos = [sf.encoder(s) for s in positive_examples["mol"].tolist()]
    selfies_neg = [sf.encoder(s) for s in negative_examples["mol"].tolist()]

    class_label = positive_examples["Class"].tolist() + negative_examples["Class"].tolist()
    class_label = ["Yes" if i == 1 else "No" for i in class_label]

    bace_examples = list(zip(selfies_pos + selfies_neg, class_label))
    return bace_examples

def top_k_scaffold_similar_molecules(target_smiles, bace_data, k):
    bace_data = bace_data[bace_data["mol"] != target_smiles]
    molecule_smiles_list = bace_data['mol'].tolist()
    label_list = bace_data['Class'].tolist()
    label_list = ["Yes" if i == 1 else "No" for i in label_list]

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is not None:
        target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
    else:
        print("Error: Unable to create a molecule from the provided SMILES string.")
        return None

    target_fp = rdMolDescriptors.GetMorganFingerprint(target_scaffold, 2)
    warnings.filterwarnings("ignore", category=UserWarning)
    similarities = []

    for i, smiles in enumerate(molecule_smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_fp = rdMolDescriptors.GetMorganFingerprint(scaffold, 2)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(target_fp, scaffold_fp)
            similarities.append((smiles, tanimoto_similarity, label_list[i]))
        except:
            continue

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_5_similar_molecules = similarities[:k]
    return top_5_similar_molecules

def create_bace_prompt(input_selfie, pp_examples):
    prompt = "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\nPlease strictly follow the format, no other information can be provided. Given the SELFIE string of a molecule, predict the molecular properties of a given chemical compound based on its structure, by analyzing whether it can inhibit (Yes) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit (No) BACE1. Consider factors such as molecular weight, atom count, bond types, and functional groups in order to assess the compound's drug-likeness and its potential to serve as an effective therapeutic agent for Alzheimer's disease. Please answer with only Yes or No. A few examples are provided in the beginning.\n"
    for example in pp_examples:
        prompt += f"SELFIE: {example[0]}\nBACE-1 Inhibit: {example[-1]}\n"
    prompt += f"SELFIE: {input_selfie}\nBACE-1 Inhibit:\n"
    return prompt

def main():
    # root = "/content/"
    root = "/home/de575594/Deepan/LLM/Chem/Property/selfies/"
    bace = pd.read_csv(root + "Datasets/BACE.csv")
    sample_size = 1513
    bace_sample = bace.sample(sample_size, replace=True)
    print("The length of df is", len(bace_sample))

    model_engine = ['blip']
    sample_nums = [0,2,4]
    sample_methods = ['random']
    detail_save_folder = '/home/de575594/Deepan/LLM/Chem/Property/blip/clintox/Logs/' # path to save the generated result
    paras = 0
    for sample_method in sample_methods:
            for sample_num in sample_nums:
                with open(root + f'Results/BACE_{sample_num}.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Question', 'url', 'Answer']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for model in model_engine:
                        if paras < 0:
                            paras += 1
                            continue

                        if sample_method == 'random':
                            para_index = 0
                            bace_examples = random_sample_examples(bace, sample_num)
                            for i in tqdm(range(0, len(bace_sample))):
                                if para_index < 0:
                                    para_index += 1
                                    continue

                                if bace_sample.iloc[i]["Class"] is None:
                                    break
                                example = [(bace_sample.iloc[i]["mol"], bace_sample.iloc[i]["Class"])]
                                pred_y = []
                                generated_results = []
                                for text in example:
                                    try:
                                        input_selfie = sf.encoder(text[0])
                                        prompt = create_bace_prompt(input_selfie, bace_examples)
                                        path = f'BACEImages/{i}.jpeg'
                                        m = Chem.MolFromSmiles(text[0])
                                        if m is None:
                                            # print("The smile structure is ", text[0])
                                            break

                                        img = Draw.MolToImage(m)
                                        img.save(path)
                                        writer.writerow({'url': path,
                                                        'Question': prompt,
                                                        'Answer': f"{text[1]}"})
                                    except Exception as e:
                                        print(f"Error processing molecule {text[0]}: {e}")
                                        continue

    
        
if __name__ == '__main__':
    main()
