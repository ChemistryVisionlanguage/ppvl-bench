import warnings
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os
import csv
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
import selfies as sf

# warnings.filterwarnings("ignore", category="UserWarning", module=".*rdkit.*")

def random_sample_examples(bace, sample_size):
    positive_examples = bace[bace["FDA_APPROVED"].str.lower() == "yes"].sample(int(sample_size / 2))
    negative_examples = bace[bace["FDA_APPROVED"].str.lower() == "no"].sample(int(sample_size / 2))
    smiles = positive_examples["smiles"].tolist() + negative_examples["smiles"].tolist()

    selfies_pos = [sf.encoder(s) for s in positive_examples["smiles"].tolist()]
    selfies_neg = [sf.encoder(s) for s in negative_examples["smiles"].tolist()]

    class_label = positive_examples["FDA_APPROVED"].tolist() + negative_examples["FDA_APPROVED"].tolist()
    class_label = ["Yes" if i.lower() == "yes" else "No" for i in class_label]

    bace_examples = list(zip(selfies_pos + selfies_neg, class_label))
    return bace_examples

def top_k_scaffold_similar_molecules(target_smiles, bace_data, k):
    bace_data = bace_data[bace_data["smiles"] != target_smiles]
    molecule_smiles_list = bace_data["smiles"].tolist()
    label_list = bace_data["FDA_APPROVED"].tolist()
    label_list = ["Yes" if i.lower() == "yes" else "No" for i in label_list]

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
    top_k_similar_molecules = similarities[:k]
    return top_k_similar_molecules

def create_clintox_prompt(input_selfie, pp_examples):
    prompt = '''You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n
Please strictly follow the format, no other information can be provided. Given the SELFIE string of a molecule, the task focuses on predicting molecular properties, specifically whether a molecule is Clinically-trail-Toxic (Yes) or Not Clinically-trail-toxic (No) based on the SELFIE string representation of each molecule. The FDA-approved status will specify if the drug is approved by the FDA for clinical trials (Yes) or Not approved by the FDA for clinical trials (No).\n
You will be provided with task template. The task is to predict the binary label for a given molecule, please answer with only Yes or No.\n'''
    for example in pp_examples:
        prompt += f"SELFIE: {example[0]}\nToxic: {example[1]}\n"
    prompt += f'''\nBelow is the molecule whose property you have to predict. Along with is the image structure of the molecule.\nSELFIE: {input_selfie}\nToxic:\n You have to predict whether it is Toxic with answer Yes or No.\n'''
    return prompt

def main():
    root = "/home/de575594/LLM/Chem/Property/selfies/"
    bace = pd.read_csv(root + "Datasets/ClinTox.csv")
    sample_size = 1484
    bace_sample = bace.sample(sample_size)
    print("The length of df is", len(bace_sample))
    bace_sample['FDA_APPROVED'] = bace_sample['FDA_APPROVED'].apply(lambda x: "Yes" if x ==1 else "No")
    model_engine = ['blip']
    sample_nums = [2]
    sample_methods = ['random']
    detail_save_folder = '/home/de575594/LLM/Chem/Property/blip/clintox/Logs/'
    for sample_method in sample_methods:
            for sample_num in sample_nums:
                with open(root + f'Results/Clintoxprompts_{sample_num}.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Question', 'url', 'Answer']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for model in model_engine:
                        if sample_method == 'random':
                            bace_examples = random_sample_examples(bace, sample_num)
                            for i in tqdm(range(len(bace_sample))):
                                example = [(bace_sample.iloc[i]["smiles"], bace_sample.iloc[i]["FDA_APPROVED"])]
                                for text in example:
                                    try:
                                        input_selfie = sf.encoder(text[0])
                                        prompt = create_clintox_prompt(input_selfie, bace_examples)
                                        path = f'ClintoxImages/{i}.jpeg'
                                        m = Chem.MolFromSmiles(text[0])
                                        if m is None:
                                            print("The SMILES structure is", text[0])
                                            break

                                        img = Draw.MolToImage(m)
                                        img.save(path)
                                        writer.writerow({'url': path, 'Question': prompt, 'Answer': text[1]})
                                    except Exception as e:
                                        print(f"Error processing molecule {text[0]}: {e}")
                                        continue
    
        

if __name__ == '__main__':
    main()
