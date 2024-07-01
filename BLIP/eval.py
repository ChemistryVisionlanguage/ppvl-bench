from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import BlipProcessor, BlipForQuestionAnswering
import requests
from PIL import Image
import json, os, csv
import logging
from tqdm import tqdm
import torch
import pandas as pd

# Set the path to your test data directory
test_data_dir = "BBBPpromptstest.csv"

name = "BBBPPromptstest"
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model").to("cuda")

results = []
df = pd.read_csv(test_data_dir)
print("Predicting Results")
for i in range(3):
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        question = row['Question']
        image_path = row['url']
        answer = row['Answer']
        # Read the image from the local file path
        image = Image.open(image_path).convert("RGB")

        # prepare inputs
        encoding = processor(image, question, return_tensors="pt").to("cuda:0", torch.float16)

        out = model.generate(**encoding)
        generated_text = processor.decode(out[0], skip_special_tokens=True)

        results.append((question, generated_text,answer))
    # Write the results to a CSV file
    csv_file_path = f"Results/results_new_{i}.csv"
    with open(csv_file_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["ID", "Prediction","Answer"])  # Write header
        csv_writer.writerows(results)

    print(f"Results saved to {csv_file_path}")