# mPR2_Bench<img src = "assets/mainlogo.png" width = "40" />
This repo contains dataset generation, evaluation code for the paper : "MPr2-Bench: Large Vision Language Models for Molecular Property Prediction"

## Datasets
You can find all the datasets on Hugging Face:

[![Hugging Face](assets/hf-logo.png)](https://huggingface.co/ChemistryVision)
## Setup

To get started, follow these steps:

1. **Clone the repository:**

    ```sh
    git clone https://github.com/ChemistryVisionlanguage/ppvl-bench.git
    cd ppvl-bench
    ```

2. **Create a Conda environment:**

    Ensure you have Conda installed. If not, you can install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

    ```sh
    conda create --name molecule-env python=3.10
    conda activate molecule-env
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Generate Datasets:**

    Navigate to the `GenerateDataset` folder and run the dataset generation scripts. For example:
    Example - To Generate SMILE representation prompt for BACE dataset. 
    ```sh
    cd GenerateDataset/SMILES
    python bace.py 
    ```


    Adjust the above command to match the specific script you need to run for generating your datasets.

2. **Add Your Data:**

    Ensure that your molecule data is in the correct format and place it in the appropriate directory. Follow the instructions within each script in the `GenerateDataset` folder to correctly format and input your data.

## Acknowledgements

The finetuning,ICL method used in this project/paper has been referenced from the following repository:

- [Llava1.5](https://github.com/haotian-liu/LLaVA.git)
- [Llama-AdapterV2](https://github.com/OpenGVLab/LLaMA-Adapter.git)
- [mPlugOWL2](https://github.com/X-PLUG/mPLUG-Owl.git)
- [QwenVL](https://github.com/QwenLM/Qwen-VL.git)
- [CogVLM](https://github.com/THUDM/CogVLM)
- [BLIP](https://huggingface.co/Salesforce/blip-vqa-base)

## Contribution

Feel free to contribute to this project by opening issues or submitting pull requests. 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request
