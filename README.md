# MedGrad-ECLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis

This repository contains the notebooks and resources accompanying our paper:

> **“MedGrad E-CLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis”**  
> Sadia Kamal & Tim Oates, University of Maryland Baltimore County  
> [arXiv:2501.06887](https://arxiv.org/abs/2501.06887)

MedGrad-ECLIP fine-tunes OpenAI’s CLIP (ViT-B/16) on dermoscopic skin lesion images paired with structured diagnostic descriptions and introduces a novel explainability method, **MedGrad-ECLIP**, which incorporates entropy weighting to highlight subtle but diagnostically important regions.

---

## Repository Contents

- **`notebooks/Training_clip_MedGrad_ECLIP.ipynb`** – Training notebook. Fine-tunes CLIP on a custom image–text dataset with augmentation and logs metrics (accuracy, precision, recall, F1, sensitivity, specificity, CLIP score).
- **`notebooks/MedGrad_ECLIP_Inference_Code.ipynb`** – Inference and visualization notebook. Loads a trained CLIP model and produces Grad-ECLIP, MedGrad-ECLIP and Grad-CAM heatmaps for a given image and set of text prompts.
- **`requirements.txt`** – Python dependencies.
- **`paper/MedGrad_ECLIP.pdf`** – The paper.

Optional:
- `examples/sample_image.bmp` – Example dermoscopic image.
- `examples/prompts.txt` – Example text prompts (diagnostic criteria).

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<yourusername>/MedGrad-ECLIP.git
cd MedGrad-ECLIP
pip install -r requirements.txt
```

We recommend Python ≥3.9 and a GPU for training (CUDA).

---

## Usage

### 1) Train CLIP on your dataset (Notebook)

Open the training notebook:

- In Jupyter: `jupyter notebook notebooks/Training_clip_MedGrad_ECLIP.ipynb`
- In Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<yourusername>/MedGrad-ECLIP/blob/main/notebooks/Training_clip_MedGrad_ECLIP.ipynb)

Prepare:
- a folder of dermoscopic images (e.g., `images/`)
- a CSV file of descriptions with a header row and one description per line (e.g., `descriptions.csv`)

Run the notebook cells to fine-tune CLIP on your dataset.  
The notebook saves the fine-tuned weights to a `.pth` file (e.g., `trainedmodel_B16.pth`).

---

### 2) Generate MedGrad-ECLIP heatmaps (Notebook)

Open the inference notebook:

- In Jupyter: `jupyter notebook notebooks/MedGrad_ECLIP_Inference_Code.ipynb`
- In Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<yourusername>/MedGrad-ECLIP/blob/main/notebooks/MedGrad_ECLIP_Inference_Code.ipynb)

Prepare:
- the trained model checkpoint (e.g., `trainedmodel_B16.pth`)
- a single image path (e.g., `examples/sample_image.bmp`)
- a text file with one prompt per line (e.g., `examples/prompts.txt`)

Run the notebook cells to generate Grad-ECLIP, MedGrad-ECLIP and Grad-CAM visualizations for each prompt, replicating the figures in the paper.

---

## Datasets

Our experiments used:

- **PH²**: ~200 dermoscopic images with histological diagnoses and dermoscopic structure criteria.
- **Derm7pt**: >2000 dermoscopic images with structured metadata (7-point checklist).

You can adapt the notebooks to your own image–text datasets by supplying the appropriate paths in the notebook cells.

---

## Citation

If you use this code, please cite:

```bibtex
@article{kamal2025medgrad,
  title={MedGrad E-CLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis},
  author={Kamal, Sadia and Oates, Tim},
  journal={arXiv preprint arXiv:2501.06887},
  year={2025}
}
```

---

## License

Specify your license (MIT, Apache-2.0, etc.) here.

---

## Contact

For questions about this code, please open an issue on GitHub or email sadia1402@umbc.edu.
