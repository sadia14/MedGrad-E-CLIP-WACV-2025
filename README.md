# MedGrad-ECLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis

This repository contains the code accompanying our paper:

> **“MedGrad E-CLIP: Enhancing Trust and Transparency in AI-Driven Skin Lesion Diagnosis”**  
> Sadia Kamal & Tim Oates, University of Maryland Baltimore County  
> [arXiv:2501.06887](https://arxiv.org/abs/2501.06887)

MedGrad-ECLIP fine-tunes OpenAI’s CLIP (ViT-B/16) on dermoscopic skin lesion images paired with structured diagnostic descriptions and introduces a novel explainability method, **MedGrad-ECLIP**, which incorporates entropy weighting to highlight subtle but diagnostically important regions.

---

## Repository Contents

- **`training_clip_medgrad_eclip.py`** – Training script. Fine-tunes CLIP on a custom image–text dataset with augmentation and logs metrics (accuracy, precision, recall, F1, sensitivity, specificity, CLIP score).
- **`grad_eclip_newtrained.py`** – Inference and visualization script. Loads a trained CLIP model and produces Grad-ECLIP, MedGrad-ECLIP and Grad-CAM heatmaps for a given image and set of text prompts.
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

### 1) Train CLIP on your dataset

Prepare:
- a folder of dermoscopic images (e.g., `images/`)
- a CSV file of descriptions with a header row and one description per line (e.g., `descriptions.csv`)

Minimal CSV example:
```csv
description
melanoma, irregular border, multiple colors
atypical nevus, symmetry in 1 axis, light brown
```

Run:
```bash
python training_clip_medgrad_eclip.py   --image_folder /path/to/images   --description_file /path/to/descriptions.csv   --num_augmentations 3   --epochs 30   --batch_size 64   --save_path trainedmodel_B16.pth
```

This trains CLIP and saves the fine-tuned weights to `trainedmodel_B16.pth`.

---

### 2) Generate MedGrad-ECLIP heatmaps

Prepare:
- a trained model checkpoint (e.g., `trainedmodel_B16.pth`)
- a single image path (e.g., `examples/sample_image.bmp`)
- a text file with one prompt per line (e.g., `examples/prompts.txt`)

Minimal prompts example:
```text
melanoma
atypical pigment network
present streaks
light brown color
```

Run:
```bash
python grad_eclip_newtrained.py   --model_path trainedmodel_B16.pth   --image_path examples/sample_image.bmp   --texts_file examples/prompts.txt
```

The script outputs Grad-ECLIP, MedGrad-ECLIP, and Grad-CAM visualizations for each prompt.

> Note: On Windows PowerShell, either put the command on one line or replace the `\` line continuations appropriately.

---

## Datasets

Our experiments used:

- **PH²**: ~200 dermoscopic images with histological diagnoses and dermoscopic structure criteria.
- **Derm7pt**: >2000 dermoscopic images with structured metadata (7-point checklist).

You can use your own image–text datasets by supplying the correct paths to `--image_folder` and `--description_file`.

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
