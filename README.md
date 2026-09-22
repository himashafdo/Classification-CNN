# Resource-Constrained CNN for Edge Image Classification

This project implements and compares lightweight convolutional neural networks for image classification under strict resource constraints (≤100,000 trainable parameters), designed to emulate deployment on embedded/edge devices such as microcontrollers or Raspberry Pi.

## Dataset

**RealWaste** — real photographs of waste items captured at the point of reception at the Whyte's Gully Waste and Resource Recovery facility in Wollongong, NSW, Australia, released at 524×524 resolution. Images are downscaled to a maximum of 64×64 pixels to emulate a low-resource sensor input.

The dataset contains 9 classes totaling 4,752 images:

| Class               | Count |
| ------------------- | ----- |
| Cardboard           | 461   |
| Food Organics       | 411   |
| Glass               | 420   |
| Metal               | 790   |
| Miscellaneous Trash | 495   |
| Paper               | 500   |
| Plastic             | 921   |
| Textile Trash       | 318   |
| Vegetation          | 436   |

There is moderate class imbalance (~3:1 max ratio), handled via stratified splitting and class-weighted loss.

[Dataset link](https://archive.ics.uci.edu/dataset/908/realwaste)

**License:** CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike). This project and any derived outputs follow the same non-commercial, share-alike terms.

**Citation:**

> Single, S.; Iranmanesh, S.; Raad, R. RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. _Information_ 2023, 14, 633. https://doi.org/10.3390/info14120633

## Models

- **Model A (Standard CNN):** Standard 2D convolutional layers interleaved with max-pooling.
- **Model B (Lightweight CNN):** Depthwise separable convolutions in place of standard convolutions, constrained to ≤100,000 trainable parameters.

Both models are also compared against two fine-tuned pre-trained lightweight architectures designed for mobile/edge deployment.

## Environment Setup

Requires **Python 3.11 or 3.12** (TensorFlow does not yet support 3.13+ at time of writing).

```bash
# Create and activate a virtual environment
py -3.11 -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

Key dependencies: TensorFlow 2.21, NumPy, Pandas, scikit-learn, Matplotlib, Pillow. See `requirements.txt` for the full pinned list.

## Repository Structure

```
├── data/              # raw, resized_raw and processed manifest (gitignored)
├── notebooks/         # exploratory analysis
├── src/
│   ├── data_prep.py   # resizing, stratified split, manifest generation
│   ├── model_a.py      # standard CNN architecture
│   ├── model_b.py      # depthwise separable CNN architecture
│   ├── train.py        # training loop
│   └── evaluate.py     # metrics, confusion matrix, comparison tables
├── results/            # plots, metrics, confusion matrices
├── requirements.txt
└── README.md
```

If you are training from scratch add the downloaded dataset to data/ folder and run data_prep.py inside the venv.

## Status

🚧 In progress

## Author

False Negatives -- ENTC
