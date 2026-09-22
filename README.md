# Resource-Constrained CNN for Edge Image Classification 

This project implements and compares lightweight convolutional neural networks for image classification under strict resource constraints (≤100,000 trainable parameters), designed to emulate deployment on embedded/edge devices such as microcontrollers or Raspberry Pi.

## Dataset

**RealWaste** — real photographs of waste items captured at the point of reception in a landfill environment, released at 524×524 resolution. Images are downscaled to a maximum of 64×64 pixels to emulate a low-resource sensor input. The dataset contains 9 classes (Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation) totaling 4,752 images, with moderate class imbalance (~3:1 max ratio) handled via stratified splitting and class-weighted loss.

[Dataset link](https://archive.ics.uci.edu/dataset/908/realwaste)

## Models

- **Model A (Standard CNN):** Standard 2D convolutional layers interleaved with max-pooling.
- **Model B (Lightweight CNN):** Depthwise separable convolutions in place of standard convolutions, constrained to ≤100,000 trainable parameters.

Both models are also compared against two fine-tuned pre-trained lightweight architectures designed for mobile/edge deployment.

## Repository Structure

```
├── data/              # raw and processed dataset (gitignored)
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

## Status

🚧 In progress


## Author

False Negatives