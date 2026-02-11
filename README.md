# Deep Learning for Biomedicine: Predicting Antibiotic Resistance in Staphylococcus aureus

This repository contains a comprehensive project on applying deep learning techniques to predict antibiotic resistance in *Staphylococcus aureus* bacteria, specifically focusing on resistance to Cefoxitin mediated by the *pbp4* gene. The project implements and compares two neural network architectures: a fine-tuned DNABERT-2 Transformer model and a custom 1D Convolutional Neural Network (CNN).

## Project Overview

Antibiotic resistance poses a significant global health challenge. This project explores the use of genomic sequence data to predict bacterial resistance, leveraging advanced deep learning models trained on DNA sequences. The work demonstrates:

- Data preprocessing and sequence fragmentation techniques
- Fine-tuning of pre-trained genomic language models (DNABERT-2)
- Implementation of a CNN baseline for comparison
- Gene-level evaluation using max pooling aggregation
- Comparative analysis of Transformer vs. CNN architectures

## Repository Structure

```
├── Code/
│   └── DNABERT2.ipynb              # Main Jupyter notebook with complete implementation
├── Presentation/
│   └── presentation_draft_1.pptx   # Presentation slides
├── WrittenPart/
│   ├── SeminarWriteup.tex          # LaTeX source for seminar writeup
│   ├── SeminarWriteup.pdf          # Compiled seminar writeup
│   ├── refs.bib                    # Bibliography file
│   └── lit/                        # Literature references
│       └── *.pdf                   # Research papers and references
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Git

### Dependencies

Install the required packages using pip:

```bash
pip install transformers datasets biopython peft torch scikit-learn matplotlib seaborn numpy pandas
```

For GPU acceleration, ensure PyTorch is installed with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Setup

The notebook automatically downloads the required dataset from the course repository. No manual data setup is required.

## Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "Deep Learning for Biomedicine/Git repo"
   ```

2. **Run on Google Colab:**
   - Upload the `Code/DNABERT2.ipynb` notebook to [Google Colab](https://colab.research.google.com/)
   - Ensure GPU runtime is enabled (Runtime > Change runtime type > Hardware accelerator > GPU)
   - Run the cells sequentially

3. **Run the cells in order:**
   - The notebook is structured to run sequentially
   - GPU acceleration is automatically detected and utilized if available
   - Training may take several hours depending on hardware

### Key Components

- **Data Loading:** Loads *Staphylococcus aureus* Cefoxitin resistance dataset
- **Preprocessing:** Sequence cleaning, sliding window fragmentation
- **DNABERT-2 Training:** Fine-tunes pre-trained genomic Transformer
- **CNN Baseline:** Trains custom 1D convolutional network
- **Evaluation:** Gene-level aggregation using max pooling
- **Comparison:** Performance metrics and visualizations

## Key Results

The project evaluates both models on a held-out test set of 99 genes:

### Performance Metrics (Test Set)
- **DNABERT-2 Transformer:** Advanced genomic language model with strong sequence understanding
- **CNN Baseline:** Lightweight architecture focusing on local motifs

Key evaluation metrics include:
- Accuracy
- Precision and Recall
- F1-Score
- ROC AUC (primary metric for imbalanced classification)

### Technical Highlights

- **Sequence Fragmentation:** Handles variable-length genes using sliding windows
- **Class Imbalance Handling:** Weighted loss functions for rare resistance cases
- **Gene-Level Aggregation:** Max pooling to combine fragment predictions
- **Comparative Analysis:** Direct comparison of Transformer vs. CNN approaches

## Methodology

1. **Data Preparation:** Stratified gene-level train/validation split to prevent leakage
2. **Sequence Processing:** Sliding window approach (510bp windows, 250bp stride)
3. **Model Training:** Weighted cross-entropy loss, early stopping, mixed precision
4. **Evaluation:** Fragment-level training, gene-level testing with aggregation

## References

The `WrittenPart/lit/` directory contains relevant research papers and references, including:

- Genomic language models and transformers
- Antibiotic resistance mechanisms
- Deep learning applications in bioinformatics
- Staphylococcus aureus penicillin-binding proteins

See `WrittenPart/refs.bib` for the complete bibliography.

## Contributing

This is an academic project for the "Deep Learning for Biomedicine" seminar. For questions or suggestions, please refer to the seminar writeup in `WrittenPart/SeminarWriteup.pdf`.

## License

Academic use only. Please cite appropriately if using this work.
