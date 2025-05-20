# Notebooks

This directory contains research notebooks for training and evaluating Mixture-of-Experts (MoE) models on the MELD and MaSaC datasets for emotion and trigger classification using the EDiReF dataset.

⚠️ **Note**: These notebooks are intended to be run on **Google Colab**.

## 📁 Directory layout:
To run the notebooks successfully, ensure the following structure in your Google Drive:

```
├── MyDrive/
│   └── Colab Notebooks/
│       └── EDiReF/
│           ├── data/
│           │   ├── EDiReF_train_data/
│           │   │   ├── MaSaC_train_efr.json
│           │   │   └── MELD_train_efr.json
│           │   └── EDiReF_val_data/
│           │       ├── MaSaC_val_efr.json
│           │       └── MELD_val_efr.json
│           ├── notebooks/
│           │       ├── mixture_of_experts_dual_gate.ipynb
│           │       ├── mixture_of_experts_single_gate.ipynb
│           │       └── mixture_of_experts_reference_model.ipynb
└── ...
```

## 📝 Notes:
- The notebooks automatically mount your Google Drive using `drive.mount()`.
- JSON files must match the naming conventions shown above.
- Download the required JSON files from the [EDiReF-SemEval2024 GitHub repository](https://github.com/LCS2-IIITD/EDiReF-SemEval2024) and place them manually in the folders above.