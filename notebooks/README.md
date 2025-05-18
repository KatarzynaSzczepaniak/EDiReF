# Notebooks

This directory contains research notebooks for training and evaluating Mixture-of-Experts (MoE) models for emotion and trigger classification using the EDiReF dataset.

⚠️ **Note**: These notebooks are intended to be run on **Google Colab**.

📁 Directory layout:
The minimal expected directory structure inside your Google Drive:

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
│           ├── mixture_of_experts_dual_gate.ipynb
│           ├── mixture_of_experts_single_gate.ipynb
│           └── mixture_of_experts_reference_model.ipynb
└── ...
```

📝 Notes:
- The notebooks automatically mount your Google Drive using `drive.mount()`.
- JSON files must match the naming conventions shown above.
- Data should be downloaded from the [official EDiReF GitHub repo](https://github.com/LCS2-IIITD/EDiReF-SemEval2024) and manually placed in the appropriate folders.