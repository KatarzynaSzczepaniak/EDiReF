# Emotion Discovery and Reasoning its Flip in Conversation (EDiReF)

This repository contains the codebase developed as part of my Master's thesis. It addresses the task of **Emotion Discovery and Reasoning its Flip in Conversation (EDiReF)**, introduced in [SemEval 2024](https://semeval.github.io/SemEval2024/). The goal is to detect emotions in dialogue and reason about how they change across conversational turns.

The solution is based on the **Mixture of Experts (MoE)** technique, incorporating **transformer-based models** for emotion classification and reasoning.

📄 A record of the thesis can be found [here](https://repo.pw.edu.pl/info/master/WUT7d61093cf864411ebc750603bb42cee5?r=supervisedwork&ps=20&tab=&title=Prace%2Bmagisterskie%2B%25E2%2580%2593%2BZastosowanie%2Bmieszaniny%2Bekspert%25C3%25B3w%2Bdo%2Brozpoznawania%2Bemocji%2Bi%2Bwypowiedzi%2Bpowoduj%25C4%2585cych%2Bzmian%25C4%2599%2Bemocji%2Bw%2Bkonwersacji%2B%25E2%2580%2593%2BPolitechnika%2BWarszawska&lang=pl).
*Note: Full document access may be restricted at this time.*

## 📁 Project structure:
```text
core/                   # Contains utilities for data processing, training, evaluation, and experiment configuration
data/                   # Folder for the dataset
experiment_configs/     # YAML files defining experimental setups 
notebooks/              # Jupyter notebooks for interactive training
results/                # Final performance results (cleaned)
scripts/                # Python scripts for training and evaluation
run_experiments.sh      # Main entry point for reproducing Mixture of Experts (MoE) experiments
```

## 🛠️ Environment Setup (with venv):

From the `EDiReF/` root directory:

**1. Create a virtual environment**
```bash
python3 -m venv venv
```
**2. Activate the virtual environment**

&nbsp;&nbsp;&nbsp;&nbsp;On Linux/macOS:
```bash
source venv/bin/activate
```
&nbsp;&nbsp;&nbsp;&nbsp;On Windows:
```bash
source venv/Scripts/activate
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Configure PYTHONPATH so Python can locate the core/ module**
```bash
export PYTHONPATH=$(pwd)
```

**5. Deactivate the virtual environment when done**
```bash
deactivate
```

## 🔧 Example usage:

To run Mixture of Experts (MoE) experiments:
```bash
./run_experiments.sh --datasets MELD,MaSaC --stage 1 --train_bert True
```

To run reference model experiments (baseline):
```bash
python3 scripts/reference_model.py MELD experiment_configs/reference_model_experiments.yaml
```