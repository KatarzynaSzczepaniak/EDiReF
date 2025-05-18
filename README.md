# Emotion Discovery and Reasoning its Flip in Conversation (EDiReF)

This repository contains the codebase developed as part of my Master's thesis. It addresses the task of **Emotion Discovery and Reasoning its Flip in Conversation (EDiReF)**, introduced in [SemEval 2024](https://semeval.github.io/SemEval2024/). The goal is to detect emotions in dialogue and reason about how they change across conversational turns.

The solution is based on the **Mixture of Experts (MoE)** technique, incorporating **transformer-based models** for emotion classification and reasoning.

📄 A record of the thesis can be found [here](https://repo.pw.edu.pl/info/master/WUT7d61093cf864411ebc750603bb42cee5?r=supervisedwork&ps=20&tab=&title=Prace%2Bmagisterskie%2B%25E2%2580%2593%2BZastosowanie%2Bmieszaniny%2Bekspert%25C3%25B3w%2Bdo%2Brozpoznawania%2Bemocji%2Bi%2Bwypowiedzi%2Bpowoduj%25C4%2585cych%2Bzmian%25C4%2599%2Bemocji%2Bw%2Bkonwersacji%2B%25E2%2580%2593%2BPolitechnika%2BWarszawska&lang=pl).
*Note: Full document access may be restricted at this time.*

🔧 Example usage:
```bash
./run_experiments.sh --datasets MELD,MaSaC --stage 1 --train_bert True
```

📁 Project structure:
```
experiment_configs/   # YAML files defining experimental setups 
notebooks/            # Jupyter notebooks for interactive training
scripts/              # Python scripts for training and evaluation
results/              # Final performance results (cleaned)
run_experiments.sh    # Main entry point for reproducing experiments
```

⚠️ **Note**: This is functional research code. While the code is well-documented and structured for readability, it is maintained as monolithic notebooks and scripts by design. Modularization is not planned.