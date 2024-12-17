import yaml

experiments = {}
experiment_count = 1
GATE_TYPE = 'linear'
EXPERT_TYPE = 'linear'

for top_k in [2, 8]:
    for num_epochs in [3, 5, 7]:
        for lr in [2e-5, 3e-5, 4e-5, 5e-5]:
            experiments[experiment_count] = {'GATE_TYPE': GATE_TYPE,
                                             'EXPERT_TYPE': EXPERT_TYPE,
                                             'NUM_EXPERTS': 8,
                                             'TOP_K': top_k,
                                             'LEARNING_RATE': lr,
                                             'NUM_EPOCHS': num_epochs,
            }
            experiment_count += 1

yaml_file_path = 'experiment_configs/stage_3_masac_experiments_dual_gate.yaml'

with open(yaml_file_path, 'w') as file:
    yaml.dump(experiments, file, default_flow_style=False)