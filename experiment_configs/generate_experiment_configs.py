import yaml

experiments = {}
experiment_count = 1

for num_epochs in [3, 5, 7]:
    for lr in [2e-5, 3e-5, 4e-5, 5e-5]:
        experiments[experiment_count] = {'LEARNING_RATE': lr,
                                            'NUM_EPOCHS': num_epochs,
        }
        experiment_count += 1

yaml_file_path = 'experiment_configs/reference_model_experiments.yaml'

with open(yaml_file_path, 'w') as file:
    yaml.dump(experiments, file, default_flow_style=False)