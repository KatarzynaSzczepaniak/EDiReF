#!/bin/bash

usage() {
    echo "Usage: $0 --datasets <dataset1,dataset2,..> --stage <1/2/3> --train_bert <True/False>"
    echo "Example: $0 --datasets MELD,MaSaC --stage 1 --train_bert True"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            IFS=',' read -r -a DATASETS <<< "$2"  # Split datasets by comma
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --train_bert)
            TRAIN_BERT="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "${DATASETS[*]}" || -z "$STAGE" ]]; then
    usage
fi

if [[ "$TRAIN_BERT" != "True" && "$TRAIN_BERT" != "False" ]]; then
    echo "Error: --train_bert must be 'True' or 'False'"
    usage
fi

STAGE_NUM=${STAGE#stage_}
GATE_TYPES=("single" "dual")

for dataset in "${DATASETS[@]}"; do
    for gate_type in "${GATE_TYPES[@]}"; do
        SCRIPT="moe_${gate_type}_gate.py"

        if [[ "$STAGE" == 1 ]]; then
            CONFIG_FILE="stage_1_experiments.yaml"
        elif [[ "$STAGE" == 2 || "$STAGE" == 3 ]]; then
            if [[ "$dataset" == "MELD" ]]; then
                BASE_FILE="meld_experiments"
            elif [[ "$dataset" == "MaSaC" || "$dataset" == "MaSaC_translated" ]]; then
                BASE_FILE="masac_experiments"
            fi

            if [[ "$gate_type" == "single" ]]; then
                GATE_PART="single_gate"
            elif [[ "$gate_type" == "dual" ]]; then
                GATE_PART="dual_gate"
            fi
            CONFIG_FILE="stage_${STAGE_NUM}_${BASE_FILE}_${GATE_PART}.yaml"
        fi

        if [[ "$dataset" == "MELD" ]]; then
            WEIGHT_TRIGGER_FLAG=""
        else
            WEIGHT_TRIGGER_FLAG="--weight_triggers"
        fi

        if [[ "$TRAIN_BERT" == "True" ]]; then
            TRAIN_BERT_FLAG="--train_bert"
        else
            TRAIN_BERT_FLAG=""
        fi

        echo "Running scripts/$SCRIPT $dataset experiment_configs/$CONFIG_FILE $WEIGHT_TRIGGER_FLAG $TRAIN_BERT_FLAG..."
        python3 "scripts/$SCRIPT" "$dataset" "experiment_configs/$CONFIG_FILE" $WEIGHT_TRIGGER_FLAG $TRAIN_BERT_FLAG
    done
done
