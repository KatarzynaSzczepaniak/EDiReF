from pathlib import Path
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pyhinavrophonetic import hinavro

def translate_conversation(conv):
    translated_text = []
    for s in conv:
        text = hinavro.parse(s)
        inputs = tokenizer(text, return_tensors='pt')
        outputs = model.generate(**inputs, forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn"), max_length = 30)
        translated_text.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return translated_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Translate MaSaC dataset into English.')
    parser.add_argument('stage', type=str, help = 'Select dataset subtype. Must be train, val or test.')
    args = parser.parse_args()
    stage = args.stage

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    df = pd.read_json(f'data/EDiReF_{stage}_data/MaSaC_{stage}_efr.json')
    df.loc[:, 'utterances'] = df['utterances'].applly(translate_conversation)

    Path(f'data/EDiReF_{stage}_data/').mkdir(parents=True, exist_ok=True)
    df.to_json(f'data/EDiReF_{stage}_data/MaSaC_translated_{stage}_efr.json')