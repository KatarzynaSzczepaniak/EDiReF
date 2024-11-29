import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16, attn_implementation="flash_attention_2").to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

article = "Şeful ONU spune că nu există o soluţie militară în Siria"
inputs = tokenizer(article, return_tensors="pt").to("cuda")

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("deu_Latn"), max_length=30
)

output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print(output)