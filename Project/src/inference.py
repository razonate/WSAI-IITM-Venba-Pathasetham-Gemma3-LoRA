import torch
from unsloth import FastModel

from grapheme_tokenizer import clean_grapheme_split, post_process_grapheme

INSTRUCTION = (
    "கீழ்க்காணும் தமிழ் வெண்பாவை, அதன் சொற்களை சரியான எல்லைகளுடன் "
    "பிரித்து எழுதுக (பதப்பிரிவு). சொற்களின் வரிசை மாறாமல், "
    "ஒட்டிய சொற்களை பிரிக்கவும், தேவையெனில் பிரிந்த எழுத்துக்களை "
    "இணைத்து சரியான சொல்லை உருவாக்கவும். பதப்பிரிவை மட்டும் தரவும்."
)


def load_model(model_name="unsloth/gemma-3-270m-it", max_seq_length=512):
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )
    FastModel.for_inference(model)
    return model, tokenizer


def format_inference_prompt(venba, model, tokenizer):
    venba_g = clean_grapheme_split(venba, tokenizer)
    messages = [{
        "role": "user",
        "content": f"{INSTRUCTION}\n\nவெண்பா: {venba_g}\nபதப்பிரிவு:"
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_pathaseth(venba, model, tokenizer, max_new_tokens=256):
    prompt = format_inference_prompt(venba, model, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return post_process_grapheme(raw)
