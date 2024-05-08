import spaces
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_3b_mt = AutoTokenizer.from_pretrained("google/madlad400-3b-mt", use_fast=True)
language_codes = [token for token in tokenizer_3b_mt.get_vocab().keys() if token.startswith("<2")]
remove_codes = ['<2>', '<2en_xx_simple>', '<2translate>', '<2back_translated>', '<2zxx_xx_dtynoise>', '<2transliterate>']
language_codes = [token for token in language_codes if token not in remove_codes]

model_choices = [
    "google/madlad400-3b-mt", 
    "google/madlad400-7b-mt", 
    "google/madlad400-10b-mt", 
    "google/madlad400-7b-mt-bt"
]

model_resources = {}

def load_tokenizer_model(model_name):
    """
    Load tokenizer and model for a chosen model name.
    """
    if model_name not in model_resources:
        # Load tokenizer and model for first time
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.to_bettertransformer()
        model.to(device)
        model_resources[model_name] = (tokenizer, model)
    return model_resources[model_name]

@spaces.GPU
def translate(text, target_language, model_name):
    """
    Translate the input text from English to another language.
    """
    # Load tokenizer and model if not already loaded
    tokenizer, model = load_tokenizer_model(model_name)
    
    text = target_language + text
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    outputs = model.generate(input_ids=input_ids, max_new_tokens=128000)
    text_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return text_translated[0]

title = "MADLAD-400 Translation"
description = """
Translation from English to over 400 languages based on [research](https://arxiv.org/pdf/2309.04662) by Google DeepMind and Google Research. Initial inference will be slow as models load.
"""

input_text = gr.Textbox(
    label="Text",
    placeholder="Enter text here"
)
target_language = gr.Dropdown(
    choices=language_codes,
    value="<2haw>",
    label="Target language"
)
model_choice = gr.Dropdown(
    choices=model_choices, 
    value="google/madlad400-3b-mt", 
    label="Model"
)
output_text = gr.Textbox(label="Translation")

demo = gr.Interface(
    fn=translate,
    inputs=[input_text, target_language, model_choice],
    outputs=output_text,
    title=title,
    description=description
)

demo.queue()

demo.launch()