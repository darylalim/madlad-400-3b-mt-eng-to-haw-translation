import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.bettertransformer import BetterTransformer
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained(
    "google/madlad400-3b-mt",
    use_fast=True
)

model_hf = AutoModelForSeq2SeqLM.from_pretrained(
    "google/madlad400-3b-mt",
    torch_dtype=torch.bfloat16
)

model = BetterTransformer.transform(model_hf, keep_original=True)

def translate(text):
    """
    Translates the input text from English to Hawaiian.
    """
    text = "<2haw> " + text
    
    inputs = tokenizer(
        text,
        return_tensors="pt"
    )
    
    outputs = model.generate(**inputs, max_new_tokens=1000)
    text_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return text_translated[0]

demo = gr.Interface(
    fn=translate,
    inputs=[gr.Textbox(label="English")],
    outputs=[gr.Textbox(label="Hawaiian")],
    title="MADLAD-400-3B-MT English-to-Hawaiian Translation",
    description="[Code](https://github.com/darylalim/madlad-400-3b-mt-eng-to-haw-translation)")

demo.queue()

demo.launch()