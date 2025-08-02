
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Language codes as per IndicTrans2
lang_options = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva"
}

# Load model and tokenizer once to avoid repeated loading
model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

def translate(input_text, src_lang, tgt_lang):
    if not input_text.strip():
        return "Please enter some text to translate."

    # Construct input with language tags as expected by IndicTrans2
    tagged_text = f"{input_text}"
    model_inputs = tokenizer(
        tagged_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    with torch.no_grad():
        output_tokens = model.generate(**model_inputs, max_length=512)
    output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return output_text

with gr.Blocks() as demo:
    gr.Markdown("# 🌍 Multilingual Translator (IndicTrans2 + OpusMT)")
    with gr.Row():
        src = gr.Dropdown(choices=list(lang_options.keys()), label="Source Language", value="English")
        tgt = gr.Dropdown(choices=list(lang_options.keys()), label="Target Language", value="Hindi")
    text_input = gr.Textbox(lines=5, placeholder="Enter text here...", label="Input")
    translate_btn = gr.Button("Translate")
    output = gr.Textbox(label="Translation Output")

    translate_btn.click(
        fn=lambda text, src, tgt: translate(text, lang_options[src], lang_options[tgt]),
        inputs=[text_input, src, tgt],
        outputs=output
    )

demo.launch()
