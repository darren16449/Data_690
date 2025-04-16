# ✅ First Streamlit command — must be at the top
import streamlit as st
st.set_page_config(page_title="Mistral RAG Web App", layout="centered")

# 🔧 Imports AFTER set_page_config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 🧠 Model Loading
@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model():
    model_id = "Darren01/mistral-rag-model"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    return tokenizer, model

# 📦 Load model + set device
tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔄 Inference Function
def generate_text(prompt: str, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🎨 Streamlit UI
st.title("🧠 Mistral RAG Web App")
st.markdown("Generate responses using your fine-tuned model.")

user_input = st.text_area("Enter your prompt:", height=150)

if st.button("Generate"):
    with st.spinner("Thinking..."):
        try:
            output = generate_text(user_input)
            st.markdown("### ✨ Response")
            st.write(output)
        except Exception as e:
            st.error(f"❌ Error: {e}")
