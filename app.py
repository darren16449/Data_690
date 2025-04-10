import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
@st.cache_resource()
def load_model():
    model_name = "Darren01/opt-6.7b-custom"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

# UI Header
st.title("🚀 AI-Powered Marketing Text Generator")
st.markdown("Generate high-quality marketing content tailored for different platforms!")

# User Inputs
service_description = st.text_area("Enter a brief description of your service:")

format_option = st.selectbox("Select Content Format:", [
    "LinkedIn (Formal, Professional)", 
    "Email (Formal, Persuasive)", 
    "Instagram (Casual, Engaging, Hashtags)"
])

tone_option = st.selectbox("Select Tone:", ["Formal", "Casual", "Slang/Conversational"])

language_option = st.selectbox("Select Language:", ["English", "Spanish", "French", "German"])

temperature = st.slider("Creativity Level (Temperature)", 0.1, 1.0, 0.7)
max_length = st.slider("Max Length of Text", 50, 300, 150)

if st.button("Generate Marketing Text ✨"):
    if service_description.strip():
        with st.spinner("Generating text..."):
            prompt = f"Generate a {tone_option.lower()} marketing message for {format_option.split(' ')[0].lower()} about: {service_description}. Output in {language_option}."
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_length=max_length, temperature=temperature)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Display Generated Text
        st.subheader("📢 Generated Marketing Content:")
        st.write(generated_text)
        
        # Copy and Download Options
        st.button("📋 Copy to Clipboard")
        st.download_button("💾 Download as Text", generated_text, file_name="marketing_text.txt")
    else:
        st.warning("Please enter a service description!")
