import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Device-safe setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
@st.cache_resource()
def load_model():
    model_name = "Darren01/opt-6.7b-custom"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
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

# Generate Prompt
def build_prompt(service, format_choice, tone, lang):
    platform = format_choice.split(" ")[0].lower()
    return (
        f"Generate a {tone.lower()} marketing message for {platform}. "
        f"The message should promote: {service.strip()} "
        f"and be written in {lang}."
    )

# Generate Output
if st.button("Generate Marketing Text ✨"):
    if service_description.strip():
        with st.spinner("Generating text..."):
            prompt = build_prompt(service_description, format_option, tone_option, language_option)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model.to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Display Generated Text
        st.subheader("📢 Generated Marketing Content:")
        st.write(generated_text)

        # Download Option
        filename = f"marketing_text_{platform}_{language_option.lower()}.txt"
        st.download_button("💾 Download as Text", generated_text, file_name=filename)
    else:
        st.warning("Please enter a service description!")
