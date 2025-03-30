import streamlit as st
from PIL import Image
from pathlib import Path
from transformers.models.clip.tokenization_clip import CLIPTokenizer
import torch

from model import model_loader
from model import pipeline

# Device configuration
DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

# Load tokenizer and model
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "./data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

# Streamlit layout
st.title("Text-to-Image Generation with Stable Diffusion")
st.write("Enter a text prompt to generate an image, or upload an image for image-to-image generation.")

# Input for text prompt
prompt = st.text_input("Text Prompt:", "Cartoon version of this person, 8k resolution.")

# Option to upload an image
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

# Strength slider for image-to-image generation
strength = st.slider("Strength (for image-to-image):", 0.0, 1.0, 0.4)

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating..."):
        # Load input image if uploaded
        input_image = None
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)

        uncond_prompt = ""  # Negative prompt
        do_cfg = True
        cfg_scale = 8  # Configuration scale
        sampler = "ddpm"
        num_inference_steps = 50
        seed = 42

        # Generate the image using the pipeline
        output_image = pipeline.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=models,
            device=DEVICE,
            idle_device="cuda",
            tokenizer=tokenizer,
        )

        # Convert output to PIL image and display
        output_pil_image = Image.fromarray(output_image)
        st.image(output_pil_image, caption="Generated Image", use_column_width=True)

# Run the app with: streamlit run app.py