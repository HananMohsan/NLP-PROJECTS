!pip install -q streamlit
pip install tqdm
pip install pandas
pip install numpy
pip install diffusers
pip install transformers
pip install matplotlib
pip install opencv-python
pip install torch
%%writefile app.py
import streamlit as st
from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Configure Streamlit layout
st.set_page_config(page_title="Image Generation App", layout="wide")

# Define configuration class
class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (300, 300)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt4"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Load the image generation model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

# Define the Streamlit app
def main():
    # Set the app title
    st.title("Image Generation App")

    # Generate the image based on user input
    prompt = st.text_input("Enter a prompt:")
    if st.button("Generate Image"):
        if prompt:
            # Generate the image
            image = generate_image(prompt, image_gen_model)

            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)

# Function to generate an image based on a prompt
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return np.array(image)

# Run the app
if __name__ == "__main__":
    main()


!npm install localtunnel
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501 & curl ipv4.icanhazip.com
