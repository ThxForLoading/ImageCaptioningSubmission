import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set up the model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_image_description(image):
    try:
        # Process the image and generate a caption
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=50, num_beams=3, temperature=0.8)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return f"The image shows: {caption}"
    except Exception as e:
        return f"Error processing image: {e}"

# Streamlit app interface
st.title("Image Captioning App")
st.write("Upload an image, and the app will generate a description.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Generate and display the image description
    with st.spinner("Generating description..."):
        description = generate_image_description(image)
    st.success("Caption generated!")
    st.write(description)
