import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import time

# Generator Model (same as training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.embed_dim = 50
        self.label_embed = nn.Embedding(10, self.embed_dim)
        
        self.model = nn.Sequential(
            nn.Linear(100 + self.embed_dim, 256 * 8 * 8),
            nn.BatchNorm1d(256 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_embed(labels)
        gen_input = torch.cat((noise, label_embed), dim=1)
        return self.model(gen_input)

# Load trained generator
@st.cache_resource
def load_generator(model_path="models/generator_epoch_35.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
    except:
        st.error(f"❌ Model not found at {model_path}. Using random weights instead.")
    
    generator.eval()
    return generator

# Generate images function
def generate_digits(generator, digit, num_images=1):
    device = next(generator.parameters()).device
    z = torch.randn(num_images, 100, device=device)
    labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = generator(z, labels)
    
    # Convert to numpy and denormalize
    generated = generated.cpu().detach()
    generated = (generated + 1) / 2  # Scale from [-1,1] to [0,1]
    return generated

# Convert tensor to image grid
def tensor_to_grid(tensor, nrow=8, padding=2):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=1.0)
    grid = grid.permute(1, 2, 0).numpy()  # CHW to HWC
    return grid

# Streamlit app
def main():
    # Configure page
    st.set_page_config(
        page_title="MNIST GAN Generator",
        page_icon="🔢",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Load model
    generator = load_generator()
    
    # Custom CSS
    st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background-color: #FF4B4B;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #2E86C1;
            text-align: center;
        }
        .stNumberInput {
            margin-bottom: 1rem;
        }
        .stButton button {
            background-color: #2E86C1;
            color: white;
            font-weight: bold;
            transition: all 0.3s;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #1B4F72;
            transform: scale(1.05);
        }
        .stSlider {
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # App title and description
    st.title("🔢 MNIST Digit Generator")
    st.markdown("Generate realistic handwritten digits using a Generative Adversarial Network (GAN).")
    st.markdown("---")
    
    # Create sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        digit = st.number_input("Select digit (0-9):", 
                               min_value=0, 
                               max_value=9, 
                               value=5,
                               step=1)
        
        num_images = st.slider("Number of images:", 
                              min_value=1, 
                              max_value=64, 
                              value=16, 
                              step=1)
        
        grid_size = st.slider("Grid columns:", 
                             min_value=1, 
                             max_value=8, 
                             value=4, 
                             step=1)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("This GAN was trained on the MNIST dataset to generate handwritten digits.")
        st.markdown("The model learns to create realistic digits from random noise.")
        st.markdown("Try different digits and quantities to see the results!")
        st.markdown("---")
        st.markdown("Made with [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/)")
    
    # Create two columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Parameters")
        st.write(f"Digit: **{digit}**")
        st.write(f"Images: **{num_images}**")
        st.write(f"Grid: **{grid_size} columns**")
        
        # Generate button
        if st.button("✨ Generate Digits", use_container_width=True):
            # Create a progress bar
            with st.spinner("Generating digits..."):
                progress_bar = st.progress(0)
                
                # Simulate progress (actual generation is fast)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                # Generate images
                generated = generate_digits(generator, digit, num_images)
                
                # Convert to image grid
                grid = tensor_to_grid(generated, nrow=grid_size)
                
                # Update state
                st.session_state.generated = generated
                st.session_state.grid = grid
                st.session_state.digit = digit
                st.session_state.num_images = num_images
    
    with col2:
        st.subheader("Generated Digits")
        
        # Display generated images
        if "grid" in st.session_state:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(st.session_state.grid.squeeze(), cmap="gray")
            ax.axis("off")
            ax.set_title(f"Generated {st.session_state.num_images} digits of '{st.session_state.digit}'", 
                        fontsize=14, pad=20)
            st.pyplot(fig)
            
            # Add download button
            img = Image.fromarray((st.session_state.grid.squeeze() * 255).astype(np.uint8))
            st.download_button(
                label="💾 Download Image",
                data=img,
                file_name=f"mnist_{digit}_{num_images}.png",
                mime="image/png"
            )
        else:
            st.info("👆 Click the 'Generate Digits' button to create images")
            st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", 
                    caption="Example MNIST digits", 
                    use_column_width=True)
    
    # Add examples section
    st.markdown("---")
    st.subheader("🎨 Generate Multiple Digits")
    
    # Create a grid of buttons for digits 0-9
    digits = st.columns(10)
    for i, col in enumerate(digits):
        with col:
            if st.button(f"{i}", key=f"digit_{i}"):
                with st.spinner(f"Generating {i}'s..."):
                    generated = generate_digits(generator, i, 9)
                    grid = tensor_to_grid(generated, nrow=3)
                    
                    # Update state
                    st.session_state.generated = generated
                    st.session_state.grid = grid
                    st.session_state.digit = i
                    st.session_state.num_images = 9
                    
                    # Rerun to update display
                    st.rerun()

# Run the app
if __name__ == "__main__":
    main()