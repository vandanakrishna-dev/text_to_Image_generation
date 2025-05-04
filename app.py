
import tkinter as tk
import customtkinter as ctk
from PIL import Image
from customtkinter import CTkImage
from authtoken import auth_token

import torch
from diffusers import StableDiffusionPipeline

# Initialize the app window
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# UI components
prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

# Load the Stable Diffusion pipeline
modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Use "cuda" if you have a compatible GPU and CUDA-enabled PyTorch

pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    use_auth_token=auth_token,
    torch_dtype=torch.float32,
)
pipe.to(device)

# Generate image from text prompt
def generate():
    lmain.configure(text="Generating...")
    prompt_text = prompt.get()

    with torch.autocast(device) if device != "cpu" else torch.inference_mode():
        result = pipe(prompt_text, guidance_scale=8.5)
        image = result["images"][0] if "images" in result else result[0]

    image.save("generatedimage.png")

    img = CTkImage(light_image=image, dark_image=image, size=(512, 512))
    lmain.configure(image=img, text="")  # Clear "Generating..." text
    lmain.image = img  # Keep reference to avoid garbage collection

# Button to trigger image generation
trigger = ctk.CTkButton(
    master=app,
    height=40,
    width=120,
    font=("Arial", 20),
    text_color="white",
    fg_color="blue",
    text="Generate",
    command=generate,
)
trigger.place(x=206, y=60)

# Run the app
app.mainloop()
