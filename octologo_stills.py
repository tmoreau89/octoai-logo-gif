import streamlit as st
from PIL import Image
import webuiapi
import random


api = webuiapi.WebUIApi(
    host="octoai-logo-4jkxk521l3v1.octoai.cloud", port=443, use_https=True
)

def generate_gif(upload, food_type, invert_logo, strength):

    input_img = Image.open(upload)
    seed = random.randint(0,1000)
    unit1 = webuiapi.ControlNetUnit(
        input_image=input_img,
        module="invert" if invert_logo else "none",
        weight=strength,
        guidance_start=0,
        guidance_end=1,
        model="controlnetQRPatternQR_v2Sd15 [2d8d5750]"
    )
    res = api.txt2img(
        prompt=food_type+", RAW photo, <lora:foodphoto:0.8> foodphoto, dslr, soft lighting, high quality, film grain, Fujifilm XT",
        seed=seed,
        cfg_scale=7,
        steps=30,
        width=512,
        height=512,
        n_iter=3,
        sampler_name="Euler a",
        controlnet_units=[unit1],
        override_settings={"sd_model_checkpoint": "v1-5-pruned.safetensors"},
    )

    col1, col2, col3 = st.columns(3)
    col1.image(res.images[0])
    col2.image(res.images[1])
    col3.image(res.images[2])

st.set_page_config(layout="wide", page_title="Logo Chef (stills version)")

st.write("## Logo Chef (stills version) - Powered by OctoAI")

food_type = st.text_input("Your favorite food!", "cheese board")

invert_logo = st.checkbox('Invert Logo (turns black on white to white on black)')

strength = st.slider(
    'Select the logo strength',
    0.0, 2.0, 1.25)

my_upload = st.file_uploader("Upload a logo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    generate_gif(my_upload, food_type, invert_logo, strength)