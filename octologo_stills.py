import streamlit as st
from PIL import Image
import webuiapi
import random


api = webuiapi.WebUIApi(
    host="octoai-qr-logo-demo-4jkxk521l3v1.octoai.cloud", port=443, use_https=True
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
        prompt=food_type+", RAW photo, <lora:foodphoto:0.8> foodphoto, soft lighting, high quality, film grain, Fujifilm XT",
        negative_prompt="nsfw, nudity, human, person",
        seed=seed,
        cfg_scale=7,
        steps=30,
        width=768,
        height=768,
        n_iter=1,
        sampler_name="Euler a",
        controlnet_units=[unit1],
        override_settings={"sd_model_checkpoint": "realistic-v40.safetensors"},
    )

    st.image(res.images[0])

st.set_page_config(layout="wide", page_title="Foody Logos (stills version)")

st.write("## Foody Logos (stills version) - Powered by OctoAI")

st.write("Try the GIF version [here](https://foodylogos.streamlit.app/)")

food_type = st.text_input("Your favorite food!", "cheese board")

invert_logo = st.checkbox('Invert Logo (turns black on white to white on black)')

strength = st.slider(
    'Select the logo contour strength',
    0.0, 2.0, 1.25)

my_upload = st.file_uploader("Upload a logo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    generate_gif(my_upload, food_type, invert_logo, strength)