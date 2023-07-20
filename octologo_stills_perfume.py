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
        guidance_end=0.75,
        model="controlnetQRPatternQR_v2Sd15 [2d8d5750]"
    )
    res = api.txt2img(
        # prompt=food_type+", RAW photo, <lora:foodphoto:0.8> foodphoto, dslr, soft lighting, high quality, film grain, Fujifilm XT",
        prompt="perfume bottle, no humans, gradient background, simple background, {}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3 <lora:Perfume_Bottle_v001:1>".format(food_type),
        negative_prompt="(blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation,watermark,",
        seed=seed,
        cfg_scale=7,
        steps=25,
        width=768,
        height=768,
        n_iter=3,
        sampler_name="Euler a",
        controlnet_units=[unit1],
        override_settings={"sd_model_checkpoint": "realistic.safetensors"},
    )

    col1, col2, col3 = st.columns(3)
    col1.image(res.images[0])
    col2.image(res.images[1])
    col3.image(res.images[2])

st.set_page_config(layout="wide", page_title="Logo Chef (perfume version)")

st.write("## Logo Chef (perfume version) - Powered by OctoAI")

food_type = st.text_input("Describe your ideal perfume bottle!", "pink flower")

invert_logo = st.checkbox('Invert Logo (turns black on white to white on black)')

strength = st.slider(
    'Select the logo strength',
    0.0, 2.0, 1.0)

my_upload = st.file_uploader("Upload a logo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    generate_gif(my_upload, food_type, invert_logo, strength)