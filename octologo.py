import streamlit as st
from PIL import Image
import webuiapi
import random
from base64 import b64encode
import tempfile



api = webuiapi.WebUIApi(
    host="octoai-qr-logo-demo-4jkxk521l3v1.octoai.cloud", port=443, use_https=True
)

def generate_gif(upload, invert_logo, strength, meal):

    input_img = Image.open(upload)
    seed = random.randint(0,1000)
    frames = []
    progress_text = "GIF is in the oven..."
    my_bar = st.progress(0, text=progress_text)
    percent_complete = 0

    if meal == "breakfast":
        meal_list = [
            "latte",
            "pancakes",
            "croissant",
            "yoghurt with berries",
            "smoothie",
            "waffles",
            "cheese omelette",
            "granola with milk",
            "scrambled eggs with bacon",
            "cookies",
        ]
    elif meal == "lunch":
        meal_list = [
            "margherita pizza",
            "healthy wrap",
            "brie and fruit",
            "sandwitch",
            "potato chips",
            "quiche",
            "quinoa salad",
            "sushi rolls",
            "tacos",
            "espresso",
        ]
    elif meal == "dinner":
        meal_list = [
            "cheese platter",
            "beer",
            "roasted fish"
            "sashimi",
            "spaghetti with meatballs",
            "pho soup",
            "steak frites",
            "ramen",
            "fried rice",
            "cheese burger",
            "chocolate desert"
        ]
    elif meal == "birthday special":
        meal_list = [
            "birthday cake, candles",
            "rainbow colored cupcakes",
            "colorful candy, gummies, bubblegum",
            "pink sprinkled doughnuts",
            "fruit punch",
            "colorful macarons"
            "balloons",
            "rainbow colored fruit skewers",
            "purple milkshake",
            "ice cream with sprinkles",
            "chocolate brownies"
        ]

    placeholder = st.empty()
    for food_type in meal_list:
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
            width=512,
            height=512,
            n_iter=1,
            sampler_name="Euler a",
            controlnet_units=[unit1],
            override_settings={"sd_model_checkpoint": "realistic-v40.safetensors"},
        )
        # Uncomment if you want to see the photos frame by frame
        percent_complete += 10
        if percent_complete == 100:
            progress_text = "GIF is ready!"
        my_bar.progress(percent_complete, text=progress_text)
        frames.append(res.images[0])
        placeholder.empty()
        placeholder.image(res.images[0])

    # Save the gif and display it
    with tempfile.NamedTemporaryFile() as tmp:
        frame_one = frames[0]
        frame_one.save(tmp.name, format="GIF", append_images=frames,
                save_all=True, duration=200, loop=0)

        # Display GIF
        with open(tmp.name, "rb") as gif_f:
            contents = gif_f.read()
            data_url = b64encode(contents).decode("utf-8")

            placeholder.empty()
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
                unsafe_allow_html=True,
            )
            # st.markdown("\n")
            # st.download_button("Download your gif", gif_f, file_name="foody_logo.gif")

st.set_page_config(layout="wide", page_title="Foody Logos")

st.write("## Foody Logos - Powered by OctoAI")

st.write("Try the stills version [here](https://foodylogos-stills.streamlit.app/)")

meal = st.radio(
        "Meal Choice",
        ["breakfast", "lunch", "dinner", "birthday special"]
    )

invert_logo = st.checkbox('Invert Logo (turns black on white to white on black)')

strength = st.slider(
    'Select the logo contour strength',
    0.0, 2.0, 1.5)

my_upload = st.file_uploader("Upload a logo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    generate_gif(my_upload, invert_logo, strength, meal)