import streamlit as st
from PIL import Image
import webuiapi
import random
from base64 import b64encode
import tempfile



api = webuiapi.WebUIApi(
    host="a1111-webui-api-4jkxk521l3v1.octoai.cloud", port=443, use_https=True
)

def generate_gif(upload, meal):

    input_img = Image.open(upload)
    seed = random.randint(0,1000)
    frames = []
    progress_text = "GIF is being cooked... Please wait."
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
            "iced coffee",
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

    for food_type in meal_list:
        unit1 = webuiapi.ControlNetUnit(
            input_image=input_img,
            module="none",
            weight=1.2,
            guidance_start=0,
            guidance_end=0.90,
            model="controlnetQRPatternQR_v2Sd15 [2d8d5750]"
        )
        res = api.txt2img(
            prompt=food_type+", RAW photo, <lora:foodphoto:0.8> foodphoto, dslr, soft lighting, high quality, film grain, Fujifilm XT",
            seed=seed,
            cfg_scale=7,
            steps=30,
            width=512,
            height=512,
            n_iter=1,
            sampler_name="Euler a",
            controlnet_units=[unit1],
            override_settings={"sd_model_checkpoint": "v1-5-pruned.safetensors"},
        )
        # Uncomment if you want to see the photos frame by frame
        # st.image(res.images[0])
        percent_complete += 10
        if percent_complete == 100:
            progress_text = "GIF is ready!"
        my_bar.progress(percent_complete, text=progress_text)
        frames.append(res.images[0])

    # Save the gif and display it
    frame_one = frames[0]
    with tempfile.NamedTemporaryFile() as tmp:
        frame_one.save(tmp.name, format="GIF", append_images=frames,
                save_all=True, duration=200, loop=0)

        # Display GIF
        file_ = open(tmp.name, "rb")
        contents = file_.read()
        data_url = b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )

st.set_page_config(layout="wide", page_title="Logo Chef")

st.write("## Logo Chef - Powered by OctoAI")

meal = st.radio(
        "Meal Choice",
        ["breakfast", "lunch", "dinner"]
    )

my_upload = st.file_uploader("Upload a logo", type=["png", "jpg", "jpeg"])


if my_upload is not None:
    generate_gif(my_upload, meal)