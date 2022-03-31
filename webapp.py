import json
import multiprocessing as mp
import os
import random
import re
from collections import defaultdict

import streamlit as st
from streamlit.script_runner import get_script_run_ctx

from constants import *
from database import (
    get_image_data,
    get_random_image_id,
    image_id_from_filename,
    process_vote,
    push_job,
)
from models import BIGGAN, VQGAN
from utils import download_file

IMAGENET_CLASSES = None


@st.cache
def load_imagenet_dict():
    if not os.path.exists(IMAGENET_CLASSES_PATH):
        download_file(IMAGENET_CLASSES_URL, IMAGENET_CLASSES_PATH)

    with open(IMAGENET_CLASSES_PATH, "r") as f:
        contents = json.load(f)

    return {value[1]: key for key, value in contents.items()}


def get_session_id():
    return get_script_run_ctx().session_id


def retrieve_images(directory):
    dirs = directory if type(directory) is list else [directory]
    image_dict = defaultdict(lambda: [None, None])

    for dirname in dirs:
        for filename in os.listdir(dirname):
            if not filename.startswith("."):
                split_filename = os.path.splitext(filename)
                image_dict[os.path.join(dirname, split_filename[0])][
                    0 if split_filename[1] == ".gif" else 1
                ] = os.path.join(dirname, filename)

    return [value for _, value in image_dict.items()]


def inject_css():
    with open("res/styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def setup_sidebar():
    with st.sidebar:
        st.title("Page Selection")
        st.selectbox(
            "Page View",
            ("Text-to-Image Generation", "Previous Inferences", "Vote images"),
            key="page_view",
        )

        if st.session_state.page_view == "Text-to-Image Generation":
            st.title("Configuration Parameters")
            st.selectbox(
                "Model Type",
                ("VQ-GAN", "BigGAN"),
                key="model_type",
            )

            if st.session_state.model_type == "VQ-GAN":
                model_names = VQGAN.available_models()
            elif st.session_state.model_type == "BigGAN":
                model_names = BIGGAN.available_models()

            st.selectbox(
                "Model Name",
                model_names,
                key="model_name",
            )
            st.number_input(
                "Random Seed",
                step=1,
                value=0,
                key="seed",
            )
            st.slider(
                "Step Size",
                min_value=0.01,
                max_value=0.99,
                step=0.01,
                value=0.5,
                key="step_size",
            )
            st.slider(
                "Update Frequency",
                min_value=50,
                max_value=150,
                value=100,
                key="update_freq",
            )
            st.slider(
                "Optimization Steps (~ 1 min / step)",
                min_value=1,
                max_value=5,
                value=1,
                key="optimization_steps",
            )
            st.slider(
                "Similarity Factor",
                min_value=0.1,
                max_value=100.0,
                step=0.1,
                value=10.0,
                key="similarity_factor",
            )

            if st.session_state.model_type == "BigGAN":
                st.slider(
                    "Class Smoothing Factor",
                    min_value=0.01,
                    max_value=1.0,
                    step=0.01,
                    value=0.1,
                    key="smoothing_factor",
                )
                st.slider(
                    "Truncation Factor",
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1,
                    value=1.0,
                    key="truncation_factor",
                )

            st.button(
                "Reset Configuration Parameters",
                on_click=reset_config,
            )


def setup_main_area():
    st.info(
        "Work done by Xabier Lahuerta Vázquez as the Deep Learning course final project. \n\nFeel free to contact me at [xlahuerta@pm.me](mailto:xlahuerta@pm.me)."
    )

    if "model_type" not in st.session_state:
        st.session_state.model_type = "VQ-GAN"

    if "filter_key" not in st.session_state:
        st.session_state.filter_key = "All"

    if "page_view" not in st.session_state:
        st.session_state.page_view = "Text-to-Image Generation"

    if "inferences" not in st.session_state:
        st.session_state.inferences = []

    if "quality_voted" not in st.session_state:
        st.session_state.quality_voted = defaultdict(bool)

    if "agreement_voted" not in st.session_state:
        st.session_state.agreement_voted = defaultdict(bool)

    if st.session_state.page_view == "Text-to-Image Generation":
        text_to_image_generation_page()
    elif st.session_state.page_view == "Previous Inferences":
        previous_inferences_page(st.session_state.filter_key)
    elif st.session_state.page_view == "Vote images":
        image_voting_page()


def text_to_image_generation_page():
    st.title("Text-to-Image Generation")
    st.text_input("Text Input", key="current_text")

    if st.session_state.model_type == "BigGAN":
        st.multiselect(
            "Class Values",
            IMAGENET_CLASSES.keys(),
            key="imagenet_classes",
        )
        st.checkbox(
            "Optimize class vector",
            value=True,
            key="optimize_class",
        )

    containers = st.columns(2)
    containers[0].button("Clear", on_click=reset_text)
    containers[1].button("Run", on_click=process_input)
    st.session_state.history_container = st.container()
    st.session_state.history_container.header("Inference History")


def render_images(gif_filename, image_filename):
    if gif_filename is None:
        st.image(image_filename)
    elif image_filename is None:
        st.image(gif_filename)
    else:
        columns = st.columns(2)
        columns[0].image(gif_filename)
        columns[1].image(image_filename)


def previous_inferences_page(filter_key):
    st.title("Previous Inferences")
    containers = st.columns(3)
    containers[0].selectbox(
        "Filter by Model",
        ["All", "Poster", "Poster Session", "VQ-GAN"]
        + VQGAN.available_models()
        + ["BigGAN"]
        + BIGGAN.available_models(),
        key="filter_key",
    )

    filter_dirs = []
    if st.session_state.filter_key == "All":
        for upper_dirname in [INFERENCE_PATH, POSTER_PATH, POSTER_SESSION_PATH]:
            for dirname in os.listdir(upper_dirname):
                filter_dirs.append(os.path.join(upper_dirname, dirname))
    elif st.session_state.filter_key == "Poster Session":
        for dirname in os.listdir(POSTER_SESSION_PATH):
            filter_dirs.append(os.path.join(POSTER_SESSION_PATH, dirname))
    elif st.session_state.filter_key == "Poster":
        filter_dirs = [
            os.path.join(POSTER_PATH, "biggan-deep-512"),
            os.path.join(POSTER_PATH, "vqgan-imagenet"),
        ]
    elif st.session_state.filter_key == "VQ-GAN":
        for model_name in VQGAN.available_models():
            filter_dirs.append(os.path.join(INFERENCE_PATH, model_name))
    elif st.session_state.filter_key == "BigGAN":
        for model_name in BIGGAN.available_models():
            filter_dirs.append(os.path.join(INFERENCE_PATH, model_name))
    else:
        for top_level_dirname in [INFERENCE_PATH, POSTER_PATH, POSTER_SESSION_PATH]:
            dirname = os.path.join(top_level_dirname, filter_key)
            if os.path.exists(dirname):
                filter_dirs.append(dirname)

    for animated_gif, final_image in retrieve_images(filter_dirs):
        not_none = animated_gif if animated_gif is not None else final_image
        basename = os.path.splitext(os.path.basename(not_none))[0]
        prompt = re.sub(
            "\+.*$", "", " ".join((" | ".join(basename.split("_"))).split("-")), re.M
        )
        key_base = re.sub("res/", "", os.path.splitext(not_none)[0])
        container = st.container()

        if "vqgan" in not_none:
            model_type = "VQ-GAN"
            model_name = re.search("\/vqgan-[^\/]+\/", not_none).group(0)[1:-1]
        elif "biggan" in not_none:
            model_type = "BigGAN"
            model_name = re.search("\/biggan-[^\/]+\/", not_none).group(0)[1:-1]

        container.subheader(prompt)
        container.markdown(f"Images generated with model {model_type}: {model_name}")
        render_images(animated_gif, final_image)


def goto_view_page(filename):
    image_id = image_id_from_filename(filename)
    st.session_state.vote_now_id = image_id
    st.session_state.page_view = "Vote images"


def submit_vote(vote_type, vote_score):
    process_vote(vote_type, vote_score, st.session_state.vote_now_id, get_session_id())
    if vote_type == "quality":
        st.session_state.quality_voted[st.session_state.vote_now_id] = True
    elif vote_type == "agreement":
        st.session_state.agreement_voted[st.session_state.vote_now_id] = True
    else:
        raise RuntimeError(f"Unrecognized vote_type: {vote_type}")


def image_voting_page():
    if "vote_now_id" not in st.session_state:
        st.session_state.vote_now_id = get_random_image_id(get_session_id())

    image_id = st.session_state.vote_now_id
    prompt, gif_filename, image_filename = get_image_data(st.session_state.vote_now_id)
    st.title("Vote the generated images")
    st.info(
        "You can submit two votes for a given image: one corresponding to the __quality__ of the generation, and the other correspondng to the __agreement__ with the textual description."
    )
    st.warning("Voting scale: 1 (Lowest score) - 5 (Highest score)")
    st.subheader(prompt)
    render_images(gif_filename, image_filename)
    vote_container = st.container()

    vote_container.markdown("Generation quality")
    if not st.session_state.quality_voted[image_id]:
        columns = vote_container.columns(5)
        for i in range(1, 6):
            btn = columns[i - 1].button(
                label=f"{i}",
                key="quality_{i}",
                on_click=submit_vote,
                args=(
                    "quality",
                    i,
                ),
            )
    else:
        vote_container.success("Vote successfully processed!")

    vote_container.markdown("Agreement with the textual decription")
    if not st.session_state.agreement_voted[image_id]:
        columns = vote_container.columns(5)
        for i in range(1, 6):
            btn = columns[i - 1].button(
                label=f"{i}",
                key="agreement_{i}",
                on_click=submit_vote,
                args=(
                    "agreement",
                    i,
                ),
            )
    else:
        vote_container.success("Vote successfully processed!")

    st.button("Next image", on_click=reset_voting_status)


def reset_voting_status():
    del st.session_state.vote_now_id


def reset_config():
    st.session_state.model_type = "VQ-GAN"
    st.session_state.model_name = "vqgan-imagenet"
    st.session_state.seed = 0
    st.session_state.step_size = 0.5
    st.session_state.update_freq = 100
    st.session_state.optimization_steps = 1
    st.session_state.similarity_factor = 10.0


def reset_text():
    st.session_state.current_text = ""
    st.session_state.imagenet_classes = []
    st.session_state.optimize_class = True
    st.session_state.smoothing_factor = 0.1
    st.session_state.truncation_factor = 1.0


def render_history():
    with st.session_state.history_container:
        for i in range(len(st.session_state.inferences), 0, -1):
            with st.container():
                st.subheader(f"Run #{i}: {st.session_state.inferences[i - 1]['text']}")
                with st.expander("Configuration Parameters"):
                    config_str = "{\n"
                    for key, value in st.session_state.inferences[i - 1].items():
                        if key in ["text", "generation", "dirname"]:
                            continue
                        delim_char = "'" if type(value) == str else ""
                        config_str += f"  '{key}': {delim_char}{value}{delim_char},\n"
                    config_str += "}"
                    st.code(config_str)

                render_images(*st.session_state.inferences[i - 1]["generation"])


def process_input():
    config = {
        "text": st.session_state.current_text,
        "model_type": st.session_state.model_type,
        "model_name": st.session_state.model_name,
        "seed": st.session_state.seed,
        "step_size": st.session_state.step_size,
        "update_freq": st.session_state.update_freq,
        "optimization_steps": st.session_state.optimization_steps,
        "similarity_factor": st.session_state.similarity_factor,
        "smoothing_factor": None,
        "truncation_factor": None,
        "imagenet_classes": None,
        "optimize_class": None,
    }

    if config["model_type"] == "BigGAN":
        config["smoothing_factor"] = st.session_state.smoothing_factor
        config["truncation_factor"] = st.session_state.truncation_factor
        config["imagenet_classes"] = "&".join(
            [IMAGENET_CLASSES[x] for x in st.session_state.imagenet_classes]
        )
        config["optimize_class"] = st.session_state.optimize_class

    output_dir = os.path.join(POSTER_SESSION_PATH, config["model_name"])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    config["dirname"] = output_dir
    st.session_state.inferences.append(config)
    with st.spinner("Generating image..."):
        inject_css()  # Keep css styling on update (maybe also render history?)
        image_files = mp.Manager().list([None, None])
        proc = mp.Process(target=push_job, args=(config, image_files))
        proc.start()
        proc.join()
        st.session_state.inferences[-1]["generation"] = image_files[:]


def run_app():
    global IMAGENET_CLASSES

    inject_css()
    IMAGENET_CLASSES = load_imagenet_dict()
    setup_main_area()
    setup_sidebar()
    if st.session_state.page_view == "Text-to-Image Generation":
        render_history()


if __name__ == "__main__":
    st.set_page_config("DL4NLP XLahuerta", page_icon=":camera_with_flash:")
    run_app()
