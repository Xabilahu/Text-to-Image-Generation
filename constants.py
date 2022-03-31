import os

RESOURCES_PATH = "./res"
CKPT_PATH = "./checkpoints"
DATABASE_PATH = os.path.join(RESOURCES_PATH, "image_generation.db")
DATABASE_INIT_SCRIPT = os.path.join(RESOURCES_PATH, "image_generation.sql")
INFERENCE_PATH = os.path.join(RESOURCES_PATH, "inferences")
POSTER_PATH = os.path.join(RESOURCES_PATH, "poster")
POSTER_SESSION_PATH = os.path.join(RESOURCES_PATH, "poster-session")
STYLES_PATH = os.path.join(RESOURCES_PATH, "styles.css")
IMAGENET_CLASSES_PATH = os.path.join(RESOURCES_PATH, "imagenet_classes.json")
IMAGENET_CLASSES_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
