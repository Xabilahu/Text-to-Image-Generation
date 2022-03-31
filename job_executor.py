import os
import signal

import torch
from torchvision.transforms import (
    CenterCrop,
    InterpolationMode,
    RandomAffine,
    RandomHorizontalFlip,
)

from database import get_image_config, poll_pending_jobs, update_image
from models import BIGGAN, VQGAN
from utils import display_tensor, save_to_gif, set_seed


# Taken from https://discuss.pytorch.org/t/using-nn-function-interpolate-inside-nn-sequential/23588/2#post_2
class Interpolate(torch.nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, size=self.size, mode=self.mode, align_corners=True
        )
        return x


# We can use either crop or interpolation to get the image to the wanted resolution,
# anyways, better results are expected when using interpolation
def init_transforms(output_resolution):
    return torch.nn.Sequential(
        RandomHorizontalFlip(),
        RandomAffine(30.0, interpolation=InterpolationMode.BILINEAR),
        # CenterCrop(output_resolution),
        Interpolate(output_resolution, "bicubic"),
    )


def augment(img, transform_func, k=32):
    imgs = [transform_func(img) for _ in range(k)]
    augmented_imgs = torch.cat(imgs, 0)
    return augmented_imgs


def run_inference(config):
    set_seed(config["seed"])

    if config["model_name"] in VQGAN.available_models():
        model = VQGAN(config["model_name"], "ViT-B/32")
    elif config["model_name"] in BIGGAN.available_models():
        model = BIGGAN(
            config["model_name"],
            "ViT-B/32",
            class_smoothing=config["smoothing_factor"],
            truncation=config["truncation_factor"],
            optimize_class=config["optimize_class"],
        )
    else:
        raise NotImplemented

    imgs = inference(
        model,
        [300, 300],
        config["text"],
        learning_rate=config["step_size"],
        similarity_factor=config["similarity_factor"],
        optimization_steps=config["optimization_steps"],
        update_freq=config["update_freq"],
        imagenet_classes=list(map(int, config["imagenet_classes"].split("&")))
        if config["imagenet_classes"] is not None
        else [],
        display_images=False,
    )

    output_filename = os.path.join(
        config["dirname"],
        "_".join(["-".join(x.split(" ")) for x in config["text"]]),
    )
    return save_to_gif(imgs, output_filename, save_last_image=True)


def inference(
    model,
    image_shape,
    text,
    learning_rate=0.5,
    similarity_factor=1,
    optimization_steps=5,
    update_freq=100,
    imagenet_classes=[],
    display_images=True,
):
    if type(text) is not list:
        text = [text]

    data_augmentation = init_transforms(model.get_embedder_input_resolution())
    with torch.no_grad():
        z = model.get_initial_latent(
            image_shape=image_shape, imagenet_classes=imagenet_classes
        )
        embed_text = model.embed_text(text)

    imgs = []
    if display_images:
        print("Initial image")
        with torch.no_grad():
            display_tensor(model.synthesize_image(z))

    optimizer = torch.optim.Adam(
        [{"params": [z] if type(z) is not list else z, "lr": learning_rate}]
    )
    for it in range(1, optimization_steps + 1):
        for i in range(update_freq):
            out = model.synthesize_image(z)
            out = augment(out, data_augmentation)
            out = model.embed_image(out)
            if len(text) == 1:
                loss = -torch.cosine_similarity(embed_text, out, -1)
                loss = loss.mean()
            else:
                loss = []
                for i in range(len(text)):
                    alpha = (
                        -similarity_factor if i == 0 else -(similarity_factor / 2)
                    )  # Weight the first prompt twice the rest
                    loss.append(
                        alpha * torch.cosine_similarity(embed_text[i, :], out, -1)
                    )
                loss = (
                    torch.stack(loss, dim=0).sum().div((len(text) - 1) / 2 + 1)
                )  # Take the weighting into account when computing the mean
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if display_images:
            print(f"Iteration: {it} - Loss: {loss.item()}")
        with torch.no_grad():
            imgs.append(model.synthesize_image(z).clone().detach().cpu())
            if display_images:
                display_tensor(imgs[-1])

    return imgs


def main():
    job_id, proc_id, image_id = poll_pending_jobs(poll_freq=1)
    config = get_image_config(image_id)
    gif_filename, image_filename = run_inference(config)
    update_image(
        job_id,
        image_id,
        os.path.basename(gif_filename) if gif_filename is not None else None,
        os.path.basename(image_filename),
    )
    os.kill(proc_id, signal.SIGUSR1)


if __name__ == "__main__":
    main()
