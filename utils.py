import matplotlib.pyplot as plt
import numpy as np
import torch
import urllib3
import random
import os

from tqdm import tqdm
from PIL import Image


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def display_tensor(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
        
    img = tensor.clone()
    img = img.mul(255).byte()
    img = img.cpu().numpy().transpose((1,2,0)).astype(np.uint8)
    plt.figure(figsize=(10,7))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    

def save_to_gif(imgs, filename, save_last_image=True):
    pil_img_array = [
        Image.fromarray(
            img[0].mul(255).byte().numpy().transpose((1,2,0)).astype(np.uint8)
        ) 
        for img in imgs
    ]
    
    gif_name, image_name = f"{filename}.gif", f"{filename}.png"
    
    if os.path.exists(gif_name) or os.path.exists(image_name):
        all_files = os.listdir(os.path.dirname(gif_name))
        basename = os.path.splitext(os.path.basename(gif_name))[0]
        n_files = len(list(filter(lambda x: x.startswith(basename), all_files)))
        gif_name, image_name = f"{filename}+{n_files}.gif", f"{filename}+{n_files}.png"
    
    generated_image_filenames = [None, None]
    
    if len(pil_img_array) > 1:
        pil_img_array[0].save(
            gif_name,
            save_all=True,
            append_images=pil_img_array[1:],
            duration=10,
            loop=0
        )
        generated_image_filenames[0] = gif_name
    
    if save_last_image or len(pil_img_array) == 1:
        pil_img_array[-1].save(image_name)
        generated_image_filenames[1] = image_name
    
    assert generated_image_filenames != [None, None], "No image has been provided to the save_to_gif function"
    return generated_image_filenames
        
def download_file(url, filename, chunk_size=8192):
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    with tqdm(
        total=int(r.headers["Content-Length"]) // chunk_size + 1,
        unit='iB', 
        unit_scale=True,
        unit_divisor=1024
    ) as pbar:
        with open(filename, "wb") as f:
            while True:
                data = r.read(chunk_size)
                if not data:
                    break
                f.write(data)
                pbar.update(1)
    r.release_conn()