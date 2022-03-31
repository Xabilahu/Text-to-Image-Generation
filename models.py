import sys, os, random

sys.path.append('./taming-transformers')

import torch
from torchvision.transforms import Normalize
from CLIP import clip
from taming.models import cond_transformer, vqgan
from omegaconf import OmegaConf
from pytorch_pretrained_biggan import (
    BigGAN,
    one_hot_from_int,
    truncated_noise_sample,
)

from abc import ABCMeta, abstractmethod
from utils import download_file
from constants import CKPT_PATH

def retrieve_config_and_model(model_name, download_dict, config_filename, checkpoint_filename):
    """Return the path to the config and checkpoint files, downloading them if needed"""
    model_ckpt_path = f"{CKPT_PATH}/{model_name}"
    if not os.path.exists(model_ckpt_path):
        os.makedirs(model_ckpt_path)
        download_file(
            download_dict["config"],
            f"{model_ckpt_path}/{config_filename}",
        )
        download_file(
            download_dict["checkpoint"],
            f"{model_ckpt_path}/{checkpoint_filename}",
        )
    return f"{model_ckpt_path}/{config_filename}", f"{model_ckpt_path}/{checkpoint_filename}"

class ModelWrapper(metaclass=ABCMeta):    
    def __init__(self, clip_model, device='cuda:0'):
        if clip_model not in clip.available_models():
            raise ValueError(f"Provided model_name = {model_name} is not available. The list of available models is {clip.available_models()}")
            
        self.device = device
        self.normalizer = Normalize(
            mean = [0.48145466, 0.4578275, 0.40821073], 
            std = [0.26862954, 0.26130258, 0.27577711],
        ).to(self.device)
        self.tokenizer = clip.tokenize
        self.embedder, _ = clip.load(clip_model, jit=False)
        self.embedder.eval()
        self.embedder = self.embedder.float().to(self.device)
        self.generator = None
        
    def get_embedder_input_resolution(self):
        return self.embedder.visual.input_resolution
        
    def embed_image(self, imgs):
        """Returns the embedded representation of the given image"""
        return self.embedder.encode_image(self.normalizer(imgs))

    def embed_text(self, texts):
        """Returns the embedded representation of the given text"""
        if type(texts) is not list:
            texts = [texts]
        return self.embedder.encode_text(self.tokenizer(texts).to(self.device))
        
    @classmethod
    def available_models(cls):
        """Return a list of available pretrained model names"""
        raise NotImplementedError
        
    @abstractmethod
    def get_initial_latent(self, **kwargs):
        """Returns the initial latent representation that will be used to guide the generation"""
        raise NotImplementedError
        
    @abstractmethod
    def synthesize_image(self, latent):
        """Returns the image corresponding to the given latent vector"""
        raise NotImplementedError
        
class VQGAN(ModelWrapper):
    model_data = {
        'vqgan-imagenet': {
            'config': 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
            'checkpoint': 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
        },
        'vqgan-faceshq': {
            'config': 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT',
            'checkpoint': 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt',
        },
        'vqgan-wikiart': {
            'config': 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml',
            'checkpoint': 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt',
        },
    }

    def __init__(self, model_name, clip_model, device='cuda:0'):
        if model_name not in self.model_data:
            raise ValueError(f"Provided model_name = \"{model_name}\" is not available. The list of available models is {self.available_models()}")
        
        super(VQGAN, self).__init__(clip_model, device)
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load the corresponding VQGAN model"""
        config_path, checkpoint_path = retrieve_config_and_model(
            self.model_name,
            self.model_data[self.model_name],
            "config.yaml",
            "model.ckpt",
        )
        config = OmegaConf.load(config_path)
        if config.model.target == 'taming.models.vqgan.VQModel':
            self.generator = vqgan.VQModel(**config.model.params)
            self.generator.eval()
            self.generator.init_from_ckpt(checkpoint_path)
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            tmp_model = cond_transformer.Net2NetTransformer(**config.model.params)
            tmp_model.eval()
            tmp_model.init_from_ckpt(checkpoint_path)
            self.generator = tmp_model.first_stage_model
        else:
            raise ValueError(f"Unsupported model architecture: \"{config.model.target}\"")
        del self.generator.loss
        self.generator = self.generator.to(self.device)

    @classmethod
    def available_models(cls):
        """Return a list of available VQGAN model names"""
        return list(cls.model_data.keys())
    
    def get_initial_latent(self, **kwargs):
        """Returns the initial latent representation that will be used to guide the generation"""
        if "image_shape" not in kwargs:
            raise ValueError("Missing image_shape named parameter.")
            
        image_shape = kwargs["image_shape"]
        patch_size = 2 ** (self.generator.decoder.num_resolutions - 1)
        tokens_x, tokens_y = image_shape[0] // patch_size, image_shape[1] // patch_size
        codebook_len = self.generator.quantize.n_e
        
        z = torch.nn.functional.one_hot(torch.randint(codebook_len, [tokens_y * tokens_x]).to(self.device), codebook_len).float()
        z = torch.matmul(z, self.generator.quantize.embedding.weight)
        z = z.view([-1, tokens_y, tokens_x, self.generator.quantize.e_dim]).permute(0, 3, 1, 2)
        z = torch.rand_like(z) * 2 - 1
        
        return z.requires_grad_()
    
    def synthesize_image(self, latent):
        """Returns the image corresponding to the given latent vector"""
        l_quant, _, _ = self.generator.quantize(latent)
        l_decode = self.generator.decode(l_quant)
        return torch.clamp(l_decode.add(1).div(2), min=0, max=1) # Map values from [-1,1] to [0,1]
    
class BIGGAN(ModelWrapper):
    model_data = {
        'biggan-deep-128': {
            'config': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-config.json",
            'checkpoint': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-128-pytorch_model.bin",
        },
        'biggan-deep-256': {
            'config': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-config.json",
            'checkpoint': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-256-pytorch_model.bin",
        },
        'biggan-deep-512': {
            'config': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-config.json",
            'checkpoint': "https://s3.amazonaws.com/models.huggingface.co/biggan/biggan-deep-512-pytorch_model.bin",
        },
    }
    
    def __init__(self, model_name, clip_model, class_smoothing=0.1, truncation=1, optimize_class=True, device='cuda:0'):
        if model_name not in self.model_data:
            raise ValueError(f"Provided model_name = \"{model_name}\" is not available. The list of available models is {self.available_models()}")
        
        super(BIGGAN, self).__init__(clip_model, device)
        self.class_smoothing = class_smoothing
        self.truncation = truncation
        self.optimize_class = optimize_class
        self.model_name = model_name
        self._load_model()
        
    def _load_model(self):
        config_path, _ = retrieve_config_and_model(
            self.model_name,
            self.model_data[self.model_name],
            "config.json",
            "pytorch_model.bin",
        )
        
        self.generator = BigGAN.from_pretrained(os.path.dirname(config_path))
        self.generator.eval()
        self.generator = self.generator.to(self.device)
    
    @classmethod
    def available_models(cls):
        """Return a list of available BigGAN model names"""
        return list(cls.model_data.keys())
    
    def get_initial_latent(self, **kwargs):
        """Returns the initial latent representation that will be used to guide the generation"""
        if "imagenet_classes" not in kwargs:
            raise ValueError(f"Missing {keyword} named parameter.")
            
        class_idx = kwargs["imagenet_classes"]
        if len(class_idx) > 0:
            class_vector = one_hot_from_int(class_idx, batch_size=len(class_idx)).sum(0)
            class_vector = class_vector * (1 - self.class_smoothing * 1000 / 999) + self.class_smoothing / 999
        else:
            class_vector = torch.ones((1, 1000), dtype=torch.float32) * self.class_smoothing / 999
            class_vector[random.randint(0, 999)] = 1 - self.class_smoothing
            
        noise = torch.tensor(truncated_noise_sample(), requires_grad=True, device=self.device)
        class_vector = torch.tensor(class_vector, device=self.device)
        class_vector = torch.log(class_vector + 1e-8).unsqueeze(0) # Avoid possible log(0)
        
        if self.optimize_class:
            class_vector = class_vector.requires_grad_()
            
        return [noise, class_vector]
        
    def synthesize_image(self, latent):
        noise_vector, class_vector = latent
        noise_vector_trunc = noise_vector.clamp(-2 * self.truncation, 2 * self.truncation)
        class_vector_norm = torch.nn.functional.softmax(class_vector, dim=-1)
        l_decode = self.generator(noise_vector_trunc, class_vector_norm, self.truncation)
        return torch.clamp(l_decode.add(1).div(2), min=0, max=1)
    
if __name__ == "__main__":
    from utils import inference, save_to_gif
    
    model_name = "biggan-deep-512"
    model = BIGGAN(model_name, "ViT-B/32")
    
    prompts = [
        # ["nightmare landscape"],
        # ["nightmare landscape", "psychedelic"],
        # ["snowed mountains with a glacial valley"],
        # ["pirate ship battle in the middle of the ocean"],
        # ["caribean beach on a sunny day"],
        # ["the Mona Lisa in the style of Vincent van Gogh"],
        # ["the persistence of memory by Dali"],
        # ["starry night"],
        # ["Guernica by Pablo Picasso"],
        # ["Guernica in the style of Vincent van Gogh"],
        # ["Guernica in the style of Claude Monet"],
        # ["las meninas"],
        # ["las meninas in modernist style"],
        # ["spacecraft landing on the moon"],
        # ["spacecraft landing on the moon trending on artstation"],
        # ["spacecraft landing on the moon unreal engine"],
        # ["the beginning of the universe"],
        # ["the beginning of the universe trending on artstation"],
        # ["the beginning of the universe unreal engine"],
        # ["samurai trending on artstation"],
        # ["cyberpunk samurai"],
        # ["a girl in a picnic table using a laptop"],
        # ["laptop"],
        ["A photo of a flying penguin"],
        # ["A photo of a gold garden spider hanging on web"],
        # ["A photo of dog with a cigarette"],
        # ["A photo of a blue bird with big beak"],
        # ["A photo of a lemon with hair and face"],
        # ["A photo of a polar bear under the blue sky"]
    ]
    
    for prompt in prompts:
        print(f"Generating: {' | '.join(prompt)}")
        imgs = inference(model, (300,300), prompt, display_images=True)
        filename = '_'.join(['-'.join(text.split(' ')) for text in prompt])
        save_to_gif(imgs, f"res/poster/{model_name}/{filename}")
