import torch

from transformers.tools.base import Tool, get_default_device
from transformers.utils import (
    is_accelerate_available,
    is_vision_available,
)

from diffusers import DiffusionPipeline


IMAGE_TRANSFORMATION_DESCRIPTION = (
    "This is a tool that transforms an image according to a prompt. It takes two inputs: `image`, which should be "
    "the image to transform, and `prompt`, which should be the prompt to use to change it. It returns the "
    "modified image."
)


class ImageTransformationTool(Tool):
    default_stable_diffusion_checkpoint = "timbrooks/instruct-pix2pix"
    description = IMAGE_TRANSFORMATION_DESCRIPTION
    inputs = ['image', 'text']
    outputs = ['image']

    def __init__(self, device=None, controlnet=None, stable_diffusion=None, **hub_kwargs) -> None:
        if not is_accelerate_available():
            raise ImportError("Accelerate should be installed in order to use tools.")
        if not is_vision_available():
            raise ImportError("Pillow should be installed in order to use the StableDiffusionTool.")

        super().__init__()

        self.stable_diffusion = self.default_stable_diffusion_checkpoint

        self.device = device
        self.hub_kwargs = hub_kwargs

    def setup(self):
        if self.device is None:
            self.device = get_default_device()

        self.pipeline = DiffusionPipeline.from_pretrained(self.stable_diffusion)

        self.pipeline.to(self.device)
        if self.device.type == "cuda":
            self.pipeline.to(torch_dtype=torch.float16)

        self.is_initialized = True

    def __call__(self, image, prompt):
        if not self.is_initialized:
            self.setup()

        negative_prompt = "low quality, bad quality, deformed, low resolution"
        added_prompt = " , highest quality, highly realistic, very high resolution"

        return self.pipeline(
            prompt + added_prompt,
            image,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
        ).images[0]