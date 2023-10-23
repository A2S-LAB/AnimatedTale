from diffusers import DiffusionPipeline
import torch
import os
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
"""
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
https://github.com/Stability-AI/generative-models

"""

def makeBackground(_prompt):
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    
    base.to("cuda")
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    n_steps = 40
    high_noise_frac = 0.8
    prompt = _prompt

    # 1024 x 1024로 사이즈는 고정인 듯 함
    # 한 장당 10초 정도 걸리는 듯
    for i in range(4):
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]
        
        dir_path = 'backgrounds'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filename = f'{dir_path}/background_{i}.jpg'
        image.save(filename)

if __name__ == '__main__':
    prompt = "a cute fennec fox, as if a kid drew, simple, in stlye of croquis"
    print(torch.cuda.is_available())
    makeBackground(prompt)