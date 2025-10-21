import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalFPSInferencePipeline,
    CausalInferencePipeline
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
import os, time, threading

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline_chunk_1 = CausalFPSInferencePipeline(config, device=device, device_cond="cuda:0", device_uncond="cuda:0", save="latents_chunk1.pt")
    pipeline_chunk_2 = CausalFPSInferencePipeline(config, device=device, device_cond="cuda:1", device_uncond="cuda:1", save="latents_chunk2.pt")
    pipeline_chunk_3 = CausalFPSInferencePipeline(config, device=device, device_cond="cuda:2", device_uncond="cuda:2", save="latents_chunk3.pt")
    pipeline_chunk_4 = CausalFPSInferencePipeline(config, device=device, device_cond="cuda:3", device_uncond="cuda:3", save="latents_chunk4.pt")

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    weight_key = 'generator_ema' if args.use_ema else 'generator'

    pipeline_chunk_1.generator_cond.load_state_dict(state_dict[weight_key])
    pipeline_chunk_2.generator_cond.load_state_dict(state_dict[weight_key])
    pipeline_chunk_3.generator_cond.load_state_dict(state_dict[weight_key])
    pipeline_chunk_4.generator_cond.load_state_dict(state_dict[weight_key])

pipeline_chunk_1 = pipeline_chunk_1.to(dtype=torch.bfloat16)
pipeline_chunk_1.independent_first_frame = False

pipeline_chunk_2 = pipeline_chunk_2.to(dtype=torch.bfloat16)
pipeline_chunk_2.independent_first_frame = False

pipeline_chunk_3 = pipeline_chunk_3.to(dtype=torch.bfloat16)
pipeline_chunk_3.independent_first_frame = False

pipeline_chunk_4 = pipeline_chunk_4.to(dtype=torch.bfloat16)
pipeline_chunk_4.independent_first_frame = False

# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output

set_seed(args.seed)

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    set_seed(args.seed)
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    if args.i2v:
        # For image-to-video, batch contains image and caption
        prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16 
        ) # [B, F, C, H, W]

    # print('text prompt: ', prompts)
    
    def wait_for_file(path, interval=1.0):
        while not os.path.exists(path):
            time.sleep(interval)

    def generate_video_chunk(pipeline, sampled_noise, prompts, chunk_idx, args, wait_file=None):
        if wait_file:
            pipeline.vae = pipeline.vae.to("cuda")
            print(f"[Chunk {chunk_idx}] Waiting for {wait_file}...")
            while not os.path.exists(wait_file):
                time.sleep(1.0)
            print(f"[Chunk {chunk_idx}] Detected {wait_file}, loading as initial_latent...")
            initial_latent = torch.load(wait_file, map_location="cpu")
            os.remove(wait_file)
            print(f"[Chunk {chunk_idx}] Removed {wait_file}")

            latents_chunk_1 = initial_latent.to("cuda").to(torch.bfloat16)
            mask_latents_chunk_1 = torch.zeros(1, 21, 16, 60, 104).to("cuda").to(torch.bfloat16)
            mask_latents_chunk_1[:, 0:1] = latents_chunk_1[:, 0:1]
            mask_latents_chunk_1[:, 1:2] = latents_chunk_1[:, -2:-1]
            mask_latents_chunk_1[:, 2:4] = latents_chunk_1[:, -2:]

            vid_chunk_1 = pipeline.vae.decode_to_pixel(mask_latents_chunk_1).to("cuda").to(torch.bfloat16)
            vid_chunk_1 = (vid_chunk_1 * 0.5 + 0.5).clamp(0, 1).to("cuda").to(torch.bfloat16)

            vid_test_chunk_1 = torch.zeros_like(vid_chunk_1).to("cuda").to(torch.bfloat16)
            vid_test_chunk_1[:, 0:5] = vid_chunk_1[:, 8:13]
            vid_test_normalized_chunk_1 = vid_test_chunk_1 * 2.0 - 1.0
            vid_test_rearranged_chunk_1 = rearrange(vid_test_normalized_chunk_1, "b t c h w -> b c t h w")
            vid_test_latents_chunk_1 = pipeline.vae.encode_to_latent(vid_test_rearranged_chunk_1)
            initial_latent = vid_test_latents_chunk_1[:, :2].to(torch.bfloat16)
        else:
            initial_latent = None
        
        pipeline.vae = pipeline.vae.to("cpu")
        video, latents = pipeline.inference(
            noise=sampled_noise.to(torch.bfloat16),
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
        )
        pipeline.vae = pipeline.vae.to("cuda")

        # Save video
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_save = (current_video * 255.0).clamp(0, 255)

        for i in range(args.num_samples):
            output_path = os.path.join(
                args.output_folder, f"{prompts[0][:100]}-chunk{chunk_idx}-sample{i}.mp4"
            )
            write_video(output_path, video_save[i], fps=16)
            print(f"[Chunk {chunk_idx}] Saved video to {output_path}")

    t1 = threading.Thread(target=generate_video_chunk, args=(
        pipeline_chunk_1, sampled_noise, prompts, 1, args, None))

    sampled_noise = torch.randn(
        [args.num_samples, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
    )
    t2 = threading.Thread(target=generate_video_chunk, args=(
        pipeline_chunk_2, sampled_noise, prompts, 2, args,
        "latents_chunk1.pt"))
    
    sampled_noise = torch.randn(
        [args.num_samples, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
    )
    t3 = threading.Thread(target=generate_video_chunk, args=(
        pipeline_chunk_3, sampled_noise, prompts, 3, args,
        "latents_chunk2.pt"))

    sampled_noise = torch.randn(
        [args.num_samples, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
    )
    t4 = threading.Thread(target=generate_video_chunk, args=(
        pipeline_chunk_4, sampled_noise, prompts, 4, args,
        "latents_chunk3.pt"))
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()