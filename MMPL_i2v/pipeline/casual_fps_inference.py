from tqdm import tqdm
from typing import List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanFPSWrapper

torch.backends.cuda.preferred_linalg_library("magma")
import os
import copy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def move_kv_cache_to_cpu(kv_cache):
    """Move all tensors in *kv_cache* to CPU."""
    tensor_keys = ("k", "v", "global_end_index", "local_end_index")
    for block in kv_cache:
        for k in tensor_keys:
            if k in block and isinstance(block[k], torch.Tensor):
                block[k] = block[k].to("cpu", non_blocking=True)
    return kv_cache


def move_kv_cache_to_gpu(kv_cache, device="cuda:0"):
    """Move all tensors in *kv_cache* to specified GPU device."""
    tensor_keys = ("k", "v", "global_end_index", "local_end_index")
    for block in kv_cache:
        for k in tensor_keys:
            if k in block and isinstance(block[k], torch.Tensor):
                block[k] = block[k].to(device, non_blocking=True)
    return kv_cache


class CausalFPSInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            device_cond="cuda:0",
            device_uncond="cuda:0",
            save="latents_chunk1.pt",
    ):
        super().__init__()

        # self.device_cond = "cuda:1"     
        # self.device_uncond = "cuda:0" 
        self.need_wait=False
        self.save = save
        self.device_cond = device_cond
        self.device_uncond = device_uncond

        print(torch.cuda.device_count())

        # Step 1: Initialize all models
        self.generator_cond = WanFPSWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        for module in [self.generator_cond.model]:
            for param in module.parameters():
                param.requires_grad = False
        self.generator_cond = self.generator_cond.to(self.device_cond)
        self.generator_cond.model.num_frame_per_block = 1

        torch.cuda.empty_cache()
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        for module in [self.text_encoder]:
            for param in module.parameters():
                param.requires_grad = False
        self.vae = WanVAEWrapper() if vae is None else vae
        for module in [self.vae]:
            for param in module.parameters():
                param.requires_grad = False

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep

        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = 40 # 30 for 1.3B, 40 for 14B
        self.frame_seq_length = 1560

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = 1
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = -1

        '''----------- add new noise on previous frames ------------'''
        self.ddpm_scheduler = self.generator_cond.get_scheduler()
        self.image_or_video_shape = [1, 1, 16, 60, 104] # 1 for next frame
        self.ddpm_index = self._get_timestep(
            980,
            self.num_train_timesteps,
            self.image_or_video_shape[0],
            self.image_or_video_shape[1],
            1,
            uniform_timestep=False
        )
        self.ddpm_batch_size, self.ddpm_num_frame = self.image_or_video_shape[:2]
        self.ddpm_scheduler.timesteps = self.ddpm_scheduler.timesteps.to(self.ddpm_index.device)
        self.ddmp_timestep = self.ddpm_scheduler.timesteps[self.ddpm_index]
        self.ddmp_timestep = self.ddmp_timestep + 1000
        print('self.ddmp_timestep: ', self.ddmp_timestep)
        '''----------- add new noise on previous frames ------------'''

        print(f"KV inference with {self.num_frame_per_block} frames per block")

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device_cond,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device_cond,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                pass
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames # add the initial latent frames

        with torch.no_grad():
            self.text_encoder.to((self.device_cond))
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )

            self.text_encoder.to((self.device_uncond))
            unconditional_dict = self.text_encoder(
                text_prompts=[self.args.negative_prompt] * len(text_prompts)
            )  

        self.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=self.device_cond)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=self.device_cond)
                self.kv_cache_pos[block_index]["attention_vis_index"] = []

                self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=self.device_uncond)
                self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=self.device_uncond)
                self.kv_cache_neg[block_index]["attention_vis_index"] = []
                    

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0

        # Step 3: Temporal denoising loop

        ''' for anchor next-chunk generation ''' 
        clean_steps = [0, 1, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 4, 4, 4, 4, 4, 4, 2, 2] # important for image2video
        all_num_frames = [1, 1, 7, 6, 6]
        result = [[i for i, v in enumerate(clean_steps) if v == target] for target in range(5)]

        global_chunk_index = 0

        print('noise:', noise.shape)
        print('num_input_frames: ', num_input_frames)
        print('num_blocks:', num_blocks)
        print('cache_start_frame: ', cache_start_frame)
        print('all_num_frames: ', all_num_frames)
        print('-' * 30)

        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        for current_num_frames in all_num_frames:
            
            if (initial_latent is None) or (global_chunk_index != 0):
                
                noise_position = result[global_chunk_index]
                noisy_input = noise[
                    :, result[global_chunk_index]]

                if current_num_frames != noisy_input.shape[1]: continue

                # update the start and end frame indices
                current_start_frame = result[global_chunk_index]
                cache_start_frame = result[global_chunk_index]
                latents = noisy_input

                print('noisy_input: ', noisy_input.shape)
                print('current_start_frame: ', current_start_frame)
                print('cache_start_frame: ', cache_start_frame)
                print('global_chunk_index: ', global_chunk_index)
                print('attention_vis_index_pos:', self.kv_cache_pos[0]["attention_vis_index"])
                print('attention_vis_index_neg:', self.kv_cache_neg[0]["attention_vis_index"])
                print('current_num_frames: ', current_num_frames)

                # if current_num_frames != latents.shape[1]: continue

                # Step 3.1: Spatial denoising loop
                sample_scheduler = self._initialize_sample_scheduler(noise)
                for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                    latent_model_input = latents
                    timestep = t * torch.ones(
                        [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                    )

                    with torch.no_grad():
                        # print(latent_model_input.shape)
                        # print(timestep.shape)
                        # print(current_num_frames)
                        flow_pred_cond, _ = self.generator_cond(
                            noisy_image_or_video=latent_model_input.to(self.device_cond),
                            conditional_dict=conditional_dict,
                            timestep=timestep.to(self.device_cond),
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=[i * 1560 for i in current_start_frame],
                            cache_start=[i * 1560 for i in cache_start_frame],
                        )

                        flow_pred_uncond, _ = self.generator_cond(
                            noisy_image_or_video=latent_model_input.to(self.device_uncond),
                            conditional_dict=unconditional_dict,
                            timestep=timestep.to(self.device_uncond),
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=[i * 1560 for i in current_start_frame],
                            cache_start=[i * 1560 for i in cache_start_frame],
                        )

                        flow_pred = flow_pred_uncond.to(self.device_cond) + self.args.guidance_scale * (
                            flow_pred_cond - flow_pred_uncond.to(self.device_cond))

                        temp_x0 = sample_scheduler.step(
                            flow_pred.to(self.device_cond),
                            t.to(self.device_cond),
                            latents.to(self.device_cond),
                            return_dict=False)[0]
                        latents = temp_x0

                # Step 3.2: record the model's output
                output = output.to(self.device_cond)
                output[:, result[global_chunk_index]] = latents.to(self.device_cond)

                if global_chunk_index == 2:
                    # save_latents = torch.cat([output[:, :1], latents], dim=1)
                    save_latents = torch.cat([output[:, :1], output[:, -2:]], dim=1)
                    torch.save(save_latents, self.save)

                # Step 3.3: rerun with timestep zero to update KV cache using clean context
                self.generator_cond(
                    noisy_image_or_video=latents.to(self.device_cond),
                    conditional_dict=conditional_dict,
                    timestep=timestep.to(self.device_cond) * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=[i * 1560 for i in current_start_frame],
                    cache_start=[i * 1560 for i in cache_start_frame],
                )
                self.generator_cond(
                    noisy_image_or_video=latents.to(self.device_uncond),
                    conditional_dict=unconditional_dict,
                    timestep=timestep.to(self.device_uncond) * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=[i * 1560 for i in current_start_frame],
                    cache_start=[i * 1560 for i in cache_start_frame],
                )

                global_chunk_index += 1
            
            elif initial_latent is not None and global_chunk_index == 0:
                
                if initial_latent.shape[1] > 1:
                    for step in range(2):
                        print('step:', step)
                        latents = initial_latent[:, step:step+1]
                        current_start_frame = result[step]
                        cache_start_frame = result[step]

                        timestep = 0 * torch.ones(
                            [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                        )

                        self.generator_cond(
                            noisy_image_or_video=latents.to(self.device_cond),
                            conditional_dict=conditional_dict,
                            timestep=timestep.to(self.device_cond) * 0,
                            kv_cache=self.kv_cache_pos,
                            crossattn_cache=self.crossattn_cache_pos,
                            current_start=[i * 1560 for i in current_start_frame],
                            cache_start=[i * 1560 for i in cache_start_frame],
                        )
                        self.generator_cond(
                            noisy_image_or_video=latents.to(self.device_uncond),
                            conditional_dict=unconditional_dict,
                            timestep=timestep.to(self.device_uncond) * 0,
                            kv_cache=self.kv_cache_neg,
                            crossattn_cache=self.crossattn_cache_neg,
                            current_start=[i * 1560 for i in current_start_frame],
                            cache_start=[i * 1560 for i in cache_start_frame],
                        )

                        output = output.to(self.device_cond)
                        output[:, result[step]] = initial_latent[:, step:step+1].to(self.device_cond)
                    global_chunk_index = 2
                    print('segment connect finished ! ')
                else:
                    step = 0
                    latents = initial_latent[:, step:step+1]
                    current_start_frame = result[step]
                    cache_start_frame = result[step]

                    timestep = 0 * torch.ones(
                        [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                    )

                    self.generator_cond(
                        noisy_image_or_video=latents.to(self.device_cond),
                        conditional_dict=conditional_dict,
                        timestep=timestep.to(self.device_cond) * 0,
                        kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=[i * 1560 for i in current_start_frame],
                        cache_start=[i * 1560 for i in cache_start_frame],
                    )
                    self.generator_cond(
                        noisy_image_or_video=latents.to(self.device_uncond),
                        conditional_dict=unconditional_dict,
                        timestep=timestep.to(self.device_uncond) * 0,
                        kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=[i * 1560 for i in current_start_frame],
                        cache_start=[i * 1560 for i in cache_start_frame],
                    )

                    output = output.to(self.device_cond)
                    output[:, result[step]] = initial_latent.to(self.device_cond)
                    
                    global_chunk_index += 1


        # Step 4: Decode the output
        torch.cuda.empty_cache()
        self.vae.to(self.device_cond)
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_pos = []
        kv_cache_neg = []

        # Use the default KV cache size
        # kv_cache_size = 32760
        kv_cache_size = 32760 - (6 * 1560)

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=self.device_cond), # 12 for 1.3B, 40 for 14B
                "v": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=self.device_cond),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device_cond),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device_cond),
                "attention_vis_index": [],
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=self.device_uncond),
                "v": torch.zeros([batch_size, kv_cache_size, 40, 128], dtype=dtype, device=self.device_uncond),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device_uncond),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device_uncond),
                "attention_vis_index": [],
            })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, 40, 128], dtype=dtype, device=self.device_cond),    # 12 for 1.3B, 40 for 14B
                "v": torch.zeros([batch_size, 512, 40, 128], dtype=dtype, device=self.device_cond),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, 40, 128], dtype=dtype, device=self.device_uncond),
                "v": torch.zeros([batch_size, 512, 40, 128], dtype=dtype, device=self.device_uncond),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
