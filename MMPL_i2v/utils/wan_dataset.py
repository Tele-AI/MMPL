import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import lightning as pl
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
import random

START = 0

def discover_paths(base_dir, metadata_dir):
    """
    Automatically discover all subdirectories under base_dir
    and the corresponding CSV files under metadata_dir.
    """
    base_paths = []
    metadata_paths = []
    
    # Get all subdirectories under base_dir
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort()  # Sort to ensure consistency
    
    for subdir in subdirs:
        base_path = os.path.join(base_dir, subdir)
        # Look for the corresponding CSV file
        csv_file = os.path.join(metadata_dir, f"{subdir}.csv")
        
        if os.path.exists(csv_file):
            base_paths.append(base_path)
            metadata_paths.append(csv_file)
            print(f"✅ Found pair: {base_path} -> {csv_file}")
        else:
            print(f"⚠️  Warning: No CSV file found for {subdir} at {csv_file}")
    
    return base_paths, metadata_paths


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):

        if isinstance(base_path, str):
            base_path = [base_path]
        if isinstance(metadata_path, str):
            metadata_path = [metadata_path]
        self.path = []
        self.text = []
        for bp, mp in zip(base_path, metadata_path):
            metadata = pd.read_csv(mp)
            column_name = "file_name" if "file_name" in metadata.columns else "file_path"
            self.path.extend([os.path.join(bp, fname) for fname in metadata[column_name]])
            self.text.extend(metadata["text"].to_list())
        print(len(self.path), len(self.text))

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        fps = reader.get_meta_data().get("fps", None)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames

    # def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
    #     reader = imageio.get_reader(file_path)
    #     fps = reader.get_meta_data().get("fps", None)
    #     total_frames = reader.count_frames()

    #     if fps is None or fps <= 16:
    #         interval = 1
    #     else:
    #         interval = int((fps - 1) // 16 + 1)

    #     start_frame_id = START
        
    #     required_span = start_frame_id + (num_frames - 1) * interval
    #     if total_frames <= required_span:
    #         reader.close()
    #         return None

    #     frames = []
    #     first_frame = None
    #     for frame_id in range(num_frames):
    #         real_frame_id = start_frame_id + frame_id * interval
    #         frame = reader.get_data(real_frame_id)
    #         frame = Image.fromarray(frame)
    #         frame = self.crop_and_resize(frame)
    #         if first_frame is None:
    #             first_frame = frame
    #         frame = frame_process(frame)
    #         frames.append(frame)
    #     reader.close()

    #     frames = torch.stack(frames, dim=0)
    #     frames = rearrange(frames, "T C H W -> C T H W")
    #     return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        else:
            video = self.load_video(path)

        if video is None:
            video = torch.zeros((3, 81, 480, 832), dtype=torch.float32)

        if self.is_i2v:
            video, first_frame = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
        else:
            data = {"text": text, "video": video, "path": path}
        return data
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.created_dirs = set()  
        
    def test_step(self, batch, batch_idx):
        text, video, path = batch["text"][0], batch["video"], batch["path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, None, num_frames, height, width)
            else:
                image_emb = {}
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb}

            output_path = path.replace("merged_videos", f"merged_videos_latents") + ".tensors.pth"
            output_dir = os.path.dirname(output_path)
            if output_dir not in self.created_dirs:
                os.makedirs(output_dir, exist_ok=True)
                self.created_dirs.add(output_dir)

            if video.sum() == 0: 
                pass
            else:
                torch.save(data, output_path)

def cycle(dl):
    while True:
        for data in dl:
            yield data

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, pth_paths, metadata_paths):
        # Ensure inputs are lists
        if isinstance(pth_paths, str):
            pth_paths = [pth_paths]
        if isinstance(metadata_paths, str):
            metadata_paths = [metadata_paths]
        assert len(pth_paths) == len(metadata_paths), "Mismatch between pth_paths and metadata_paths"

        self.path = []

        # Load each dataset source
        for pth_path, metadata_path in zip(pth_paths, metadata_paths):
            metadata = pd.read_csv(metadata_path)
            print(f"🔍 {len(metadata)} entries found in {metadata_path}")
            
            if "file_name" in metadata.columns:
                name_column = "file_name"
            elif "file_path" in metadata.columns:
                name_column = "file_path"

            # Construct full tensor paths and check for file existence
            for file_name in metadata[name_column]:
                tensor_path = os.path.join(pth_path, file_name) + ".tensors.pth"
                self.path.append(tensor_path)

        print(f"✅ Total valid tensor files loaded: {len(self.path)}")
        assert len(self.path) > 0, "No valid tensor files found."

    def __getitem__(self, index):
        # Load tensor from file
        path = self.path[index]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data

    def __len__(self):
        # Total number of available tensor files
        return len(self.path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/gemini/space/xxz/datasets/2-EnviromentData/merged_videos", help="Root path to video/image files")
    parser.add_argument("--metadata_path", type=str, default="/gemini/space/xxz/datasets/2-EnviromentData/merged_videos.csv", help="File listing all video paths")
    parser.add_argument("--text_encoder_path", type=str, default="/gemini/space/xxz/Self-Forcing/wan_models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--vae_path", type=str, default="/gemini/space/xxz/Self-Forcing/wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth")
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size_height", type=int, default=34)
    parser.add_argument("--tile_size_width", type=int, default=34)
    parser.add_argument("--tile_stride_height", type=int, default=18)
    parser.add_argument("--tile_stride_width", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--output_path", type=str, default="./output")
    return parser.parse_args()

def data_process(args):
    print('meta_path: ', args.metadata_path)
    print('base_path: ', args.base_path)

    # base_paths, metadata_paths = discover_paths(args.base_path, args.metadata_path)
    base_paths, metadata_paths = [args.base_path], [args.metadata_path]

    print(f'Found {len(base_paths)} datasets:')
    for base_path, metadata_path in zip(base_paths, metadata_paths):
        print(f'  base_path: {base_path}')
        print(f'  metadata_path: {metadata_path}')

    dataset = TextVideoDataset(
        base_paths,
        metadata_paths,
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    data_process(args)
