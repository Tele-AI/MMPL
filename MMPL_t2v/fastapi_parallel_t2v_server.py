from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
import torch
import argparse
import traceback
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange
import uuid
from tqdm import tqdm
import gc
import torch.distributed as dist
import uvicorn
import requests
import datetime
import json
import time
import subprocess
import threading
from PIL import Image, PngImagePlugin
from moviepy import VideoFileClip
import boto3
from botocore.client import Config
import collections
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "0925_new_mmpl/MMPL_9.25_2/MMPL_t2v"))

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalFPSInferencePipeline,
    CausalInferencePipeline,
)
from utils.misc import set_seed

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 枚举和常量 ====================
class TaskStatus(Enum):
    """任务状态枚举"""
    NOT_STARTED = "0"
    PROCESSING = "1"
    SUCCESS = "2"
    FAILED = "3"

class ResponseCode(Enum):
    """响应代码枚举"""
    SUCCESS = 10000
    NOT_FOUND = 10404
    SERVER_ERROR = 10903

# ==================== 配置类 ====================
@dataclass
class ParallelServerConfig:
    """并行服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8001
    output_folder: str = "videos/parallel_fps"
    use_ema: bool = False
    num_output_frames: int = 21
    num_chunks: int = 4  # 并行处理的片段数
    gpu_devices: List[str] = None  # GPU设备列表
    use_text_expansion: bool = True
    text_expansion_url: str = "http://10.127.16.1:8998/expand"
    prompt_log_file: str = "/app/prompt_extend.txt"
    temp_latents_dir: str = "temp_latents"  # 临时latents文件目录

@dataclass
class S3Config:
    """S3配置"""
    access_key: str = 'wzv-mS2zb75TyETjrU'
    secret_key: str = 'TPDo-N55VEI6Iyax2E'
    endpoint: str = 'http://ops-public-ceph.teleagi.in:8081'
    region: str = 'YOUR_REGION'
    bucket: str = 'telestudio-bucket'

# ==================== 数据模型 ====================
class ParallelVideoRequest(BaseModel):
    """并行视频生成请求模型"""
    prompt: str
    num_chunks: int = 4  # 生成的视频片段数量
    seed: int = 0
    use_expansion: bool = True
    callback_url: Optional[str] = None
    seqid: Optional[str] = None

class TaskSearchItem(BaseModel):
    """任务查询请求模型"""
    seqid: str

class ParallelVideoResponse(BaseModel):
    """并行视频生成响应模型"""
    task_id: str
    video_paths: List[str] = []  # 多个视频片段路径
    original_prompt: str
    expanded_prompt: Optional[str] = None
    seqid: Optional[str] = None
    flag: int = 0
    status: str = "1"
    num_chunks: int = 4

class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: Optional[str] = None
    seqid: Optional[str] = None
    code: int
    message: str
    flag: int
    status: str
    data: Dict

# ==================== 工具类 ====================
class MediaMetadataHandler:
    """媒体元数据处理器"""
    
    METADATA_TEMPLATE = {
        "Label": "1",
        "ContentProducer": "TeleStudio",
        "ProduceID": "",
        "ReservedCode1": "",
        "ContentPropagator": "TeleStudio",
        "PropagateID": "",
        "ReservedCode2": ""
    }

    @classmethod
    def write_png_metadata(cls, seqid: str, input_png: str, output_png: str, keyword: str = "AIGC") -> None:
        """将元数据写入PNG图片"""
        try:
            metadata = cls.METADATA_TEMPLATE.copy()
            metadata['ProduceID'] = seqid
            metadata['PropagateID'] = seqid
            
            img = Image.open(input_png)
            meta = PngImagePlugin.PngInfo()
            text_value = json.dumps(metadata)
            meta.add_text(keyword, text_value)
            img.save(output_png, pnginfo=meta)
            logger.info(f"PNG元数据已写入: {output_png}")
        except Exception as e:
            logger.error(f"写入PNG元数据失败: {e}")
            raise

    @classmethod
    def write_video_metadata(cls, seqid: str, input_video: str, output_video: str) -> None:
        """将元数据写入视频文件"""
        try:
            metadata = cls.METADATA_TEMPLATE.copy()
            metadata['ProduceID'] = seqid
            metadata['PropagateID'] = seqid
            metadata_str = json.dumps(metadata)
            
            command = [
                'ffmpeg', '-y', '-i', input_video,
                '-metadata', f'AIGC={metadata_str}',
                '-movflags', 'use_metadata_tags',
                '-c', 'copy',
                output_video
            ]
            subprocess.run(command, check=True)
            logger.info(f"视频元数据已写入: {output_video}")
        except Exception as e:
            logger.error(f"写入视频元数据失败: {e}")
            raise

class S3Uploader:
    """S3文件上传器"""
    
    def __init__(self, config: S3Config):
        self.config = config
        self._client = None
    
    @property
    def client(self):
        """懒加载S3客户端"""
        if self._client is None:
            config = Config(connect_timeout=30, read_timeout=30)
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                endpoint_url=self.config.endpoint,
                use_ssl=False,
                config=config
            )
        return self._client
    
    def upload_file(self, local_path: str, s3_key: str) -> str:
        """上传文件到S3并返回URL"""
        try:
            self.client.upload_file(
                Bucket=self.config.bucket,
                Key=s3_key,
                Filename=local_path,
                ExtraArgs={'ACL': 'public-read'}
            )
            
            url = self.client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': self.config.bucket, 'Key': s3_key},
                ExpiresIn=24 * 3600
            )
            return url.split('?')[0]
        except Exception as e:
            logger.error(f"S3上传失败: {e}")
            raise
    
    def upload_with_retry(self, seqid: str, local_path: str, s3_key: str, max_retries: int = 3) -> str:
        """带重试机制的文件上传"""
        logger.info(f"【{seqid}】开始上传文件: {local_path}")
        
        for attempt in range(max_retries):
            try:
                url = self.upload_file(local_path, s3_key)
                logger.info(f"【{seqid}】文件上传成功: {url}")
                
                # 上传成功后删除本地文件
                if os.path.exists(local_path):
                    os.remove(local_path)
                
                return url
            except Exception as e:
                logger.warning(f"【{seqid}】上传失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"【{seqid}】文件上传重试{max_retries}次均失败")
                    return ""
        return ""

class TaskStorage:
    """任务存储管理器"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.tasks = collections.OrderedDict()
    
    def add_task(self, key: str, value: str) -> None:
        """添加任务"""
        if key in self.tasks:
            del self.tasks[key]
        
        if len(self.tasks) >= self.max_size:
            removed_key, _ = self.tasks.popitem(last=False)
            logger.info(f"移除最早的任务: {removed_key}")
        
        self.tasks[key] = value
        logger.info(f"添加任务: {key}, 当前任务数量: {len(self.tasks)}")
    
    def get_task(self, key: str) -> Optional[str]:
        """获取任务"""
        return self.tasks.get(key)

class TextExpander:
    """文本扩写服务"""
    
    def __init__(self, url: str, log_file: str):
        self.url = url
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def expand(self, prompt: str) -> str:
        """扩写文本"""
        try:
            response = requests.post(self.url, json={"prompt": prompt})
            
            if response.status_code == 200:
                result = response.json()
                expanded_prompt = result["expanded"]
                logger.info(f"原始提示词: '{prompt}'")
                logger.info(f"扩写后提示词: '{expanded_prompt}'")
                self._save_to_log(expanded_prompt)
                return expanded_prompt
            else:
                logger.warning(f"文本扩写请求失败: {response.status_code}")
                return prompt
        except Exception as e:
            logger.error(f"文本扩写过程出错: {e}")
            return prompt
    
    def _save_to_log(self, expanded_prompt: str) -> None:
        """保存扩写结果到日志文件"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{expanded_prompt}\n")
        except Exception as e:
            logger.error(f"保存提示词到文件时出错: {e}")

class CallbackHandler:
    """回调处理器"""
    
    @staticmethod
    def execute_callback(callback_url: str, seqid: str, code: int, message: str, 
                        flag: int, video_urls: List[str], cover_images: List[str], 
                        text_en: str, max_retries: int = 3) -> None:
        """执行回调"""
        if not callback_url:
            logger.info(f"【{seqid}】未提供回调URL，跳过回调")
            return
        
        req_data = {
            "seqid": seqid,
            "code": code,
            "message": message,
            "flag": flag,
            "data": {
                "video": video_urls,
                "cover_image": cover_images,
                "text_en": text_en
            }
        }
        
        logger.info(f"【{seqid}】回调请求体: {req_data}")
        
        for attempt in range(max_retries):
            try:
                headers = {'Content-type': 'application/json'}
                response = requests.post(
                    url=callback_url, 
                    data=json.dumps(req_data), 
                    headers=headers
                )
                
                if response.status_code == 200:
                    logger.info(f"【{seqid}】回调成功: {response.text}")
                    return
                else:
                    logger.warning(f"【{seqid}】回调失败 (尝试 {attempt + 1}/{max_retries}): {response.status_code}")
            except Exception as e:
                logger.warning(f"【{seqid}】回调异常 (尝试 {attempt + 1}/{max_retries}): {e}")
        
        logger.error(f"【{seqid}】回调重试{max_retries}次仍失败")

class VideoProcessor:
    """视频处理器"""
    
    @staticmethod
    def extract_first_frame(video_path: str, output_path: str) -> bool:
        """提取视频首帧"""
        try:
            video = VideoFileClip(video_path)
            first_frame = video.get_frame(0)
            image = Image.fromarray(first_frame)
            image.save(output_path)
            video.close()
            logger.info(f"视频首帧提取成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"提取首帧失败: {e}")
            return False

# ==================== 服务类 ====================
class ParallelModelManager:
    """并行模型管理器"""
    
    def __init__(self, config: ParallelServerConfig):
        self.config = config
        self.model_config = None
        self.pipelines = {}  # 存储多个pipeline
        self.device = None
        self.gpu_devices = config.gpu_devices or ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    
    def load_models(self, config_path: str, checkpoint_path: str, use_ema: bool = False) -> bool:
        """加载多个并行模型"""
        try:
            self.device = torch.device("cuda")
            set_seed(0)
            
            torch.set_grad_enabled(False)
            
            # 加载配置
            config = OmegaConf.load(config_path)
            default_config = OmegaConf.load("configs/default_config.yaml")
            self.model_config = OmegaConf.merge(default_config, config)
            
            # 为每个GPU创建pipeline
            for i, gpu_device in enumerate(self.gpu_devices[:self.config.num_chunks]):
                chunk_idx = i + 1
                logger.info(f"初始化Pipeline Chunk {chunk_idx} 在 {gpu_device}")
                
                pipeline = CausalFPSInferencePipeline(
                    self.model_config, 
                    device=self.device,
                    device_cond=gpu_device,
                    device_uncond=gpu_device,
                    save=f"latents_chunk{chunk_idx}.pt"
                )
                
                # 加载检查点
                if checkpoint_path:
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                    weight_key = 'generator_ema' if use_ema else 'generator'
                    pipeline.generator_cond.load_state_dict(state_dict[weight_key])
                
                # 转换数据类型
                pipeline = pipeline.to(dtype=torch.bfloat16)
                pipeline.independent_first_frame = False
                
                self.pipelines[f"chunk_{chunk_idx}"] = pipeline
                logger.info(f"Pipeline Chunk {chunk_idx} 加载完成")
            
            logger.info(f"所有{len(self.pipelines)}个并行模型加载完成")
            return True
            
        except Exception as e:
            logger.error(f"并行模型加载失败: {e}")
            traceback.print_exc()
            return False
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return len(self.pipelines) > 0

class ParallelVideoGenerationService:
    """并行视频生成服务"""
    
    def __init__(self, config: ParallelServerConfig, s3_config: S3Config):
        self.config = config
        self.model_manager = ParallelModelManager(config)
        self.task_storage = TaskStorage()
        self.s3_uploader = S3Uploader(s3_config)
        self.text_expander = TextExpander(config.text_expansion_url, config.prompt_log_file)
        
        # 创建输出目录和临时目录
        os.makedirs(config.output_folder, exist_ok=True)
        os.makedirs(config.temp_latents_dir, exist_ok=True)
    
    def load_models(self, config_path: str, checkpoint_path: str, use_ema: bool = False) -> bool:
        """加载模型"""
        return self.model_manager.load_models(config_path, checkpoint_path, use_ema)
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model_manager.is_loaded()
    
    async def generate_parallel_video_task(self, request: ParallelVideoRequest, task_id: str) -> None:
        """生成并行视频任务"""
        start_time = time.time()
        timestamp = str(int(start_time * 1000))
        seqid = request.seqid or task_id
        
        # 初始化任务状态
        self._update_task_status(
            task_id, seqid, TaskStatus.PROCESSING, 
            ResponseCode.SUCCESS, "任务执行中", [], [], request.prompt
        )
        
        try:
            # 创建任务输出目录
            task_output_folder = os.path.join(self.config.output_folder, task_id)
            os.makedirs(task_output_folder, exist_ok=True)
            
            # 设置随机种子
            set_seed(request.seed)
            
            # 文本扩写
            expanded_prompt = request.prompt
            if self.config.use_text_expansion and request.use_expansion:
                expanded_prompt = self.text_expander.expand(request.prompt)
            
            # 并行生成视频片段
            video_urls, cover_images = await self._generate_parallel_videos(
                expanded_prompt, request.num_chunks, task_id, timestamp, 
                task_output_folder, seqid
            )
            
            # 更新成功状态
            self._update_task_status(
                task_id, seqid, TaskStatus.SUCCESS, 
                ResponseCode.SUCCESS, "任务执行成功", video_urls, cover_images, expanded_prompt
            )
            
            # 执行成功回调
            if request.callback_url:
                CallbackHandler.execute_callback(
                    request.callback_url, seqid, ResponseCode.SUCCESS.value, 
                    "任务执行成功", 1, video_urls, cover_images, expanded_prompt
                )
            
            logger.info(f"【{seqid}】并行任务完成，耗时: {time.time() - start_time:.2f}秒")
            
        except Exception as e:
            error_msg = f"并行算法执行异常: {str(e)}"
            logger.error(f"【{seqid}】{error_msg}")
            traceback.print_exc()
            
            # 更新失败状态
            self._update_task_status(
                task_id, seqid, TaskStatus.FAILED, 
                ResponseCode.SERVER_ERROR, error_msg, [], [], expanded_prompt
            )
            
            # 执行失败回调
            if request.callback_url:
                CallbackHandler.execute_callback(
                    request.callback_url, seqid, ResponseCode.SERVER_ERROR.value, 
                    error_msg, 0, [], [], expanded_prompt
                )
    
    async def _generate_parallel_videos(self, prompt: str, num_chunks: int, task_id: str, 
                                       timestamp: str, output_folder: str, seqid: str) -> tuple:
        """并行生成视频片段"""
        prompts = [prompt]
        
        # 生成初始噪声
        sampled_noise = torch.randn(
            [1, self.config.num_output_frames, 16, 60, 104], 
            device="cuda", 
            dtype=torch.bfloat16
        )
        
        # 创建线程列表
        threads = []
        chunk_results = {}
        
        def generate_video_chunk(pipeline, noise, prompts, chunk_idx, wait_file=None):
            """生成单个视频片段的线程函数"""
            try:
                logger.info(f"【{seqid}】开始生成Chunk {chunk_idx}")
                
                initial_latent = None
                if wait_file:
                    # 等待前一个chunk的latents文件
                    wait_file_path = os.path.join(self.config.temp_latents_dir, wait_file)
                    logger.info(f"【{seqid}】Chunk {chunk_idx} 等待文件: {wait_file_path}")
                    
                    while not os.path.exists(wait_file_path):
                        time.sleep(1.0)
                    
                    logger.info(f"【{seqid}】Chunk {chunk_idx} 检测到文件，开始加载...")
                    
                    # 处理前一个chunk的latents
                    pipeline.vae = pipeline.vae.to("cuda")
                    initial_latent_data = torch.load(wait_file_path, map_location="cpu")
                    os.remove(wait_file_path)
                    
                    # 按照原算法处理latents
                    latents_chunk = initial_latent_data.to("cuda").to(torch.bfloat16)
                    mask_latents_chunk = torch.zeros(1, 21, 16, 60, 104).to("cuda").to(torch.bfloat16)
                    mask_latents_chunk[:, 0:1] = latents_chunk[:, 0:1]
                    mask_latents_chunk[:, 1:2] = latents_chunk[:, -2:-1]
                    mask_latents_chunk[:, 2:4] = latents_chunk[:, -2:]

                    vid_chunk = pipeline.vae.decode_to_pixel(mask_latents_chunk).to("cuda").to(torch.bfloat16)
                    vid_chunk = (vid_chunk * 0.5 + 0.5).clamp(0, 1).to("cuda").to(torch.bfloat16)

                    vid_test_chunk = torch.zeros_like(vid_chunk).to("cuda").to(torch.bfloat16)
                    vid_test_chunk[:, 0:5] = vid_chunk[:, 8:13]
                    vid_test_normalized_chunk = vid_test_chunk * 2.0 - 1.0
                    vid_test_rearranged_chunk = rearrange(vid_test_normalized_chunk, "b t c h w -> b c t h w")
                    vid_test_latents_chunk = pipeline.vae.encode_to_latent(vid_test_rearranged_chunk)
                    initial_latent = vid_test_latents_chunk[:, :2].to(torch.bfloat16)
                    
                    pipeline.vae = pipeline.vae.to("cpu")
                
                # 执行推理
                pipeline.vae = pipeline.vae.to("cpu")
                video, latents = pipeline.inference(
                    noise=noise.to(torch.bfloat16),
                    text_prompts=prompts,
                    return_latents=True,
                    initial_latent=initial_latent,
                )
                pipeline.vae = pipeline.vae.to("cuda")
                
                # 保存视频
                current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
                video_save = (current_video * 255.0).clamp(0, 255)
                
                output_path = os.path.join(
                    output_folder, f"{task_id}_{timestamp}_chunk{chunk_idx}.mp4"
                )
                write_video(output_path, video_save[0], fps=16)
                
                chunk_results[chunk_idx] = output_path
                logger.info(f"【{seqid}】Chunk {chunk_idx} 生成完成: {output_path}")
                
            except Exception as e:
                logger.error(f"【{seqid}】Chunk {chunk_idx} 生成失败: {e}")
                chunk_results[chunk_idx] = None
        
        # 创建并启动线程
        for i in range(num_chunks):
            chunk_idx = i + 1
            pipeline_key = f"chunk_{chunk_idx}"
            pipeline = self.model_manager.pipelines[pipeline_key]
            
            # 为每个chunk生成新的噪声
            chunk_noise = torch.randn(
                [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
            )
            
            # 确定等待文件
            wait_file = None if chunk_idx == 1 else f"latents_chunk{chunk_idx-1}.pt"
            
            thread = threading.Thread(
                target=generate_video_chunk,
                args=(pipeline, chunk_noise, prompts, chunk_idx, wait_file)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 处理结果
        video_urls = []
        cover_images = []
        
        for chunk_idx in range(1, num_chunks + 1):
            video_path = chunk_results.get(chunk_idx)
            if video_path and os.path.exists(video_path):
                # 提取首帧
                first_frame_path = os.path.join(
                    output_folder, f"{task_id}_{timestamp}_chunk{chunk_idx}_frame.png"
                )
                VideoProcessor.extract_first_frame(video_path, first_frame_path)
                
                # 添加元数据
                media_video_path = os.path.join(
                    output_folder, f"{task_id}_{timestamp}_chunk{chunk_idx}_media.mp4"
                )
                MediaMetadataHandler.write_video_metadata(seqid, video_path, media_video_path)
                
                media_img_path = os.path.join(
                    output_folder, f"{task_id}_{timestamp}_chunk{chunk_idx}_media.png"
                )
                MediaMetadataHandler.write_png_metadata(seqid, first_frame_path, media_img_path)
                
                # 上传文件
                video_url = self.s3_uploader.upload_with_retry(
                    seqid, media_video_path, f"{task_id}_{timestamp}_chunk{chunk_idx}.mp4"
                )
                cover_url = self.s3_uploader.upload_with_retry(
                    seqid, media_img_path, f"{task_id}_{timestamp}_chunk{chunk_idx}_frame.png"
                )
                
                if video_url:
                    video_urls.append(video_url)
                if cover_url:
                    cover_images.append(cover_url)
        
        return video_urls, cover_images
    
    def _update_task_status(self, task_id: str, seqid: str, status: TaskStatus, 
                           code: ResponseCode, message: str, video_urls: List[str], 
                           cover_images: List[str], text_en: str) -> None:
        """更新任务状态"""
        task_data = TaskStatusResponse(
            task_id=task_id,
            seqid=seqid,
            code=code.value,
            message=message,
            flag=1 if status == TaskStatus.SUCCESS else 0,
            status=status.value,
            data={
                "video": video_urls,
                "cover_image": cover_images,
                "text_en": text_en
            }
        )
        
        # 同时以task_id和seqid为key存储
        self.task_storage.add_task(task_id, task_data.dict())
        if seqid and seqid != task_id:
            self.task_storage.add_task(seqid, task_data.dict())
    
    def get_task_status(self, key: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.task_storage.get_task(key)

# ==================== 全局配置和服务实例 ====================
parallel_server_config = ParallelServerConfig()
s3_config = S3Config()
parallel_video_service = ParallelVideoGenerationService(parallel_server_config, s3_config)

# ==================== FastAPI应用 ====================
app = FastAPI(title="并行文生视频API服务", version="1.0.0")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": parallel_video_service.is_model_loaded(),
        "timestamp": datetime.datetime.now().isoformat(),
        "service_type": "parallel_t2v",
        "num_chunks": parallel_server_config.num_chunks
    }

@app.post("/parallel_text_2_video")
async def generate_parallel_video(request: ParallelVideoRequest, background_tasks: BackgroundTasks):
    """并行视频生成接口"""
    if not parallel_video_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    seqid = request.seqid or task_id
    
    logger.info(f"【{seqid}】接收到并行视频生成请求: {request.prompt}")
    logger.info(f"【{seqid}】任务参数: chunks={request.num_chunks}, seed={request.seed}")
    
    # 添加后台任务
    background_tasks.add_task(parallel_video_service.generate_parallel_video_task, request, task_id)
    
    return ParallelVideoResponse(
        task_id=task_id,
        video_paths=[],
        original_prompt=request.prompt,
        seqid=seqid,
        flag=1,
        status=TaskStatus.PROCESSING.value,
        num_chunks=request.num_chunks
    )

@app.post("/openapi/task_search")
async def task_search_main(request: TaskSearchItem):
    """任务状态查询接口"""
    try:
        result = parallel_video_service.get_task_status(request.seqid)
        if result:
            return result
        else:
            return TaskStatusResponse(
                seqid=request.seqid,
                code=ResponseCode.NOT_FOUND.value,
                message="任务不存在",
                flag=0,
                status="not_found",
                data={"video": [], "cover_image": [], "text_en": ""}
            ).dict()
    except Exception as e:
        logger.error(f"查询任务状态失败: {e}")
        return TaskStatusResponse(
            seqid=request.seqid,
            code=ResponseCode.SERVER_ERROR.value,
            message="服务执行失败",
            flag=0,
            status="error",
            data={"video": [], "cover_image": [], "text_en": ""}
        ).dict()

@app.get("/status/{task_id}")
@app.post("/status/{task_id}")
async def check_task_status(task_id: str):
    """查询任务状态接口（保持向后兼容）"""
    try:
        result = parallel_video_service.get_task_status(task_id)
        if result:
            return result
        else:
            return TaskStatusResponse(
                task_id=task_id,
                code=ResponseCode.NOT_FOUND.value,
                message="任务不存在",
                flag=0,
                status="not_found",
                data={"video": [], "cover_image": [], "text_en": ""}
            ).dict()
    except Exception as e:
        logger.error(f"查询任务状态失败: {e}")
        return TaskStatusResponse(
            task_id=task_id,
            code=ResponseCode.SERVER_ERROR.value,
            message="服务执行失败",
            flag=0,
            status="error",
            data={"video": [], "cover_image": [], "text_en": ""}
        ).dict()

# ==================== 主函数 ====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动并行FastAPI文生视频服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8001, help='服务器端口')
    parser.add_argument('--config_path', type=str, required=True, help='模型配置文件路径')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_folder', type=str, default='videos/parallel_fps', help='输出目录')
    parser.add_argument('--use_ema', action='store_true', help='是否使用EMA参数')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3", help='指定使用的GPU ID，例如"0,1,2,3"')
    parser.add_argument('--num_chunks', type=int, default=4, help='并行处理的视频片段数')
    parser.add_argument('--no_text_expansion', action='store_true', help='禁用文本扩写功能')
    parser.add_argument('--text_expansion_url', type=str, default='http://10.127.16.1:8998/expand', help='文本扩写服务URL')
    parser.add_argument('--prompt_log_file', type=str, default='/app/prompt_extend.txt', help='提示词日志文件路径')
    
    args = parser.parse_args()
    
    # 设置CUDA_VISIBLE_DEVICES环境变量
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_list = [f"cuda:{i}" for i in range(len(args.gpu_ids.split(',')))]
        logger.info(f"使用GPU: {args.gpu_ids} -> {gpu_list}")
    else:
        gpu_list = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    
    # 更新全局配置
    global parallel_server_config
    parallel_server_config.host = args.host
    parallel_server_config.port = args.port
    parallel_server_config.output_folder = args.output_folder
    parallel_server_config.use_ema = args.use_ema
    parallel_server_config.num_chunks = args.num_chunks
    parallel_server_config.gpu_devices = gpu_list
    parallel_server_config.use_text_expansion = not args.no_text_expansion
    parallel_server_config.text_expansion_url = args.text_expansion_url
    parallel_server_config.prompt_log_file = args.prompt_log_file
    
    # 重新初始化服务
    global parallel_video_service
    parallel_video_service = ParallelVideoGenerationService(parallel_server_config, s3_config)
    
    # 打印服务信息
    logger.info(f"启动并行文生视频API服务: http://{args.host}:{args.port}")
    logger.info(f"  - 健康检查: http://{args.host}:{args.port}/health")
    logger.info(f"  - 并行生成视频: http://{args.host}:{args.port}/parallel_text_2_video")
    logger.info(f"  - 任务状态查询: http://{args.host}:{args.port}/openapi/task_search")
    logger.info(f"  - 兼容任务状态: http://{args.host}:{args.port}/status/{{task_id}}")
    logger.info(f"  - API文档: http://{args.host}:{args.port}/docs")
    logger.info(f"  - 并行片段数: {args.num_chunks}")
    logger.info(f"  - GPU设备: {gpu_list}")
    logger.info(f"  - 文本扩写: {'启用' if parallel_server_config.use_text_expansion else '禁用'}")
    
    # 加载模型
    if parallel_video_service.load_models(args.config_path, args.checkpoint_path, args.use_ema):
        # 启动服务器
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        logger.error("并行模型加载失败，服务未启动")

if __name__ == '__main__':
    main()
