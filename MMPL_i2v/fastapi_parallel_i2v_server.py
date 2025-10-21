import asyncio
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import argparse

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from omegaconf import OmegaConf
from pydantic import BaseModel
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import requests
from PIL import Image
import io
import base64
import json

# 添加S3支持
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# 确保日志目录存在
log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/parallel_i2v_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 计算正确的image_downloader.py路径
hangzhou_mmpl_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(hangzhou_mmpl_dir)

from pipeline import CausalFPSInferencePipeline, CausalInferencePipeline
from utils.dataset import TextImagePairDataset
from utils.misc import set_seed

# 导入image_downloader模块中的函数
try:
    from image_downloader import download_image_by_object_name, get_s3_presigned_url
    HAVE_IMAGE_DOWNLOADER = True
    logger.info("成功导入image_downloader模块")
except ImportError as e:
    logger.warning(f"无法导入image_downloader模块: {e}")
    HAVE_IMAGE_DOWNLOADER = False

# S3配置
AK = 'wzv-mS2zb75TyETjrU'
SK = 'TPDo-N55VEI6Iyax2E'
endpoint = 'http://ops-public-ceph.teleagi.in:8081'
region = 'YOUR_REGION'
bucket_name = 'telestudio-bucket'

# S3客户端配置
s3_config = Config(connect_timeout=30, read_timeout=30)
s3_client = boto3.client(
    's3',
    aws_access_key_id=AK,
    aws_secret_access_key=SK,
    endpoint_url=endpoint,
    use_ssl=False,
    config=s3_config
)

# 如果没有导入image_downloader，则定义本地版本的函数
if not HAVE_IMAGE_DOWNLOADER:
    def get_s3_presigned_url(object_name, expiration=3600):
        """生成S3对象的预签名URL"""
        try:
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"生成预签名URL失败: {e}")
            return None

    def download_image_by_object_name(object_name, local_path, create_dirs=True, expiration=3600, timeout=30):
        """通过S3对象名生成预签名URL并下载图片"""
        try:
            presigned_url = get_s3_presigned_url(object_name, expiration)
            
            if not presigned_url:
                logger.error(f"无法生成预签名URL: {object_name}")
                return False
            
            if create_dirs:
                local_dir = os.path.dirname(local_path)
                if local_dir and not os.path.exists(local_dir):
                    os.makedirs(local_dir)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(presigned_url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"图片下载成功: {object_name} -> {local_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败: {e}")
            return False
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False

# 枚举定义
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ResponseCode(str, Enum):
    SUCCESS = "200"
    PROCESSING = "202"
    NOT_FOUND = "404"
    ERROR = "500"

# 配置类
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8002
    workers: int = 1
    use_text_expansion: bool = True
    # 修改默认扩写服务URL
    text_expansion_url: str = "http://10.127.16.1:8998/expand"  # 使用T2V的扩写服务
    prompt_log_file: str = "./logs/i2v_prompt_extend.txt"

@dataclass
class S3Config:
    bucket: str = ""
    region: str = ""
    access_key: str = ""
    secret_key: str = ""
    endpoint_url: str = ""

# 请求和响应模型
class ParallelI2VRequest(BaseModel):
    seqid: str
    prompt: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    num_chunks: int = 4
    seed: int = 0
    num_samples: int = 1
    callback_url: Optional[str] = None

class TaskSearchRequest(BaseModel):
    seqid: str

class ParallelI2VResponse(BaseModel):
    code: str
    message: str
    task_id: str
    seqid: str
    flag: int = 1

class TaskStatusResponse(BaseModel):
    code: str
    message: str
    status: str
    seqid: str
    task_id: str
    video_url: Optional[str] = None
    progress: Optional[float] = None

# 工具类
class MediaMetadataHandler:
    @staticmethod
    def inject_metadata(video_path: str, metadata: Dict[str, Any]) -> str:
        """注入元数据到视频文件"""
        try:
            logger.info(f"Injecting metadata to {video_path}: {metadata}")
            return video_path
        except Exception as e:
            logger.error(f"Failed to inject metadata: {e}")
            return video_path

class S3Uploader:
    def __init__(self, config: S3Config):
        self.config = config
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            cfg = Config(connect_timeout=30, read_timeout=30)
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                endpoint_url=self.config.endpoint_url or None,
                use_ssl=False,
                config=cfg
            )
        return self._client
    
    def upload_file(self, local_path: str, s3_key: str) -> str:
        """上传文件到S3并返回可访问URL"""
        try:
            logger.info(f"Uploading {local_path} to S3 as {s3_key}")
            self.client.upload_file(
                Filename=local_path,
                Bucket=self.config.bucket,
                Key=s3_key,
                ExtraArgs={'ACL': 'public-read'}
            )
            url = self.client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': self.config.bucket, 'Key': s3_key},
                ExpiresIn=24 * 3600
            )
            # 删除本地文件
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
            except Exception as e:
                logger.warning(f"删除本地文件失败: {e}")
            # 返回纯链接（去除签名参数）
            final_url = url.split('?', 1)[0]
            logger.info(f"S3 上传成功，URL: {final_url}")
            return final_url
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise

class TaskStorage:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.seqid_to_task: Dict[str, str] = {}
    
    def create_task(self, task_id: str, seqid: str, request_data: Dict) -> None:
        self.tasks[task_id] = {
            "task_id": task_id,
            "seqid": seqid,
            "status": TaskStatus.PENDING,
            "request_data": request_data,
            "created_at": time.time(),
            "updated_at": time.time(),
            "video_url": None,
            "progress": 0.0,
            "error_message": None
        }
        self.seqid_to_task[seqid] = task_id
    
    def update_task(self, task_id: str, **kwargs) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].update(kwargs)
            self.tasks[task_id]["updated_at"] = time.time()
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        return self.tasks.get(task_id)
    
    def get_task_by_seqid(self, seqid: str) -> Optional[Dict]:
        task_id = self.seqid_to_task.get(seqid)
        return self.tasks.get(task_id) if task_id else None

class ImageProcessor:
    @staticmethod
    def download_image(url: str) -> Image.Image:
        """从S3下载图像"""
        try:
            if url.startswith("http://ops-public-ceph.teleagi.in:8081/telestudio-bucket/"):
                object_name = url.replace("http://ops-public-ceph.teleagi.in:8081/telestudio-bucket/", "")
                logger.info(f"提取S3对象名称: {object_name}")
            else:
                object_name = url
                logger.info(f"使用对象名称: {object_name}")
            
            temp_dir = "./temp_downloads"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            temp_filename = os.path.join(temp_dir, f"img_{uuid.uuid4().hex}.jpg")
            
            success = download_image_by_object_name(
                object_name=object_name,
                local_path=temp_filename,
                create_dirs=True,
                expiration=3600,
                timeout=30
            )
            
            if not success:
                raise Exception(f"S3下载图片失败: {object_name}")
            
            image = Image.open(temp_filename).convert('RGB')
            
            try:
                os.remove(temp_filename)
                logger.info(f"临时文件删除成功: {temp_filename}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {temp_filename}, 错误: {e}")
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to download image from S3: {url}, error: {e}")
            raise
    
    @staticmethod
    def decode_base64_image(base64_str: str) -> Image.Image:
        """解码base64图像"""
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise

class CallbackHandler:
    @staticmethod
    def execute_callback(callback_url: str, seqid: str, code: int, message: str,
                         flag: int, video_urls: List[str], cover_images: List[str],
                         text_en: str, max_retries: int = 3) -> None:
        """同步回调（与 fastapi_t2v_server.py 一致的请求体和实现）"""
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

        headers = {'Content-type': 'application/json'}
        for attempt in range(max_retries):
            try:
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
    @staticmethod
    def save_video_chunks(video_chunks: List[torch.Tensor], output_path: str, fps: int = 16) -> str:
        """保存视频片段"""
        try:
            # 合并所有视频片段
            full_video = torch.cat(video_chunks, dim=0)  # 在时间维度上拼接
            
            write_video(output_path, full_video, fps=fps)
            logger.info(f"Video saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            raise

    @staticmethod
    def extract_first_frame(video_path: str, output_path: str) -> bool:
        """提取视频首帧。优先使用 moviepy；失败则回退到 torchvision，再失败回退到 OpenCV。"""
        # 确保输出目录存在
        try:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"创建封面输出目录失败（忽略）: {e}")

        # 方案1：moviepy（同时兼容 2.x 与 1.x 的导入路径）
        try:
            try:
                from moviepy import VideoFileClip as _VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip as _VideoFileClip

            with _VideoFileClip(video_path) as clip:
                frame = clip.get_frame(0)
            img = Image.fromarray(frame)
            img.save(output_path, format="PNG")
            logger.info(f"视频首帧提取成功(moviepy): {output_path}")
            return True
        except Exception as e_moviepy:
            logger.warning(f"moviepy 首帧提取失败，尝试 torchvision.read_video。原因: {e_moviepy}")

        # 方案2：torchvision.io.read_video（回退）
        try:
            from torchvision.io import read_video  # 按需导入
            vframes, _, _ = read_video(video_path, start_pts=0, end_pts=0.04, pts_unit="sec")
            if vframes is not None and len(vframes) > 0:
                frame0 = vframes[0].numpy()
                img = Image.fromarray(frame0)
                img.save(output_path, format="PNG")
                logger.info(f"视频首帧提取成功(torchvision): {output_path}")
                return True
            else:
                raise RuntimeError("torchvision 读取无帧")
        except Exception as e_tv:
            logger.warning(f"torchvision 首帧提取失败，尝试 OpenCV。原因: {e_tv}")

        # 方案3：OpenCV（再回退）
        try:
            import cv2  # 按需导入
            cap = cv2.VideoCapture(video_path)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.save(output_path, format="PNG")
                logger.info(f"视频首帧提取成功(OpenCV): {output_path}")
                return True
            else:
                raise RuntimeError("OpenCV 未读到帧")
        except Exception as e_cv:
            logger.error(f"提取首帧失败，已尝试 moviepy/torchvision/OpenCV。错误: moviepy={e_moviepy}; torchvision={e_tv}; opencv={e_cv}")
            return False

# 模型管理器
class ParallelI2VModelManager:
    def __init__(self, config_path: str, checkpoint_path: str, gpu_ids: List[int], num_chunks: int = 4, use_ema: bool = True):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.gpu_ids = gpu_ids
        self.num_chunks = num_chunks
        self.use_ema = use_ema
        self.pipelines = {}
        self.config = None
        
    def load_models(self):
        """加载模型到各个GPU"""
        logger.info("Loading I2V models...")
        
        # 加载配置
        config = OmegaConf.load(self.config_path)
        try:
            default_config = OmegaConf.load("configs/default_config.yaml")
        except FileNotFoundError:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                default_config_path = os.path.join(script_dir, "configs", "default_config.yaml")
                default_config = OmegaConf.load(default_config_path)
            except FileNotFoundError:
                logger.error("无法找到 default_config.yaml 文件")
                raise FileNotFoundError("default_config.yaml not found")
        
        self.config = OmegaConf.merge(default_config, config)
        
        # 为每个GPU创建pipeline
        for i in range(self.num_chunks):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            device = f"cuda:{gpu_id}"
            
            if hasattr(self.config, 'denoising_step_list'):
                pipeline = CausalInferencePipeline(self.config, device=device)
            else:
                save_file = f"latents_chunk{i+1}.pt"
                pipeline = CausalFPSInferencePipeline(
                    self.config, 
                    device=device,
                    device_cond=device,
                    device_uncond=device,
                    save=save_file
                )
            
            # 加载权重
            if self.checkpoint_path:
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")
                weight_key = 'generator_ema' if self.use_ema else 'generator'
                
                if weight_key not in state_dict:
                    logger.warning(f"权重键 '{weight_key}' 不存在，尝试使用其他键")
                    if 'generator_ema' in state_dict:
                        weight_key = 'generator_ema'
                        logger.info("使用 generator_ema 权重")
                    elif 'generator' in state_dict:
                        weight_key = 'generator'
                        logger.info("使用 generator 权重")
                    else:
                        logger.error("找不到可用的权重键")
                        available_keys = list(state_dict.keys())
                        logger.error(f"可用的权重键: {available_keys}")
                        raise KeyError(f"找不到权重键，可用键: {available_keys}")
                
                pipeline.generator_cond.load_state_dict(state_dict[weight_key])
            
            pipeline = pipeline.to(dtype=torch.bfloat16)
            pipeline.independent_first_frame = False
            
            self.pipelines[f"chunk_{i+1}"] = pipeline
            logger.info(f"Loaded pipeline for chunk {i+1} on {device}")
    
    def get_pipeline(self, chunk_id: str):
        return self.pipelines.get(chunk_id)

# 主服务类
class ParallelI2VGenerationService:
    def __init__(self, model_manager: ParallelI2VModelManager, output_folder: str, 
                 s3_config: Optional[S3Config] = None, server_config: Optional[ServerConfig] = None):
        self.model_manager = model_manager
        self.output_folder = output_folder
        self.s3_uploader = S3Uploader(s3_config) if s3_config else None
        self.task_storage = TaskStorage()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.server_config = server_config or ServerConfig()
        
        # 初始化文本扩写器
        if self.server_config.use_text_expansion:
            self.text_expander = self.TextExpander(
                self.server_config.text_expansion_url,
                self.server_config.prompt_log_file
            )
        else:
            self.text_expander = None
        
        # 确保输出目录存在
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs("temp_latents", exist_ok=True)

    async def generate_video(self, request: ParallelI2VRequest) -> str:
        """异步生成视频"""
        task_id = str(uuid.uuid4())
        
        # 创建任务记录
        self.task_storage.create_task(task_id, request.seqid, request.dict())
        
        # 提交后台任务
        self.executor.submit(self._generate_video_sync, task_id, request)
        
        return task_id
    
    class TextExpander:
        """文本扩写服务"""
        
        def __init__(self, url: str, log_file: str):
            self.url = url
            self.log_file = log_file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        def expand(self, prompt: str) -> str:
            """扩写文本（增强：显式Content-Type、非200打印响应体、容错解析多个字段、失败回退）"""
            headers = {'Content-Type': 'application/json'}
            try:
                response = requests.post(self.url, data=json.dumps({"prompt": prompt}), headers=headers)
                if response.status_code == 200:
                    try:
                        result = response.json()
                    except Exception as je:
                        logger.warning(f"文本扩写返回非JSON，使用原始提示词。解析异常: {je}")
                        return prompt
                    expanded_prompt = None
                    # 兼容多个返回字段
                    if isinstance(result, dict):
                        if isinstance(result.get("expanded"), str):
                            expanded_prompt = result.get("expanded")
                        elif isinstance(result.get("text_en"), str):
                            expanded_prompt = result.get("text_en")
                        elif isinstance(result.get("data"), dict) and isinstance(result["data"].get("expanded"), str):
                            expanded_prompt = result["data"]["expanded"]
                    if not expanded_prompt or not isinstance(expanded_prompt, str):
                        logger.warning(f"文本扩写结果缺少可用字段，使用原始提示词。返回体: {result}")
                        return prompt
                    logger.info(f"原始提示词: '{prompt}'")
                    logger.info(f"扩写后提示词: '{expanded_prompt}'")
                    self._save_to_log(expanded_prompt)
                    return expanded_prompt
                else:
                    logger.warning(f"文本扩写请求失败: {response.status_code}, body: {response.text}")
                    return prompt
            except Exception as e:
                logger.error(f"文本扩写过程出错: {e}，URL={self.url}")
                return prompt
        
        def _save_to_log(self, expanded_prompt: str) -> None:
            """保存扩写结果到日志文件"""
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{expanded_prompt}\n")
            except Exception as e:
                logger.error(f"保存提示词到文件时出错: {e}")
    
    def _generate_video_sync(self, task_id: str, request: ParallelI2VRequest):
        """同步生成视频（在线程池中执行）"""
        # 预定义提示词，避免异常路径未定义
        original_prompt = request.prompt
        expanded_prompt = original_prompt
        try:
            self.task_storage.update_task(task_id, status=TaskStatus.PROCESSING, progress=0.1)
            
            # 处理输入图像
            if request.image_url:
                image = ImageProcessor.download_image(request.image_url)
                image_url_for_expansion = request.image_url
            elif request.image_base64:
                image = ImageProcessor.decode_base64_image(request.image_base64)
                image_url_for_expansion = None
            else:
                raise ValueError("Either image_url or image_base64 must be provided")
            
            # 文本扩写
            original_prompt = request.prompt
            expanded_prompt = original_prompt
            
            if (self.text_expander and 
                self.server_config.use_text_expansion):
                try:
                    expanded_prompt = self.text_expander.expand(original_prompt)
                    logger.info(f"文本扩写成功: '{original_prompt}' -> '{expanded_prompt}'")
                except Exception as e:
                    logger.warning(f"文本扩写失败，使用原始提示词: {e}")
                    expanded_prompt = original_prompt
            
            # 更新request中的prompt为扩写后的prompt
            request.prompt = expanded_prompt
            
            self.task_storage.update_task(task_id, progress=0.2)
            
            # 生成视频
            video_path = self._generate_video_tensor(request, image, task_id)
            
            self.task_storage.update_task(task_id, progress=0.8)
            
            # 处理和上传视频
            final_video_url, cover_url = self._process_and_upload_video(
                video_path, request, task_id, original_prompt, expanded_prompt
            )
            
            # 更新任务状态
            self.task_storage.update_task(
                task_id, 
                status=TaskStatus.COMPLETED,
                video_url=final_video_url,
                cover_url=cover_url,
                progress=1.0
            )
            
            # 发送回调（同步，统一封装）
            if request.callback_url:
                self._send_completion_callback(request, task_id, final_video_url, cover_url, expanded_prompt)
            
        except Exception as e:
            logger.error(f"Video generation failed for task {task_id}: {e}")
            self.task_storage.update_task(
                task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                progress=0.0
            )
            
            # 发送失败回调（同步，统一封装）
            if request.callback_url:
                self._send_failure_callback(request, task_id, str(e), expanded_prompt)
    
    def _generate_video_tensor(self, request: ParallelI2VRequest, image: Image.Image, task_id: str) -> str:
        """生成视频张量 - 按照原版本的逻辑"""
        try:
            # 准备图像张量
            image_tensor = self._prepare_image_tensor(image)
            
            # 设置随机种子
            if request.seed is not None:
                set_seed(request.seed)
            
            # 存储视频片段的列表
            video_chunks = []
            video_chunks_lock = threading.Lock()
            
            def generate_video_chunk(pipeline, sampled_noise, prompts, chunk_idx, wait_file=None):
                try:
                    if wait_file:
                        # 等待前一个chunk的latents文件
                        pipeline.vae = pipeline.vae.to("cuda")
                        logger.info(f"[Chunk {chunk_idx}] Waiting for {wait_file}...")
                        while not os.path.exists(wait_file):
                            time.sleep(1.0)
                        logger.info(f"[Chunk {chunk_idx}] Detected {wait_file}, loading as initial_latent...")
                        
                        initial_latent = torch.load(wait_file, map_location="cpu")
                        os.remove(wait_file)
                        logger.info(f"[Chunk {chunk_idx}] Removed {wait_file}")

                        # 按照原版本的latent处理逻辑
                        latents_chunk = initial_latent.to("cuda").to(torch.bfloat16)
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
                    else:
                        # 第一个chunk：使用输入图像作为initial_latent
                        if chunk_idx == 1:
                            # 确保VAE在CUDA上，与image_tensor保持一致
                            pipeline.vae = pipeline.vae.to("cuda")
                            initial_latent = pipeline.vae.encode_to_latent(image_tensor).to(device="cuda", dtype=torch.bfloat16)
                            initial_latent = initial_latent.repeat(request.num_samples, 1, 1, 1, 1)
                            logger.info(f"[Chunk {chunk_idx}] Initial latent shape: {initial_latent.shape}")
                        else:
                            initial_latent = None
                    
                    # 将VAE移到CPU以节省显存
                    pipeline.vae = pipeline.vae.to("cpu")
                    
                    # 生成视频
                    video, latents = pipeline.inference(
                        noise=sampled_noise.to(torch.bfloat16),
                        text_prompts=prompts,
                        return_latents=True,
                        initial_latent=initial_latent,
                    )
                    
                    # 将VAE移回GPU用于后续处理
                    pipeline.vae = pipeline.vae.to("cuda")
                    
                    # 保存视频片段
                    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
                    video_save = (current_video * 255.0).clamp(0, 255)
                    
                    # 线程安全地添加到列表（修复 off-by-one）
                    with video_chunks_lock:
                        while len(video_chunks) < chunk_idx:
                            video_chunks.append(None)
                        video_chunks[chunk_idx - 1] = video_save[0]
                    
                    logger.info(f"[Chunk {chunk_idx}] Generated successfully")
                    
                except Exception as e:
                    logger.error(f"[Chunk {chunk_idx}] Generation failed: {e}")
                    logger.error(f"[Chunk {chunk_idx}] Exception details: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"[Chunk {chunk_idx}] Traceback: {traceback.format_exc()}")
                    # 确保在异常情况下也在列表中占位
                    with video_chunks_lock:
                        while len(video_chunks) < chunk_idx:
                            video_chunks.append(None)
                        if len(video_chunks) == chunk_idx:
                            video_chunks.append(None)
                        else:
                            video_chunks[chunk_idx-1] = None
                    raise
            
            # 创建并启动线程（按照原版本的逻辑）
            threads = []
            for i in range(request.num_chunks):
                chunk_id = f"chunk_{i+1}"
                pipeline = self.model_manager.get_pipeline(chunk_id)
                
                if pipeline is None:
                    raise ValueError(f"Pipeline for {chunk_id} not found")
                
                # 使用21帧（与原版本一致）
                if i == 0:
                    # 第一个chunk：使用21帧噪声（与原版本一致）
                    sampled_noise = torch.randn(
                        [request.num_samples, 21, 16, 60, 104],  # ✅ 修正为21帧
                        device="cuda", 
                        dtype=torch.bfloat16
                    )
                    wait_file = None
                    # ✅ 不使用initial_latent，保持与原版本一致
                else:
                    # 后续chunk：使用21帧噪声
                    sampled_noise = torch.randn(
                        [request.num_samples, 21, 16, 60, 104], 
                        device="cuda", 
                        dtype=torch.bfloat16
                    )
                    wait_file = f"latents_chunk{i}.pt"
                
                prompts = [request.prompt] * request.num_samples
                
                thread = threading.Thread(
                    target=generate_video_chunk,
                    args=(pipeline, sampled_noise, prompts, i+1, wait_file)
                )
                threads.append(thread)
                thread.start()
                
                # 更新进度
                progress = 0.2 + (0.6 * (i + 1) / request.num_chunks)
                self.task_storage.update_task(task_id, progress=progress)
            
            # 等待所有线程完成
            for i, thread in enumerate(threads):
                thread.join()
                logger.info(f"Thread {i+1} completed")
            
            # 检查视频片段生成情况
            logger.info(f"Generated video chunks: {len(video_chunks)}")
            for i, chunk in enumerate(video_chunks):
                if chunk is not None:
                    logger.info(f"Chunk {i+1}: shape {chunk.shape}")
                else:
                    logger.error(f"Chunk {i+1}: None (failed)")
            
            # 保存最终视频
            timestamp = int(time.time())
            video_filename = f"i2v_{task_id}_{timestamp}.mp4"
            video_path = os.path.join(self.output_folder, video_filename)
            
            # 合并所有视频片段
            if video_chunks and all(chunk is not None for chunk in video_chunks):
                # 按照原版本的方式合并视频
                full_video = torch.cat(video_chunks, dim=0)  # 在时间维度拼接
                write_video(video_path, full_video, fps=16)
                logger.info(f"Video generation completed: {video_path}")
                return video_path
            else:
                # 提供更详细的错误信息
                failed_chunks = [i+1 for i, chunk in enumerate(video_chunks) if chunk is None]
                error_msg = f"Video generation failed. Total chunks: {len(video_chunks)}, Failed chunks: {failed_chunks}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"Failed to generate video tensor: {e}")
            raise

    def _prepare_image_tensor(self, image: Image.Image) -> torch.Tensor:
        """准备图像张量作为初始latent - 按照原版本的方式"""
        try:
            # 使用与原版本相同的transform
            transform = transforms.Compose([
                transforms.Resize((480, 832)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # 注意：原版本使用单通道normalize
            ])
            
            # 转换图像并调整维度：(3, H, W) -> (1, 3, 1, H, W)
            image_tensor = transform(image).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
            
            # 确保tensor在CUDA设备上
            return image_tensor.to(device="cuda", dtype=torch.bfloat16)
            
        except Exception as e:
            logger.error(f"Failed to prepare image tensor: {e}")
            raise

    def _process_and_upload_video(self, video_path: str, request: ParallelI2VRequest, 
                                task_id: str, original_prompt: str = None, 
                                expanded_prompt: str = None) -> Tuple[str, Optional[str]]:
        """处理和上传视频（只使用S3），同时提取并上传首帧作为封面，返回(video_url, cover_url)"""
        try:
            # 注入元数据
            metadata = {
                "seqid": request.seqid,
                "task_id": task_id,
                "original_prompt": original_prompt or request.prompt,
                "expanded_prompt": expanded_prompt or request.prompt,
                "seed": request.seed,
                "num_chunks": request.num_chunks,
                "generation_time": time.time()
            }
            video_path = MediaMetadataHandler.inject_metadata(video_path, metadata)

            # 先从本地视频提取首帧（在视频被S3Uploader删除之前）
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            cover_local_path = os.path.join(os.path.dirname(video_path), f"{base_name}_cover.png")
            cover_extracted = VideoProcessor.extract_first_frame(video_path, cover_local_path)

            # 只使用 S3
            if not self.s3_uploader:
                raise RuntimeError("S3 未配置：当前服务只支持 S3 上传，请检查 S3Config。")

            # 上传视频
            s3_key_video = f"i2v_videos/{request.seqid}/{os.path.basename(video_path)}"
            video_url = self.s3_uploader.upload_file(video_path, s3_key_video)

            # 上传封面（如果提取成功）
            cover_url: Optional[str] = None
            if cover_extracted and os.path.exists(cover_local_path):
                s3_key_cover = f"i2v_covers/{request.seqid}/{os.path.basename(cover_local_path)}"
                try:
                    cover_url = self.s3_uploader.upload_file(cover_local_path, s3_key_cover)
                except Exception as e:
                    logger.warning(f"封面图上传失败，忽略: {e}")
                    cover_url = None

            return video_url, cover_url
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise

    def _send_completion_callback(self, request: ParallelI2VRequest, task_id: str, video_url: str, cover_url: Optional[str] = None, expanded_prompt: Optional[str] = None):
        """发送完成回调（同步，结构同 T2V）"""
        seqid = request.seqid
        cover_list = [cover_url] if cover_url else []
        CallbackHandler.execute_callback(
            request.callback_url,
            seqid,
            10000,
            "任务执行成功",
            1,  # 成功时 flag=1
            [video_url],
            cover_list,
            expanded_prompt or request.prompt
        )
    
    def _send_failure_callback(self, request: ParallelI2VRequest, task_id: str, error_message: str, expanded_prompt: Optional[str] = None):
        """发送失败回调（同步，结构同 T2V）"""
        seqid = request.seqid
        CallbackHandler.execute_callback(
            request.callback_url,
            seqid,
            10903,
            error_message or "任务执行异常",
            0,
            [],
            [],
            expanded_prompt or request.prompt
        )

    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.task_storage.get_task(task_id)
    
    def search_task_by_seqid(self, seqid: str) -> Optional[Dict]:
        """根据seqid搜索任务"""
        return self.task_storage.get_task_by_seqid(seqid)

# FastAPI应用
app = FastAPI(title="Parallel I2V Generation API", version="1.0.0")

# 全局变量
video_service: Optional[ParallelI2VGenerationService] = None

@app.post("/parallel_i2v", response_model=ParallelI2VResponse)
async def generate_parallel_i2v(request: ParallelI2VRequest):
    """生成并行I2V视频"""
    try:
        task_id = await video_service.generate_video(request)
        
        return ParallelI2VResponse(
            code=ResponseCode.PROCESSING,
            message="Video generation started",
            task_id=task_id,
            seqid=request.seqid,
            flag=1
        )
    except Exception as e:
        logger.error(f"Failed to start video generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/task_search", response_model=TaskStatusResponse)
async def search_task(request: TaskSearchRequest):
    """根据seqid搜索任务状态"""
    try:
        task = video_service.search_task_by_seqid(request.seqid)
        
        if not task:
            return TaskStatusResponse(
                code=ResponseCode.NOT_FOUND,
                message="Task not found",
                status="not_found",
                seqid=request.seqid,
                task_id=""
            )
        
        return TaskStatusResponse(
            code=ResponseCode.SUCCESS,
            message="Task found",
            status=task["status"],
            seqid=task["seqid"],
            task_id=task["task_id"],
            video_url=task.get("video_url"),
            progress=task.get("progress")
        )
    except Exception as e:
        logger.error(f"Failed to search task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task_status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    try:
        task = video_service.get_task_status(task_id)
        
        if not task:
            return TaskStatusResponse(
                code=ResponseCode.NOT_FOUND,
                message="Task not found",
                status="not_found",
                seqid="",
                task_id=task_id
            )
        
        return TaskStatusResponse(
            code=ResponseCode.SUCCESS,
            message="Task status retrieved",
            status=task["status"],
            seqid=task["seqid"],
            task_id=task["task_id"],
            video_url=task.get("video_url"),
            progress=task.get("progress")
        )
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "parallel_i2v_generation"}

def main():
    global video_service
    
    parser = argparse.ArgumentParser(description="Parallel I2V FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output_folder", type=str, default="videos/parallel_i2v", help="Output folder")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="GPU IDs (comma-separated)")
    parser.add_argument("--num_chunks", type=int, default=4, help="Number of parallel chunks")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    
    # S3配置（可选）
    parser.add_argument("--s3_bucket", type=str, help="S3 bucket name")
    parser.add_argument("--s3_region", type=str, help="S3 region")
    parser.add_argument("--s3_access_key", type=str, help="S3 access key")
    parser.add_argument("--s3_secret_key", type=str, help="S3 secret key")
    parser.add_argument("--s3_endpoint_url", type=str, help="S3 endpoint URL")
    
    # 文本扩写配置
    parser.add_argument("--no_text_expansion", action="store_true", help="禁用文本扩写功能")
    parser.add_argument("--text_expansion_url", type=str, default="http://10.127.16.1:8998/expand", help="文本扩写服务URL")
    parser.add_argument("--prompt_log_file", type=str, default="./logs/i2v_prompt_extend.txt", help="提示词日志文件路径")
    
    args = parser.parse_args()
    
    # 解析GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    
    # 创建服务器配置
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        use_text_expansion=not args.no_text_expansion,
        text_expansion_url=args.text_expansion_url,
        prompt_log_file=args.prompt_log_file
    )
    
    # S3 配置（只使用这一种，写死并打印）
    s3_config = S3Config(
        bucket="telestudio-bucket",
        region="YOUR_REGION",
        access_key="wzv-mS2zb75TyETjrU",
        secret_key="TPDo-N55VEI6Iyax2E",
        endpoint_url="http://ops-public-ceph.teleagi.in:8081"
    )
    logger.info("===== S3 Configuration (Hardcoded) =====")
    logger.info(f"S3 bucket: {s3_config.bucket}")
    logger.info(f"S3 region: {s3_config.region}")
    logger.info(f"S3 endpoint: {s3_config.endpoint_url}")
    logger.info("========================================")
    
    # 初始化模型管理器
    logger.info("Initializing Parallel I2V Model Manager...")
    model_manager = ParallelI2VModelManager(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        gpu_ids=gpu_ids,
        num_chunks=args.num_chunks,
        use_ema=args.use_ema
    )
    
    # 加载模型
    logger.info("Loading models...")
    model_manager.load_models()
    
    # 初始化服务（强制传入 s3_config）
    logger.info("Initializing Parallel I2V Generation Service...")
    video_service = ParallelI2VGenerationService(
        model_manager=model_manager,
        output_folder=args.output_folder,
        s3_config=s3_config,
        server_config=server_config
    )
    
    logger.info(f"Starting Parallel I2V FastAPI server on {args.host}:{args.port}")
    logger.info(f"API Documentation: http://{args.host}:{args.port}/docs")
    logger.info(f"文本扩写: {'启用' if server_config.use_text_expansion else '禁用'}")
    if server_config.use_text_expansion:
        logger.info(f"扩写服务URL: {server_config.text_expansion_url}")
    
    # 启动服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info"
    )

if __name__ == "__main__":
    main()
