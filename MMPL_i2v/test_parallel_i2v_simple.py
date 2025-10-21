import uuid
import requests

req_data = {
    "seqid": str(uuid.uuid4()),
    "prompt": "一只猫在草地上奔跑",
    "image_url": "http://ops-public-ceph.teleagi.in:8081/telestudio-bucket/b8bd85561c514b178e9fc1fc5885f7e0.png",  # 替换为实际图片URL
    "num_chunks": 4,
    "seed": 0,
    "callback_url": "http://localhost:10004/generatevideo-api/generatevideo/callback/image2video_race"
}

# 修改API路径和端口
res = requests.post(url='http://localhost:9035/parallel_i2v', json=req_data)
print(res.text)
