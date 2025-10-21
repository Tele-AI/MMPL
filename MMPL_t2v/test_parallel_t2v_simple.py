import uuid
import requests

req_data = {
    "seqid": str(uuid.uuid4()),
    "prompt": "一只猫在草地上奔跑",
    "num_chunks": 4,
    "seed": 0,
    "callback_url": "http://localhost:10004/generatevideo-api/generatevideo/callback/image2video_race"
}

res = requests.post(url='http://localhost:9024/parallel_text_2_video', json=req_data)
print(res.text)
