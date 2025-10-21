import uuid
import requests
import time

# 测试并行T2V服务的脚本
req_data = {
    "seqid": str(uuid.uuid4()),
    "prompt": "一只猫在草地上奔跑",
    "num_chunks": 4,  # 并行处理的片段数
    "seed": 0,
    "callback_url": "http://localhost:10004/generatevideo-api/generatevideo/callback/image2video_race"
}

print(f"🚀 发送视频生成请求...")
print(f"📝 请求数据: {req_data}")

# 调用并行T2V API
try:
    res = requests.post(url='http://localhost:9024/parallel_text_2_video', json=req_data)
    print(f"✅ 响应状态码: {res.status_code}")
    print(f"📄 响应内容: {res.text}")
    
    if res.status_code == 200:
        response_data = res.json()
        task_id = response_data.get('task_id')
        seqid = req_data['seqid']
        
        print(f"🎯 任务ID: {task_id}")
        print(f"🔍 可以使用以下方式查询任务状态:")
        print(f"   方式1 - 按task_id查询: GET http://localhost:9024/status/{task_id}")
        print(f"   方式2 - 按seqid查询: POST http://localhost:9024/openapi/task_search")
        
        # 自动查询任务状态
        print(f"\n⏳ 开始查询任务状态...")
        for i in range(10):  # 最多查询10次
            time.sleep(5)  # 等待5秒
            try:
                # 使用seqid查询
                search_data = {"seqid": seqid}
                status_res = requests.post(url='http://localhost:9024/openapi/task_search', json=search_data)
                if status_res.status_code == 200:
                    status_data = status_res.json()
                    print(f"📊 第{i+1}次查询 - 状态: {status_data}")
                    
                    if status_data.get('status') == 'completed':
                        print(f"�� 任务完成!")
                        break
                    elif status_data.get('status') == 'failed':
                        print(f"❌ 任务失败!")
                        break
                else:
                    print(f"⚠️ 状态查询失败: {status_res.text}")
            except Exception as e:
                print(f"⚠️ 查询异常: {e}")
    
except Exception as e:
    print(f"❌ 请求失败: {e}")
