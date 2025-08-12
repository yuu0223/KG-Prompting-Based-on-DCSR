import torch
print(torch.cuda.is_available())  # 如果返回 False，則沒有可用的 GPU
print(torch.cuda.device_count())  # 檢查可用的 GPU 數量