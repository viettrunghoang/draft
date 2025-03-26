import torch
from transweather_model import Transweather  # Đảm bảo đúng tên file và class

# Tạo mô hình
model = Transweather()

# Chuyển mô hình sang chế độ đánh giá (không training)
model.eval()

# Tạo ảnh giả lập có kích thước (batch_size=1, channels=3, height=32, width=128)
dummy_input = torch.randn(1, 3, 48, 192)

# Chạy forward pass
output = model(dummy_input)

# Kiểm tra kích thước output
print("Output shape:", output.shape)
