import torch
import pandas as pd
import numpy as np
from best_phone import Autoencoder, Classifier, preprocess_data

# 加载测试数据
test_data = pd.read_csv('test.csv')
X_test = test_data.iloc[:, :561].values  # 假设测试数据没有标签，只有特征

# 数据标准化（使用与训练数据相同的 mean 和 std）
mean = np.load('mean.npy')
std = np.load('std.npy')


X_test_normalized = (X_test - mean) / std

# 转换为 torch tensor
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)


# 重新创建模型并加载状态
autoencoder = Autoencoder(input_dim=561, hidden_dim=256)
classifier = Classifier(input_dim=256, num_classes=6)

autoencoder.load_state_dict(torch.load('autoencoder.pth'))
classifier.load_state_dict(torch.load('classifier.pth'))

autoencoder.eval()
classifier.eval()

# 进行预测
with torch.no_grad():
    encoded = autoencoder.encoder(X_test_tensor)
    predictions = classifier(encoded)
    predicted_labels = predictions.argmax(dim=1)

# 输出预测结果
print(predicted_labels.numpy())

import pandas as pd

# 定义标签到活动名称的映射
label_to_activity = {
    0: 'WALKING_UPSTAIRS',
    1: 'WALKING_DOWNSTAIRS',
    2: 'WALKING',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}

# 假设 predicted_labels 是模型预测出的标签 Tensor
predicted_labels_np = predicted_labels.numpy()  # 转换为 NumPy 数组

# 将预测的标签映射到活动名称
predicted_activities = [label_to_activity[label] for label in predicted_labels_np]

# 创建 DataFrame
df_predicted = pd.DataFrame(predicted_activities, columns=['PredictedActivity'])

# 保存到 CSV 文件
df_predicted.to_csv('predicted_activities.csv', index=False)

print("Predicted activities have been saved to predicted_activities.csv.")

