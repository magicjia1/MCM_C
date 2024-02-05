import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 读取CSV文件
csv_file_path = 'data.csv'
df = pd.read_csv(csv_file_path)
# 将标签映射为0和1
df['point_victor'] = df['point_victor'].map({1: 0, 2: 1})

# 获取特征和标签
target = df['point_victor']
# 2. 数据预处理
df['elapsed_time_seconds'] = pd.to_timedelta(df['elapsed_time']).dt.total_seconds()

# 获取特征和标签
features = df[['elapsed_time_seconds', 'set_no', 'game_no', 'point_no', 'p1_sets', 'p2_sets', 'p1_games', 'p2_games', 'server', 'serve_no', 'p1_points_won', 'p2_points_won', 'game_victor', 'set_victor', 'p1_ace', 'p2_ace', 'p1_winner', 'p2_winner', 'p1_unf_err', 'p2_unf_err', 'p1_net_pt', 'p2_net_pt', 'p1_net_pt_won', 'p2_net_pt_won', 'p1_break_pt', 'p2_break_pt', 'p1_break_pt_won', 'p2_break_pt_won', 'p1_break_pt_missed', 'p2_break_pt_missed']]




# 3. 数据标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 4. 数据集分割
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# 5. 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # 将标签转换为列向量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # 将标签转换为列向量

# 6. 创建神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 7. 初始化模型和定义损失函数、优化器
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# 8. 训练模型
num_epochs = 100
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


# 在训练集上评估模型
model.eval()
with torch.no_grad():
    train_outputs = model(X_train_tensor)
    train_predicted_labels = torch.round(torch.sigmoid(train_outputs))
    train_accuracy = (train_predicted_labels == y_train_tensor).sum().item() / len(y_train_tensor)
    print(f'Train Accuracy: {train_accuracy}')


# 9. 在测试集上评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted_labels = torch.round(torch.sigmoid(test_outputs))
    accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy}')
