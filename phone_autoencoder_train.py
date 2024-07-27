import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 数据预处理
def preprocess_data(data_file, header_row=0):
    df = pd.read_csv(data_file, header=header_row)
    X = df.iloc[:, :561].values
    y = df.iloc[:, 561].values

    label_mapping = {'WALKING_UPSTAIRS': 0, 'WALKING_DOWNSTAIRS': 1, 'WALKING': 2,
                     'SITTING': 3, 'STANDING': 4, 'LAYING': 5}
    y = np.vectorize(label_mapping.get)(y)

    X_noisy = add_gaussian_noise(X)
    X_combined = np.vstack((X, X_noisy))
    y_combined = np.hstack((y, y))
    return X_combined, y_combined  # 返回处理后的数据以便后续创建 DataLoader

def add_gaussian_noise(X, mean=0, std=0.01):
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def prepare_data_loaders(X, y, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    mean = X_tensor.mean(dim=0)
    std = X_tensor.std(dim=0)
    X_tensor = (X_tensor - mean) / std

    np.save('mean.npy', mean.numpy())
    np.save('std.npy', std.numpy())

    train_size = int(0.8 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

# 模型定义
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.ReLU())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# 训练函数
def train_autoencoder(model, train_loader, num_epochs=10, optimizer_lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=optimizer_lr)

    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Autoencoder Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def train_classifier(encoder, classifier, train_loader, num_epochs=20, optimizer_lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=optimizer_lr)

    for epoch in range(num_epochs):
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            features = encoder(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Classifier Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估函数
def evaluate_model(encoder, classifier, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            features = encoder(inputs)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')
    return accuracy  # 返回计算出的准确率



def main():
    X_combined, y_combined = preprocess_data('train.csv')


    input_dim = 561
    hidden_dim = 256
    num_classes = 6

    #learning_rates = [0.1, 0.01, 0.001]
    #batch_sizes = [16, 32, 64]
    learning_rates=[0.001]
    batch_sizes=[16]
    best_accuracy = 0
    best_lr = None
    best_batch_size = None
    epochs= 20


    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with lr = {lr}, batch_size = {batch_size}")
            train_loader, test_loader = prepare_data_loaders(X_combined, y_combined, batch_size=batch_size)

            autoencoder = Autoencoder(input_dim, hidden_dim)
            classifier = Classifier(hidden_dim, num_classes)

            train_autoencoder(autoencoder, train_loader, num_epochs=10, optimizer_lr=lr)
            train_classifier(autoencoder.encoder, classifier, train_loader, num_epochs=epochs, optimizer_lr=lr)

            accuracy = evaluate_model(autoencoder.encoder, classifier, test_loader)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lr = lr
                best_batch_size = batch_size
                print(f"New best accuracy: {best_accuracy:.2f}% with lr = {best_lr}, batch_size = {best_batch_size}")

    print(f"Best Learning Rate: {best_lr}, Best Batch Size: {best_batch_size}, Best Accuracy: {best_accuracy:.2f}%")
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')
    torch.save(classifier.state_dict(), 'classifier.pth')


if __name__ == '__main__':
    main()