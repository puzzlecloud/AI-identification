
import math
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset,random_split
from tqdm import tqdm
import gc

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

# configurations 设置需要用到的参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'select_all': True,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 1000,  # Number of epochs.
    'batch_size': 80,
    'learning_rate': 1e-5,
    'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
    'model_path': './models/model.ckpt',  # Your model will be saved here.
    'model_path_dnn': './models/model_dnn.ckpt',

    # model parameters
    'input_dim': 561,
    'hidden_layers': 1,
    'hidden_dim': 512
}

class PhoneIdentityDataset(Dataset):
    '''Dataset for loading and preprocessing the phone identity data'''
    def __init__(self, x, y=None, augment=False, noise_level=0.01):
        self.x = torch.FloatTensor(x.astype(np.float32))
        if y is not None:
            self.y = torch.LongTensor(y.astype(np.int64))
        else:
            self.y = None
        self.augment = augment
        self.noise_level = noise_level

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.augment:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        if self.y is not None:
            return x, self.y[idx]
        else:
            return x

# Use the dataset with augmentation





class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=6, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class My_Model_DNN(nn.Module):
    def __init__(self, input_dim):
        super(My_Model_DNN, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 6)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


# 选择features,划分x,y
def select_feat(train_data, valid_data, test_data, select_all=True):
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]

    # 转化分类的标签，变成数字
    y = y_train
    for i in range(len(y)):
        if y[i] == 'WALKING_UPSTAIRS':  y[i] = 0
        if y[i] == 'WALKING_DOWNSTAIRS':  y[i] = 1
        if y[i] == 'WALKING':  y[i] = 2
        if y[i] == 'SITTING':  y[i] = 3
        if y[i] == 'STANDING':  y[i] = 4
        if y[i] == 'LAYING':  y[i] = 5
    y_train = y

    y = y_valid
    for i in range(len(y)):
        if y[i] == 'WALKING_UPSTAIRS':  y[i] = 0
        if y[i] == 'WALKING_DOWNSTAIRS':  y[i] = 1
        if y[i] == 'WALKING':  y[i] = 2
        if y[i] == 'SITTING':  y[i] = 3
        if y[i] == 'STANDING':  y[i] = 4
        if y[i] == 'LAYING':  y[i] = 5
    y_valid = y

    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data
    y_train = y_train.astype(np.int32)
    y_valid = y_valid.astype(np.int32)
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [i for i in range(1, 4)]  # 选择合适的feature

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


train_data, test_data = pd.read_csv('./train.csv').values, pd.read_csv('./test.csv').values
print(train_data.dtype)

#Load Data
train_data, test_data = pd.read_csv('./train.csv').values, pd.read_csv('./test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed']) #valid data从训练集划分

print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

#Print out the number of features.
print(f'number of features: {x_train.shape[1]}')


#final data_set
#final data_set with augmentation for the training dataset
train_dataset = PhoneIdentityDataset(x_train, y_train, augment=True, noise_level=0.01)
valid_dataset = PhoneIdentityDataset(x_valid, y_valid)
test_dataset = PhoneIdentityDataset(x_test)


# remove raw feature to save memory
del x_train, y_train, x_valid, y_valid
gc.collect()

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# data,target = train_dataset[0]
# print(data.shape)
# print(target)

# Train
def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    best_acc = 0.0
    writer = SummaryWriter()  # Writer of tensoboard
    step = 0

    for epoch in range(config['n_epochs']):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training mode
        model.train()  # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)  # crossEntropy会把标签转化为one-hot vector形式
            loss.backward()
            optimizer.step()
            step += 1
            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

            writer.add_scalar('Loss/train', train_loss, step)

            # validation mode
        if len(valid_dataset) > 0:
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(tqdm(valid_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)

                    loss = criterion(outputs, labels)

                    _, val_pred = torch.max(outputs, 1)
                    val_acc += (
                                val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                    val_loss += loss.item()
                    writer.add_scalar('Loss/valid', val_loss, step)

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, config['n_epochs'], train_acc / len(train_dataset), train_loss / len(train_loader),
                    val_acc / len(valid_dataset), val_loss / len(valid_loader)
                ))
                # writer.add_scalar('Loss/valid', val_loss, step)

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), config['model_path'])
                    print('saving model with acc {:.3f}'.format(best_acc / len(valid_dataset)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, config['n_epochs'], train_acc / len(train_dataset), train_loss / len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(valid_dataset) == 0:
        torch.save(model.state_dict(), config['model_path'])
        print('saving model at last epoch')


# Set seed for reproducibility
same_seed(config['seed'])

model = Classifier(input_dim=config['input_dim'], hidden_layers=config['hidden_layers'], hidden_dim=config['hidden_dim']).to(device)

trainer(train_loader, valid_loader, model, config, device)


#这里用来测试，加载模型，加载权重
model = Classifier(input_dim=config['input_dim'], hidden_layers=config['hidden_layers'], hidden_dim=config['hidden_dim']).to(device)
model.load_state_dict(torch.load(config['model_path']))


test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)
        # pred  = test_pred.cpu().numpy()
    print(pred.shape)


with open('submission.csv', 'w') as f:
    f.write('Activity\n')
    for i,y in enumerate(pred):
        if y == 0: y = 'WALKING_UPSTAIRS'
        if y == 1: y = 'WALKING_DOWNSTAIRS'
        if y == 2: y = 'WALKING'
        if y == 3: y = 'SITTING'
        if y == 4: y = 'STANDING'
        if y == 5: y ='LAYING'
        f.write('{}\n'.format(y))