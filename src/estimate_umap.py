
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import numpy as np
# from src.load_model import load_model


# df_all_frames=pickle.load(open("./data/models/df_all_frames.pkl","rb"))
# X,y=df_all_frames_to_X_y(df_all_frames)

model=pickle.load(open('model_mailabs_umap_2_gmm_300.pkl','rb'))

# model = load_model(300,18,train_set='librispeech') # 'MAILABS' or 'librispeech'

# np.array(X).shape
# reduced_X=model.reducer.transform(X)
# reduced_X.shape


# scaler = preprocessing.RobustScaler().fit(model.X_train_reduced)
# X_scaled = scaler.transform(model.X_train_reduced)

# plt.clf()
# fig, axs = plt.subplots(len(X_scaled.T), 1)
# for i in range(len(X_scaled.T)):
#     x=X_scaled[:,i]
#     axs[i].hist(x, bins=100)
# plt.savefig('hists.png')

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Get cpu or gpu device for training.
import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, target_dims, loss_fn=nn.MSELoss()):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, target_dims)
        )
        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits

    def fit(self, X, y):
        tensor_x = torch.Tensor(np.array(X)) # transform to torch tensor
        tensor_y = torch.Tensor(np.array(y))

        my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
        my_dataloader = DataLoader(my_dataset, shuffle=True) # create your dataloader
        train(my_dataloader, self, self.loss_fn, self.optimizer)


# loss_fn = nn.CrossEntropyLoss()

nn_model = NeuralNetwork(target_dims=2).to(device)
nn_model.fit(np.array(model.X_train), np.array(model.X_train_reduced))




# https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader





# nn_model(tensor_x)
# tensor_y



pickle.dump(nn_model, open('data/models/nn_model_umap_estimator.pkl',"wb"))
# pickle.dump(scaler, open('data/models/scaler_nn_model_umap_estimator.pkl',"wb"))