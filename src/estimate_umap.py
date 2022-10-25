
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

from src.load_model import load_model


df_all_frames=pickle.load(open("./data/models/df_all_frames.pkl","rb"))
X,y=df_all_frames_to_X_y(df_all_frames)

model = load_model(300,18,train_set='librispeech') # 'MAILABS' or 'librispeech'

np.array(X).shape
reduced_X=model.reducer.transform(X)
reduced_X.shape


scaler = preprocessing.RobustScaler().fit(reduced_X)
X_scaled = scaler.transform(reduced_X)

plt.clf()
fig, axs = plt.subplots(len(X_scaled.T), 1)
for i in range(len(X_scaled.T)):
    x=X_scaled[:,i]
    axs[i].hist(x, bins=100)
plt.savefig('hists.png')

# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Get cpu or gpu device for training.
import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 18)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

nn_model = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(nn_model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3)

# from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

# https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

tensor_x = torch.Tensor(np.array(X)) # transform to torch tensor
tensor_y = torch.Tensor(X_scaled)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = DataLoader(my_dataset, shuffle=True) # create your dataloader


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

train(my_dataloader, nn_model, loss_fn, optimizer)

nn_model(tensor_x)
tensor_y

pickle.dump(nn_model, open('data/models/nn_model_umap_estimator.pkl',"wb"))
pickle.dump(scaler, open('data/models/scaler_nn_model_umap_estimator.pkl',"wb"))