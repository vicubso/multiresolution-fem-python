# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: science
#     language: python
#     name: python3
# ---

# %% [markdown]
# # train_2_by_2.py
# Main script to train the condensation network for a 2x2 MultiQ4 element

# %%
# %reset -f

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import inf
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from plane_elasticity import MultiQ4

# %% [markdown]
# ## Define model

# %%
nel_x = 2; nel_y = 2
nodes = np.array([[0,1],[0,0],[1,1],[1,0]]) # Coordinates of the nodes of the element.
nu = 1/3
D = 1/(1-nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
element = MultiQ4(nel_x, nel_y, nodes, D) 
net = element.get_condensation_net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Clear all weights in net
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
net.apply(init_weights)

# %% [markdown]
# ## Generate dataset

# %%
set_a = element.vertex_dofs # Set of active nodes for condensation (usually vertices)
n_a = len(set_a)
n_b = element.n_dofs - n_a
M = element.get_auxiliary_condensation_matrix(np.ones(nel_x*nel_y))

N_train = 100000
N_test = 300
N_total = N_train + N_test

E = np.random.rand(N_total, nel_x*nel_y) # Sample random Young moduli
M_flat = np.zeros((N_total, n_b*n_a)) # Save space for auxiliary matrices (Kbb^-1 Kba)

# %%
# Some checks
M = element.get_auxiliary_condensation_matrix(np.ones(nel_x*nel_y))
m = M.reshape(-1,order='F')

m.reshape(n_b,n_a,order='F') - M

# %%
# TODO: Save them to disk and load them from disk (takes a long time to compute)
for i in range(N_total):
    M_flat[i,:] = element.get_auxiliary_condensation_matrix(E[i,:]).reshape(n_b*n_a,order='F')

# %%
inputs = torch.zeros((N_total, element.n_elm), dtype=torch.float32) # Inputs are the element densities
targets = torch.zeros((N_total, n_b*n_a), dtype=torch.float32) # Targets are the numerical shape functions (rather, the auxiliary matrix Kbb^-1 Kba) 
for i in range(N_total):
    inputs[i,:] = torch.tensor(E[i,:])
    targets[i,:] = torch.tensor(M_flat[i,:])

# %%
# Check that you did reshaping correctly
# should be zero for i=j, non-zero otherwise
i = 0; j = 0
np.linalg.norm(M_flat[i].reshape(n_b,n_a,order='F') - element.get_auxiliary_condensation_matrix(E[j,:]))

# %%
# Create data loaders
batch_size = 10

train_set = torch.utils.data.TensorDataset(inputs[0:N_train,:], targets[0:N_train,:])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torch.utils.data.TensorDataset(inputs[N_train:,:], targets[N_train:,:])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# %%
# Check dimensions
iterator = enumerate(train_loader)
index, data = next(iterator)
index, data[0].shape, data[1].shape
# data[0] is the input. size is (batch_size, n_elements)
# data[1] is the target. size is (batch_size, n_b*n_a)

# %%
# Check that reshaping and criterion works.
# loss should be zero for i=j, non zero otherwise
i = 0; j = 0
M1 = torch.tensor( element.get_auxiliary_condensation_matrix(E[i,:]))
M2 = torch.tensor(M_flat[j,:].reshape(n_b,n_a,order='F'))
criterion(M1,M2)


# %% [markdown]
# ### Training

# %%
# Function to calculate loss 
def evaluate(net, set="test"):
    with torch.no_grad():
        if set == "train": dataset = train_set
        elif set == "test": dataset = test_set
        outputs = net(dataset[:][0])
        labels = dataset[:][1]
        return criterion(outputs, labels).item()


# %%
print("Loss before training")
print("--------------------")
print(f"Loss on training data: {evaluate(net, 'train')}")
print(f"Loss on test data: {evaluate(net, 'test')}")

# %%
convergence = []
for epoch in range(10): 
    clear_output(wait=False)
    print(f'Epoch {epoch + 1}')
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data # Get a batch of data. We get batch_size examples and labels

        optimizer.zero_grad() # Set all gradients to zero

        # Forward, backward, optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels) * 1000
        loss.backward()
        optimizer.step()
        
        #convergence.append(evaluate(net, 'train'))
        
        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'Epoch {epoch + 1}, example {i+1}, running loss: {running_loss}')
            running_loss = 0.0
print("--------------------")
print("Loss after training")
print("--------------------")
print(f"Loss on training data: {evaluate(net, 'train')}")
print(f"Loss on test data: {evaluate(net, 'test')}")

# %%
plt.semilogy(convergence)
plt.grid()
# plt.savefig('convergence.pdf')

# %%
# i = np.random.randint(N_test) +  N_train
M = M_flat[i]
M_pred = net.forward(torch.tensor(E[i,:], dtype=torch.float32)).detach().numpy()

plt.plot(M, 'o-', label='True')
plt.plot(M_pred, 'x-', label='Predicted')
plt.grid()
plt.legend()
plt.savefig('learned_auxiliary_condensation_matrix.pdf')

# %%
error = np.abs(M-M_pred)np.abs(M)
plt.plot(error)
plt.grid()
# plt.savefig("relative_error.pdf")

# %% [markdown]
# ### Save model

# %%
# Save weights
PATH = './weights_2_by_2.pth'
torch.save(net.state_dict(), PATH) # Network is saved as a dictionary of tensors: {conv1.weight: tensor, conv1.bias: tensor, ...}
