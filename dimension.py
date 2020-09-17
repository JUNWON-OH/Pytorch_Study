#%%
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = torch.unsqueeze(x, 0)
x = torch.squeeze(x)
x = x.view(9)
print(x)
print(x.size())
print(x.shape)
print(x.ndimension())


# %%
