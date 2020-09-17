#%%
import torch

w = torch.randn(5, 3, dtype=torch.float)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
b = torch.randn(5, 2, dtype=torch.float)
wx = torch.mm(w, x)
result = wx + b

print(w.size())
print(x.size())
print(b.size())
print(wx.size())
print(w)
print(x)
print(b)
print(wx)
print(result)

# %%
