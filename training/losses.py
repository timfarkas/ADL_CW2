from torch import nn

cel_fn = nn.CrossEntropyLoss(ignore_index=-1)
mse_fn = nn.MSELoss(reduction="none")