from torch import nn

cel_fn = nn.CrossEntropyLoss()
mse_fn = nn.MSELoss(reduction="none")