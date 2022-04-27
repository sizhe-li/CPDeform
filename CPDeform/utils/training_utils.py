import torch
import numpy as np

DEFAULT_DTYPE = torch.float32
DEFAULT_DEVIC = torch.device('cuda')


def set_default_tensor_type(dtypecls):
    global DEFAULT_DTYPE
    torch.set_default_tensor_type(dtypecls)
    if dtypecls is torch.DoubleTensor:
        DEFAULT_DTYPE = torch.float64
    else:
        DEFAULT_DTYPE = torch.float32


def np2th(nparr):
    dtype = DEFAULT_DTYPE if nparr.dtype in (np.float64, np.float32) else None
    return torch.from_numpy(nparr).to(device=dev(), dtype=dtype)


def dev():
    return DEFAULT_DEVIC


class EarlyStopper:
    def __init__(self, patience=None, delta=0.):
        if patience is None:
            patience = np.inf

        self.patience = patience
        self.patience_cnt = 0
        self.best_loss = np.inf
        self.best_idx = None
        self.delta = delta

    def __call__(self, curr_loss, index):

        stopping = False
        improved = False
        if curr_loss < self.best_loss + self.delta:
            self.best_loss = curr_loss
            self.best_idx = index
            self.patience_cnt = 0
            improved = True
        else:
            self.patience_cnt += 1
            if self.patience_cnt >= self.patience:
                print('early stopping!')
                stopping = True

        return stopping, improved

    def get_best_index(self):
        return self.best_idx

    def reset(self, loss):
        self.patience_cnt = 0
        self.best_loss = loss
        self.best_idx = None
