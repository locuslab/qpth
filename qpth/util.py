import torch

def print_header(msg):
    print('===>', msg)

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().numpy()

def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))

def get_sizes(G, A=None):
    if G.dim() == 2:
        nineq, nz = G.size()
        nBatch = 1
    elif G.dim() == 3:
        nBatch, nineq, nz = G.size()
    if A is not None:
        neq = A.size(1) if A.ndimension() > 0 else 0
    else:
        neq = None
    # nBatch = batchedTensor.size(0) if batchedTensor is not None else None
    return nineq, nz, neq, nBatch

def bdiag(d):
    nBatch, sz = d.size()
    D = torch.zeros(nBatch, sz, sz).type_as(d)
    I = torch.eye(sz).repeat(nBatch,1,1).type(torch.cuda.ByteTensor) # TODO
    D[I] = d.squeeze()
    return D
