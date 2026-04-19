import torch as tr

def num_factors(K):
    """
    result[i] = number of v factors contributing to ith position in kronecker expansion
    """
    result = tr.tensor([0,1])
    for k in range(1,K):
        result = tr.cat([result, 1+result])
    return result

class KroneckerMemory:
    def __init__(self, K):
        self.K = K
        self.m = 1 / tr.maximum(num_factors(K), tr.tensor(1.))
        self.b = 1 - self.m

    def init(self):
        return tr.zeros(self.K, 2**self.K)

    def activation(self, x):
        return self.m * x + tr.where(x > 0, self.b, -self.b)

    def embed(self, index):
        """
        embed integer index in {0,...,2**K-1}
        """
        return tr.tensor([1.-2*int(d) for d in format(index, f'0{self.K}b')])

    def read(self, M, a):
        return (M * a[...,None,:]).sum(dim=-1) / a.shape[-1]

    def write(self, M, a, v):
        v_old = self.read(M, a)
        M = M + (v - v_old)[...,:,None] * a[...,None,:]
        return M

    def expand(self, v):
        _1v = tr.stack([tr.ones(v.shape[-1]), v]).t()
        a = _1v[0]
        for row in _1v[1:]: a = tr.kron(a, row)
        return self.activation(a)


if __name__ == "__main__":

    nf = num_factors(3)
    assert (nf == tr.tensor([0,1,1,2,1,2,2,3])).all()

    km = KroneckerMemory(3)

    M = tr.randn(5,3,8)
    a = tr.randn(5,8)
    v = tr.randn(5,3)
    assert km.read(M,a).shape == (5,3)
    assert km.write(M,a,v).shape == (5,3,8)

    print("embeddings:")
    for i in range(8):
        print(km.embed(i))

    v = km.embed(5).requires_grad_()
    a = km.expand(v)
    print("v",v)
    print("a",a)
    for i in range(8):
        a[i].backward(retain_graph=True)
        print(f"da[{i}]/dv", v.grad)
        v.grad *= 0

