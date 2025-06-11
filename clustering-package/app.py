import numpy as np
import scipy as sp

X = np.array([[0, 0, 1], [0,6, 7]])
X = sp.sparse.csr_matrix(X)

print(X.nnz)

