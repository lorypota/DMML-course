import numpy as np

#3.b
def get_beta_RSS_3_b(X,y):
    X_T=X.T
    M=X_T@X
    if(np.linalg.det(M)<0.0001):#it is not invertible
        print("The matrix is not invertible!!!")
        return None
    else:
        inverse=np.linalg.inv(M)
        beta=inverse@X_T@y
        return beta 

#3.c
def get_beta_3_c(X,y,lamb):
    n = X.shape[0]
    X_T = X.T
    M = X_T@X
    M = M + np.diag([lamb*n] * M.shape[0])
    inverse = np.linalg.inv(M)
    beta = inverse@X_T@y
    return beta 





