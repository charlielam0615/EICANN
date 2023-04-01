import numpy as np
import matplotlib.pyplot as plt
import tqdm


def get_g(scale=1.):
    return (0.1+np.random.rand()) * scale


def occupy_mat(mat, rows, cols, value):
    mat[rows[0]:rows[1], cols[0]:cols[1]] = value
    return mat


def get_mat(gEE, gEI, gIE, gII, p=1.0):
    mask = (np.random.rand(num_E+num_I, num_E+num_I) < p).astype(np.float32)
    mat = np.zeros([num_E + num_I, num_E + num_I], dtype=np.float32)
    mat = occupy_mat(mat, [0, num_E], [0, num_E], gEE)
    mat = occupy_mat(mat, [0, num_E], [num_E, num_E + num_I], gEI)
    mat = occupy_mat(mat, [num_E, num_E + num_I], [0, num_E], gIE)
    mat = occupy_mat(mat, [num_E, num_E + num_I], [num_E, num_E + num_I], gII)
    mat = mat * mask
    return mat


def set_diag_zero(mat):
    assert len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]
    mask = 1.0 - np.eye(mat.shape[0], dtype=np.float32)
    return mat * mask


num_E = 300
num_I = 300
prob = 0.5
EI_scale = 1.0/np.sqrt(300.)
A_scale = 1.0/600.
a_larray = np.zeros(100)
am_larray = np.zeros(100)
for r in range(100):
    gEE, gEI, gIE, gII = get_g(A_scale), get_g(A_scale), -get_g(A_scale), -get_g(A_scale)
    A = get_mat(gEE, gEI, gIE, gII, p=prob)
    a_w, a_v = np.linalg.eig(A)
    ra_w = np.real(a_w)
    a_lambda = np.max(ra_w)

    if a_lambda > 1:
        A = A / a_lambda
        a_w, a_v = np.linalg.eig(A)
        ra_w = np.real(a_w)
        a_lambda = np.max(ra_w)

    while True:
        JEE, JEI, JIE, JII = get_g(EI_scale), get_g(EI_scale), -get_g(EI_scale), -get_g(EI_scale)
        if JIE/JII > JEE/JEI:
            break

    M = get_mat(JEE, JEI, JIE, JII, p=prob)
    # M = np.random.rand(num_E+num_I, num_E+num_I)/np.sqrt(300.)
    M = set_diag_zero(M)

    m_w, m_v = np.linalg.eig(M)
    rm_w = np.real(m_w)
    m_lamda = np.max(rm_w)

    am_w, am_v = np.linalg.eig(A+M)
    ram_w = np.real(am_w)
    am_lambda = np.max(ram_w)

    if am_lambda > a_lambda:
        print("False")
    else:
        print(f"lambda_a:{a_lambda:.4f}, lambda_m:{am_lambda:.4f}")

    a_larray[r] = a_lambda
    am_larray[r] = am_lambda

plt.hist(am_larray-a_larray, bins=50, alpha=0.5)
plt.show()