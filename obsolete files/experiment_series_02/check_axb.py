import numpy as np
from numpy.linalg import solve
from numpy.linalg import norm

false_cnt = 0
total_cnt = 0
for r in range(10000):
    b = np.array([-5, 0])
    sa = np.random.rand(1)
    A = np.random.rand(2, 2) * sa
    sd = np.random.rand(1) + 10
    D = np.random.rand(2, 2) * sd
    A[:, 1] *= -1
    D[:, 1] *= -1

    M = A + D
    x1, x2 = solve(A, b), solve(M, b)
    n1, n2 = norm(x1), norm(x2)

    criterion = lambda N: -b[0] * np.sqrt((N[1,1]**2+N[1,0]**2)/np.square(N[0,0]*N[1,1]-N[0,1]*N[1,0]))

    if criterion(M) < criterion(A):
        if np.all(x1 > 0) and np.all(x2 > 0):
            total_cnt += 1
            if n1 < n2:
                # print("------------- A -------------")
                # print("|   JIE/JII   |   JEE/JEI   |")
                # print(f"|    {A[0,1]/A[1,1]:.3f}    |    {A[0,0]/A[1,0]:.3f}    |")
                # print("------------- M -------------")
                # print("|   JIE/JII   |   JEE/JEI   |")
                # print(f"|    {M[0,1]/M[1,1]:.3f}    |    {M[0,0]/M[1,0]:.3f}    |")
                # print("------------- D -------------")
                # print("|   JIE/JII   |   JEE/JEI   |")
                # print(f"|    {D[0, 1] / D[1, 1]:.3f}    |    {D[0, 0] / D[1, 0]:.3f}    |")
                print(f"criterion(M): {criterion(M):.3f}; criterion(A): {criterion(A):.3f}")
                print(f"n1:{n1:.3f}, n2:{n2:.3f}")
                print("\n")
                false_cnt += 1

print(f"{false_cnt}/{total_cnt} = {false_cnt/total_cnt :.4f}")
