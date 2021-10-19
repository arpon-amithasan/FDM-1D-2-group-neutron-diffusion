import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import linalg
import time


def main():

    # iteration count

    it_num = int(1000)

    # geometry

    u = int(20)
    m = int(380)
    l = int(20)
    h = float(1)

    # Neutron group yield fractions

    del_g_1 = float(1)
    del_g_2 = float(0)

    # Fuel neutronic data

    f_s_12 = float(0.018193)   # group 1
    f_d_1 = float(1.0933622)
    f_a_1 = float(0.0092144)
    f_f_1 = float(0.0065697)
    f_r_1 = f_a_1 + f_s_12

    f_s_21 = float(0.0013089)  # group 2
    f_d_2 = float(0.3266693)
    f_a_2 = float(0.0778104)
    f_f_2 = float(0.13126)
    f_r_2 = f_a_2 + f_s_21

    # Upper Reflector neutronic data

    u_s_12 = float(0.0255380)  # group 1
    u_d_1 = float(1.1245305)
    u_a_1 = float(0.0008996)
    u_f_1 = float(0)
    u_r_1 = u_a_1 + u_s_12

    u_s_21 = float(0.0001245)  # group 2
    u_d_2 = float(0.7503114)
    u_a_2 = float(0.0255590)
    u_f_2 = float(0)
    u_r_2 = u_a_2 + u_s_21

    # Lower Reflector neutronic data

    l_s_12 = float(0.0255220)  # group 1
    l_d_1 = float(1.1251378)
    l_a_1 = float(0.0008984)
    l_f_1 = float(0)
    l_r_1 = l_a_1 + l_s_12

    l_s_21 = float(0.0001231)  # group 2
    l_d_2 = float(0.7501763)
    l_a_2 = float(0.0255600)
    l_f_2 = float(0)
    l_r_2 = l_a_2 + l_s_21

    # Initial guesses

    phi_1_old = np.ones((u + m + l, 1), dtype=float)
    phi_2_old = np.ones((u + m + l, 1), dtype=float)
    k_old = float(1)

    start_time = time.perf_counter()

    # Coefficient Matrices

    A_1 = make_big_ugly_matrix(u_d_1, f_d_1, l_d_1, u_r_1, f_r_1, l_r_1, h, u, m, l)
    A_2 = make_big_ugly_matrix(u_d_2, f_d_2, l_d_2, u_r_2, f_r_2, l_r_2, h, u, m, l)
    F_1 = make_diagonal_matrix(u_f_1, f_f_1, l_f_1, u, m, l)
    F_2 = make_diagonal_matrix(u_f_2, f_f_2, l_f_2, u, m, l)
    s_12 = make_diagonal_matrix(u_s_12, f_s_12, l_s_12, u, m, l)
    s_21 = make_diagonal_matrix(u_s_21, f_s_21, l_s_21, u, m, l)

    # Solve

    for i in range(0, it_num):
        S_1 = (del_g_1 / k_old) * (F_1 * phi_1_old + F_2 * phi_2_old) + s_21 * phi_2_old
        S_2 = (del_g_2 / k_old) * (F_1 * phi_1_old + F_2 * phi_2_old) + s_12 * phi_1_old

        phi_1_new, x_1 = sp.linalg.gmres(A_1, S_1)
        phi_2_new, x_2 = sp.linalg.gmres(A_2, S_2)
        k_new = k_old * np.sum(h * (F_1 * phi_1_new + F_2 * phi_2_new)) / np.sum(h * (F_1 * phi_1_old + F_2 * phi_2_old))

        phi_1_old = phi_1_new
        phi_2_old = phi_2_new
        k_old = k_new

    end_time = time.perf_counter()

    # output

    print(end_time - start_time, "seconds")
    print(k_old)
    plt.plot(phi_1_old, label = "group 1")
    plt.plot(phi_2_old, label = "group 2")
    plt.xlabel("length")
    plt.ylabel("flux")
    plt.title("Flux distribution of 1-dimensional reactor core")
    plt.show()


def make_diagonal_matrix(u_val, m_val, l_val, u, v, l):
    a = sp.lil_matrix((u + v + l, u + v + l), dtype=float)
    for i in range(0, u):
        a[i, i] = u_val
    for i in range(u, u + v):
        a[i, i] = m_val
    for i in range(u + v, u + v + l):
        a[i, i] = l_val

    a = a.tocsr()
    return a


def make_big_ugly_matrix(u_d, m_d, l_d, u_r, m_r, l_r, h, u, m, l):

    a = sp.lil_matrix((u + m + l, u + m + l), dtype=float)

    a[0, 0] = ((2 * u_d) / (h ** 2)) + u_r                          # upper reflector first point
    a[0, 1] = (- u_d) / (h ** 2)                                    # Dirichlet Boundary Condition

    for i in range(1, u):                                           # upper reflector
        a[i, i - 1] = (- u_d) / (h ** 2)
        a[i, i] = ((2 * u_d) / (h ** 2)) + u_r
        a[i, i + 1] = (- u_d) / (h ** 2)

    for i in range(u, u + m):               # fuel
        a[i, i - 1] = (- m_d) / (h ** 2)
        a[i, i] = ((2 * m_d) / (h ** 2)) + m_r
        a[i, i + 1] = (- m_d) / (h ** 2)

    for i in range(u + m, u + m + l - 1):                           # lower reflector
        a[i, i - 1] = (- l_d) / (h ** 2)
        a[i, i] = ((2 * l_d) / (h ** 2)) + l_r
        a[i, i + 1] = (- l_d) / (h ** 2)

    a[u + m + l - 1, u + m + l - 2] = (- l_d) / (h ** 2)            # lower reflector first point
    a[u + m + l - 1, u + m + l - 1] = ((2 * l_d) / (h ** 2)) + l_r  # Dirichlet Boundary Condition

    a = a.tocsr()

    return a


main()
