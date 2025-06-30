import numpy as np
import pandas as pd
from scipy.stats import norm, ncx2
import sys


from scipy import optimize
from scipy.special import erf



def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def phi_cdf(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0


def newtons_method(f, df, x0,a1,a2,a3, a4, a5, a6, a7, a8, a9, a10, a11, e):

    delta = abs(0. - f(x0, a1, a2, a3,a4,a5,a6,a7,a8,a9,a10,a11))
    while delta > e:
        x0 = x0 - f(x0, a1, a2, a3,a4,a5,a6,a7,a8,a9,a10,a11) / df(x0, a1, a2, a3,a4,a5,a6,a7,a8,a9,a10,a11)
        delta = abs(0. - f(x0, a1, a2, a3,a4,a5,a6,a7,a8,a9,a10,a11))

    return x0

def compute_N(x, t_exp, t_mat, sigma_x, sigma_y, rho_xy, mu_x, mu_y, y_star,call_flag, a1,s1,b1,s2,rho1, _coupon_array, _coupon_time_array,  rf_times, rf_values):


    sum_n = 0.0

    T = t_exp

    j_max = len(_coupon_time_array)

    for j in range(0, j_max):
        c_i = _coupon_array[j]
        t_i = _coupon_time_array[j]

        b_i = B_ztT(b1, T, t_i)

        lambda_x = c_i * A2_tT(T, t_i, rf_times, rf_values, a1, s1, b1, s2, rho1) * np.exp(-B_ztT(a1, T, t_i) * x)

        h1_x = h1_function(x, y_star, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

        h2_x = h1_x + b_i * sigma_y * np.sqrt(1. - rho_xy * rho_xy)

        k_x = - b_i * (mu_y - 0.5 * (1. - rho_xy * rho_xy) * sigma_y * sigma_y * b_i + rho_xy * sigma_y * (
                (x - mu_x) / sigma_x))

        phi_x = phi_cdf(-call_flag * h2_x)

        sum_n = sum_n + lambda_x * np.exp(k_x) * phi_x

    return sum_n


def h1_function(x, y_bar, mu_x, mu_y, sigma_x, sigma_y, rho_xy):

    rde = np.sqrt(1. - rho_xy * rho_xy)
    z1 = (y_bar - mu_y) / (sigma_y * rde)
    z2 = rho_xy * (x - mu_x) / (sigma_x * rde)

    return z1 - z2


def compute_ystar(y_star, _t_exp,_x,  a1, s1, b1, s2, rho1, _coupon_time_array, _coupon_array, rf_times, rf_values):

    tmp = 0.0
    T = _t_exp

    jmax = len(_coupon_time_array)

    for j in range(0, jmax):
        PtT = price_model(T, _coupon_time_array[j], _x, y_star, a1,s1,b1,s2,rho1, rf_times, rf_values)

        cpn_df_tmp = _coupon_array[j] * PtT
        tmp = tmp + cpn_df_tmp

    aa = tmp - 1.0

    return aa


def compute_ystar_derivate(y_star,_t_exp,_x,  a1, s1, b1, s2, rho1,
                            _coupon_time_array, _coupon_array, rf_times, rf_values):

    tmp = 0.0
    T = _t_exp

    jmax = len(_coupon_time_array)

    for j in range(0, jmax):
        t_i = _coupon_time_array[j]

        PtT = price_model(T, t_i, _x, y_star, a1,s1,b1,s2,rho1, rf_times, rf_values)

        cpn_df_tmp = _coupon_array[j] * PtT * (-B_ztT(b1, T, t_i))
        tmp = tmp + cpn_df_tmp

    aa = tmp

    return aa

def price_model( t, T, xt, yt, a1, s1, b1, s2, rho1, rf_times,rf_values):

    a = A2_tT(t, T, rf_times, rf_values, a1, s1, b1, s2, rho1)
    Ba = B_ztT(a1, t, T)
    Bb = B_ztT(b1, t, T)
    out = a * np.exp(- Ba * xt - Bb * yt)

    return out

def B_ztT(z,t,T):

    BtT = dgt(z, T - t) / z

    return BtT


def Pt_MKT_c(time, rf_times,rf_values):

    RateTime = np.interp(time, rf_times, rf_values)
    p_out = np.exp(- RateTime * time)

    return p_out


def A2_tT(t, T, rf_times, rf_values, a1, s1, b1, s2, rho1):

    P_mkt_0_t = Pt_MKT_c(t, rf_times, rf_values)
    P_mkt_0_T = Pt_MKT_c(T, rf_times, rf_values)

    v_tT = V_tT(t, T, a1,s1,b1,s2,rho1)
    v_0T = V_tT(0, T, a1,s1,b1,s2,rho1)
    v_0t = V_tT(0, t, a1,s1,b1,s2,rho1)

    A1 = 0.5 * (v_tT - v_0T + v_0t)
    AtT = (P_mkt_0_T / P_mkt_0_t) * np.exp(A1)

    return AtT


def V_tT(t, T, a, sigma, b, eta, rho):

    z1 = (sigma * sigma) / (a * a) * (
            T - t + (2 / a) * np.exp(-a * (T - t)) - (1 / (2 * a)) * np.exp(-2 * a * (T - t)) - (3 / (2 * a)))
    z2 = (eta * eta) / (b * b) * (T - t + (2 / b) * np.exp(-b * (T - t)) - (1 / (2 * b)) * np.exp(-2 * b * (T - t)) - (3 / (2 * b)))
    z3 = 2 * rho * (sigma * eta) / (a * b) * (T - t + (np.exp(-a * (T - t)) - 1) / a + (np.exp(-b * (T - t)) - 1) / b - (np.exp(-(a + b) * (T - t)) - 1) / (a + b))
    return z1 + z2 + z3




def M_x_T(s,t,T,a1,s1,b1,s2, rho1):

    g1a_Tt = np.exp(-a1 * (T - t))
    g1a_Tt2s = np.exp(-a1 * (T + t - 2. * s))

    g1b_Tt = np.exp(-b1 * (T - t))
    g1ab = np.exp(-b1 * T - a1 * t + (a1 + b1) * s)

    G1 = dgt(a1, t - s)
    G2 = g1a_Tt - g1a_Tt2s
    G3 = g1b_Tt - g1ab

    h1 = (s1 * s1) / (a1 * a1) + (rho1 * s1 * s2) / (a1 * b1)
    h2 = (s1 * s1) / (2.0 * a1 * a1)
    h3 = (rho1 * s1 * s2) / (b1 * (a1 + b1))

    m_x_T = h1 * G1 - h2 * G2 - h3 * G3

    return m_x_T


def M_y_T(s,t,T,a1,s1,b1,s2,rho1):

    g1b_Tt = np.exp(-b1 * (T - t))
    g1b_Tt2s = np.exp(-b1 * (T + t - 2. * s))

    g1a_Tt = np.exp(-a1 * (T - t))
    g1ba = np.exp(-a1 * T - b1 * t + (a1 + b1) * s)

    G1 = dgt(b1, t - s)
    G2 = g1b_Tt - g1b_Tt2s
    G3 = g1a_Tt - g1ba

    h1 = (s2 * s2) / (b1 * b1) + (rho1 * s1 * s2) / (a1 * b1)
    h2 = (s2 * s2) / (2. * b1 * b1)
    h3 = (rho1 * s1 * s2) / (a1 * (a1 + b1))

    m_y_T = h1 * G1 - h2 * G2 - h3 * G3

    return  m_y_T



def dgt(g, t):
    out = 1. - np.exp(-g * t)
    return out

def compute_swaptions_price_by_g2pp(prm_g2pp, t_exp, t_mat, dt_d, strike_tmp, rf_times, rf_values, call_flag, n_max):

    swp_atm_d = strike_tmp

    t_mat_n = t_exp + t_mat
    T = t_exp

    a1 = prm_g2pp[0]
    s1 = prm_g2pp[1]
    b1 = prm_g2pp[2]
    s2 = prm_g2pp[3]
    rho1 = prm_g2pp[4]

    G2a = dgt(2. * a1, T)
    G2b = dgt(2. * b1, T)
    Gab = dgt(a1 + b1, T)

    sigma_x = s1 * np.sqrt(G2a / (2. * a1))
    sigma_y = s2 * np.sqrt(G2b / (2. * b1))
    rho_xy = (rho1 * s1 * s2) / ((a1 + b1) * sigma_x * sigma_y) * Gab

    mu_x = -M_x_T(0, T, T, a1, s1, b1, s2, rho1)
    mu_y = -M_y_T(0, T, T, a1, s1, b1, s2, rho1)

    # ---------- set coupon time -----------------

    k_max = int((t_mat_n - t_exp) / dt_d)

    _coupon_time_array = np.zeros(k_max, dtype='double')
    _coupon_array = np.zeros(k_max, dtype='double')

    for k in range(1, k_max):
        _coupon_time_array[k - 1] = t_exp + dt_d * k
        _coupon_array[k - 1] = swp_atm_d * dt_d

    _coupon_time_array[k_max - 1] = t_exp + dt_d * k_max
    _coupon_array[k_max - 1] = 1. + swp_atm_d * dt_d

    sum_int = 0.0

    x_min = mu_x - 5. * sigma_x
    x_max = mu_x + 5. * sigma_x

    dx = (x_max - x_min) / n_max

    i_max = int(n_max)

    y_star_0 = 0.0001
    for i in range(0, i_max):
        x_i = x_min + dx * i

        ff = optimize.root(compute_ystar, y_star_0, method='lm',
                           args=(
                           t_exp, x_i, a1, s1, b1, s2, rho1, _coupon_time_array, _coupon_array, rf_times, rf_values))
        y_star_n = ff.x[0]

        # y_star_n = newtons_method(compute_ystar, compute_ystar_derivate, y_star_0,
        #                           t_exp, x_i, a1,s1,b1,s2,rho1, _coupon_time_array,
        #                           _coupon_array, rf_times, rf_values, 1e-12)

        h1_x = h1_function(x_i, y_star_n, mu_x, mu_y, sigma_x, sigma_y, rho_xy)

        gauss_x = np.exp(-0.5 * (((x_i - mu_x) * (x_i - mu_x)) / (sigma_x * sigma_x))) / (sigma_x * np.sqrt(2. * np.pi))

        phi_h1_x = phi_cdf(-call_flag * h1_x)

        sum_phi = compute_N(x_i, t_exp, t_mat_n, sigma_x, sigma_y, rho_xy, mu_x, mu_y, y_star_n, call_flag, a1, s1, b1,
                            s2, rho1,
                            _coupon_array, _coupon_time_array, rf_times, rf_values)

        int_tmp = gauss_x * (phi_h1_x - sum_phi)

        sum_int = sum_int + int_tmp * dx

    p_0_T = price_model(0, t_exp, 0., 0., a1, s1, b1, s2, rho1, rf_times, rf_values)

    mdl_value = call_flag * p_0_T * sum_int


    return mdl_value