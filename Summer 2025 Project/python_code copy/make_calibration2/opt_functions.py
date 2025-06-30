
import numpy as np
import lmm_functions as lmm_f
import sys





def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()

def compute_zc_ns_rate(list_model_params, T):
    b0 = list_model_params[0]
    b1 = list_model_params[1]
    b2 = list_model_params[2]
    tau = list_model_params[3]

    if (T < 0.0001):

        model_rate = b0 + b1

    else:

        tmp1 = T / tau
        g1 = np.exp(-tmp1)
        G1 = 1.0 - g1

        model_rate = b0 + (b1 + b2) * (G1 / tmp1) - b2 * g1

    return model_rate


def loss_zc_model_ns(list_model_params, mkt_prices_dict):
    diff_sum = 0.0

    time_mkt_list = mkt_prices_dict.keys()
    time_mkt_list.sort()

    for time_tmp in time_mkt_list:
        time_tmp = float(time_tmp)
        model_price_tmp = compute_zc_ns_rate(list_model_params, time_tmp)
        mkt_price_tmp = mkt_prices_dict[time_tmp]
        diff = abs(model_price_tmp - mkt_price_tmp)
        diff = diff * diff
        diff_sum = diff_sum + diff

    return diff_sum

#########################################################
###
########################################################



def compute_swaptions_price_test(prm_g2pp, strike_tmp, t_exp, t_mat):

    mdl_value = 100.0

    return mdl_value



def compute_swaptions_price_by_g2pp(prm_g2pp, strike_tmp, t_exp, t_mat, rf_curve_dict):

    dt_d = 1.0 / 2.0
    n_max = 20
    rf_times = rf_curve_dict['TIME']
    rf_values = rf_curve_dict['VALUE']
    call_flag = 1

    prm_g2pp = np.asarray(prm_g2pp)
    #mdl_value = price_swaption(prm_g2pp, t_exp, t_mat, dt_d, strike_tmp, rf_times, rf_values, call_flag, n_max)

    mdl_value = 100

    return mdl_value


def compute_swaptions_price_by_lmm(list_model_params, strike_tmp, exp_tmp, mat_tmp, shift, shift_ref, curve_dict):

    tenor_data = 0.5
    call_type = 1.0

    #print('tx: ', time.time())
    vol_ = lmm_f.LMM_vola_swaptions_Rebonato(param=list_model_params,
                                        curve=curve_dict,
                                        shift=shift,
                                        swap_rate=strike_tmp,
                                        t_exp=exp_tmp,
                                        maturity_swap=mat_tmp,
                                        tenor_swap=tenor_data)

    #vol_ = 0.3
    #shift = shift

    mdl_value = lmm_f.fromVolaATMToPrice(t_exp=exp_tmp,
                                        maturity= mat_tmp,
                                        tenor=tenor_data,
                                        vol=vol_,
                                        rf_times=curve_dict['TIME'],
                                        rf_values=curve_dict['VALUE'],
                                        shift=shift_ref,
                                        call_type=call_type,
                                        type_curve='rate')

    #print('exp_tmp: ', exp_tmp)
    #print('mat_tmp: ', mat_tmp)
    #print('vol_: ', vol_)
    #print('shift: ', shift)
    #print('mdl_value: ', mdl_value)
    #FQ(666)
    return mdl_value

def compute_swaptions_price_by_vsck(list_model_params, strike_tmp, exp_tmp, mat_tmp):

    mdl_value  = compute_swaptions_price_test(list_model_params, strike_tmp, exp_tmp, mat_tmp)

    return mdl_value


def compute_swaptions_price_by_cir1pp(list_model_params, strike_tmp, exp_tmp, mat_tmp):

    mdl_value  = compute_swaptions_price_test(list_model_params, strike_tmp, exp_tmp, mat_tmp)

    return mdl_value



def compute_swaptions_price_by_g1pp(prm_g1pp, strike_tmp, t_exp, t_mat, rf_curve_dict):

    dt_d = 1.0 / 2.0
    n_max = 20
    rf_times = rf_curve_dict['TIME']
    rf_values = rf_curve_dict['VALUE']
    call_flag = 1


    # prm_g1pp: alfa, sigma
    # prm_g2pp: alfa, gamma, sigma, eta, rho

    prm_g2pp = [prm_g1pp[0], 0.00001, prm_g1pp[1], 0.00001, 0.00001]
    prm_g2pp = np.asarray(prm_g2pp)
    mdl_value = price_swaption(prm_g2pp, t_exp, t_mat, dt_d, strike_tmp, rf_times, rf_values, call_flag, n_max)


    return mdl_value


#def compute_swaptions_price_by_g1pp(list_model_params, strike_tmp, exp_tmp, mat_tmp):

    #    p0 = list_model_params[0]
    #p1 = list_model_params[1]

    #mdl_value = p0*exp_tmp +  p1*mat_tmp

    #return mdl_value






def compute_swaption_prices_by_model(list_model_params, strike_tmp, exp_tmp_yy, mat_tmp_yy, shift_tmp, shift_ref, rf_curve_dict, mdl_flag):

    #if (mdl_flag in ['G1++','G2++'] ):
    #    model_price_tmp = compute_swaptions_price_by_g2pp(list_model_params, strike_tmp, exp_tmp_yy, mat_tmp_yy, rf_curve_dict)
    #elif (mdl_flag == 'G1++'):
    #    model_price_tmp = compute_swaptions_price_by_g1pp(list_model_params, strike_tmp, exp_tmp_yy, mat_tmp_yy, rf_curve_dict)
    if (mdl_flag == 'CIR1++'):
        model_price_tmp = compute_swaptions_price_by_cir1pp(list_model_params, strike_tmp, exp_tmp_yy, mat_tmp_yy)
    elif (mdl_flag == 'LMM'):
        model_price_tmp = compute_swaptions_price_by_lmm(list_model_params, strike_tmp, exp_tmp_yy, mat_tmp_yy, shift_tmp, shift_ref, rf_curve_dict)
    elif (mdl_flag == 'VSCK'):
        model_price_tmp = compute_swaptions_price_by_vsck(list_model_params, strike_tmp, exp_tmp_yy, mat_tmp_yy)
    else:
        print('Not working!')
        print('Not working! %s model not available!!'%(mdl_flag))
        FQ(77)



    return  model_price_tmp

def loss_calib_model(list_model_params, mkt_prices_dict, rf_curve_dict, mdl_flag, shift_ref):

    diff_sum = 0.0
    exp_mkt_list = mkt_prices_dict.keys()

    for exp_tmp in exp_mkt_list:

        for mat_tmp in mkt_prices_dict[exp_tmp].keys():

            strike_tmp      = mkt_prices_dict[exp_tmp][mat_tmp]['strike']
            mkt_price_tmp   = mkt_prices_dict[exp_tmp][mat_tmp]['price']

            exp_tmp_yy      = mkt_prices_dict[exp_tmp][mat_tmp]['exp_val']
            mat_tmp_yy      = mkt_prices_dict[exp_tmp][mat_tmp]['mat_val']
            shift_tmp       = mkt_prices_dict[exp_tmp][mat_tmp]['shift_val']

            model_price_tmp = compute_swaption_prices_by_model(list_model_params, strike_tmp, exp_tmp_yy, mat_tmp_yy, shift_tmp, shift_ref, rf_curve_dict, mdl_flag)
            diff = abs(model_price_tmp - mkt_price_tmp)
            diff = diff*diff
            diff_sum = diff_sum + diff


    return diff_sum


def compute_zc_cir_rate(list_model_params, T):

    r0 = list_model_params[0]
    kappa = list_model_params[1]
    theta = list_model_params[2]
    sigma = list_model_params[3]

    h = (kappa * kappa + 2.0 * sigma * sigma) ** (0.5)

    g0 = 2 * kappa * theta / (sigma * sigma)
    g1 = np.exp(T * h) - 1.0
    g2 = np.exp(T * (h + kappa) / 2.0)

    A0 = (2 * h * g2 / (2.0 * h + (kappa + h) * g1))
    B0 = (2.0 * g1 / (2.0 * h + (kappa + h) * g1))

    model_rate = -(g0 * np.log(A0) - B0 * r0) / T


    return model_rate


def loss_zc_model_cir(list_model_params, mkt_prices_dict):
    diff_sum = 0.0

    time_mkt_list = mkt_prices_dict.keys()
    time_mkt_list.sort()

    for time_tmp in time_mkt_list:
        time_tmp = float(time_tmp)
        model_price_tmp = compute_zc_cir_rate(list_model_params, time_tmp)
        mkt_price_tmp = mkt_prices_dict[time_tmp]
        diff = abs(model_price_tmp - mkt_price_tmp)
        diff = diff * diff
        diff_sum = diff_sum + diff

    return diff_sum

