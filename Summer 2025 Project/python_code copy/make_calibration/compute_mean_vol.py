

import pandas as pd
import sys
from datetime import datetime
import data_model as dm
import opt_functions as opt_f
import report_functions as report_f
import scipy.stats
from numpy import sqrt, log, exp, pi
import numpy as np
import funzioni_lmm as lmm_f
from scipy import optimize
import time

# Press the green button in the gutter to run the script.

def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



"""
def compute_swp_price(vol, t_exp, t_mat, tenor, rf_curve, strike, shift, call_type):

    rf_times  = rf_curve['TIME']
    rf_values = rf_curve['VALUE']

    vol = vol/100.0
    price = fromVolaATMToPrice(t_exp, t_mat, tenor, vol, rf_times, rf_values, shift, call_type, type_curve='rate')

    return price
"""

def Pt_MKT(p_time, rf_times, rf_values):

    RateTime = np.interp(p_time, rf_times, rf_values)
    p_out = exp(- RateTime * p_time)

    return p_out


def black_price(c_p, fwd, strike, shift, r, T, sigma):

    fwd = fwd +  shift
    N = scipy.stats.norm.cdf


    d1 = (log(fwd/strike) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    if c_p == 'c':
        price = N(d1)*fwd - N(d2)*strike*exp(-r*T)
    elif c_p == 'p':
        price = N(-d2)*strike*exp(-r*T)-N(-d1)*fwd
    else:
        price = N(d1)*fwd - N(d2)*strike*exp(-r*T)

    return price


def select_swp_ptf(ptf_label, df_pf_data):


    idx_ptf = df_pf_data['PTF_LABEL']==ptf_label

    df_ptf = df_pf_data[idx_ptf]
    df_ptf = df_ptf.reset_index()
    df_ptf = df_ptf.drop(['index'], axis = 1)

    return df_ptf


def select_model_prms_data(mdl_label, df_mdl_data):


    idx_ptf = df_mdl_data['MDL_NAME']==mdl_label
    df_mdl = df_mdl_data[idx_ptf]

    prm_dict = {}

    for i in range(0, len(df_mdl)):
        rec_ = df_mdl.iloc[i]
        prm_name  = rec_[1]
        prm_x0    = rec_[2]
        prm_min   = rec_[3]
        prm_max   = rec_[4]

        prm_dict[prm_name] = {}
        prm_dict[prm_name]['Name'] = prm_name
        prm_dict[prm_name]['x0'] = prm_x0
        prm_dict[prm_name]['min'] = prm_min
        prm_dict[prm_name]['max'] = prm_max



    return prm_dict, df_mdl


def select_swp_mkt_data(date_tmp, df_swp_mkt_data):

    date_tmp_ = datetime.strptime(date_tmp, '%d/%m/%Y')

    #print(df_swp_mkt_data.head())
    #FQ(66)
    idx_date = df_swp_mkt_data['DATES']==date_tmp_
    idx_exp  = df_swp_mkt_data['DATES']=='EXPIRY'
    idx_mat  = df_swp_mkt_data['DATES']=='MATURITY'

    exp_data = df_swp_mkt_data[idx_exp]
    mat_data = df_swp_mkt_data[idx_mat]

    swap_mkt_data = df_swp_mkt_data[idx_date]

    exp_to_concat = exp_data.T
    mat_to_concat = mat_data.T
    swp_to_concat = swap_mkt_data.T


    #print('idx_exp: ', idx_exp)
    #print('mat_to_concat: ', mat_to_concat)
    #print('exp_to_concat: ', exp_to_concat)

    df_mkt = pd.concat([exp_to_concat, mat_to_concat, swp_to_concat], axis=1)

    df_mkt = df_mkt.rename(columns={1:'EXPIRY', 0:'MATURITY', 79:'VALUES', 106:'VALUES', 86:'VALUES',
                                    2:'VALUES', 3:'VALUES', 7:'VALUES', 5:'VALUES', 13:'VALUES', 9:'VALUES',
                                    11:'VALUES', 55:'VALUES', 31:'VALUES'})
    df_mkt = df_mkt.drop(['DATES'])


    return df_mkt





def select_rf_curve(date_tmp, df_mkt_curves):

    idx_date = df_mkt_curves['DATES']==date_tmp
    rf_curve_out = df_mkt_curves[idx_date].values.tolist()[0][1:]
    col = df_mkt_curves.columns.values.tolist()[1:]

    rf_curve_val = []
    time_curve_val = []
    rf_curve_disc_val = []


    for i in range(0, len(col)):
        t_label_tmp = col[i]
        time_tmp = dm.term_dict[t_label_tmp]
        val_tmp = rf_curve_out[i]
        df_tmp  = np.exp(-time_tmp*val_tmp)
        rf_curve_val.append(val_tmp)
        time_curve_val.append(time_tmp)
        rf_curve_disc_val.append(df_tmp)

    time_curve_val = np.asarray(time_curve_val)
    rf_curve_val   = np.asarray(rf_curve_val)

    curve_dict = {'TIME': time_curve_val, 'VALUE': rf_curve_val, 'DISCOUNT': rf_curve_disc_val}

    return curve_dict


def select_data_to_calib(df_swp_vols, df_swp_data_to_calib, flag_exp, w_limit_min, w_limit_max):

    df_out = pd.DataFrame({  'EXPIRY': [], 'MATURITY': [], 'VALUES': []})


    for i in range(0, len(df_swp_data_to_calib)):


        record_tmp = df_swp_data_to_calib.iloc[i]
        exp_tmp = record_tmp['EXPIRY']
        mat_tmp = record_tmp['MATURITY']
        weight_tmp = record_tmp['WEIGHT']

        if (weight_tmp > w_limit_min) & (weight_tmp < w_limit_max):

            if (flag_exp == True):

                idx_exp      = df_swp_vols['EXPIRY'] == exp_tmp
                df_swp_vols_ = df_swp_vols[idx_exp]
                idx_mat      = df_swp_vols_['MATURITY'] == mat_tmp

                df_swap = df_swp_vols_[idx_mat]

                if (len(df_swap))> 0:

                    df_out = df_out.append(df_swap, ignore_index=True)

            else:


                idx_mat      = df_swp_vols['MATURITY'] == mat_tmp
                df_swp_vols_ = df_swp_vols[idx_mat]
                idx_exp      = df_swp_vols_['EXPIRY'] == exp_tmp


                df_swap = df_swp_vols_[idx_exp]


                if (len(df_swap))> 0:

                    df_out = df_out.append(df_swap, ignore_index=True)

        else:

            continue


    return df_out


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def mdl_prms_from_dict_to_list(mdl_prm_opt, mdl_flag):


    if (mdl_flag in ['G2++', 'G1++']):
        prm_list_labels = dm.g2pp_prm_list
    #elif (mdl_flag == 'G1++'):
    #    prm_list_labels = dm.g1pp_prm_list
    elif (mdl_flag == 'CIR1++'):
        prm_list_labels = dm.cir1pp_prm_list
    elif (mdl_flag == 'VSCK'):
        prm_list_labels = dm.vsck_prm_list
    elif (mdl_flag == 'LMM'):
        prm_list_labels = dm.lmm_prm_list
    else:
        prm_list_labels = dm.vsck_prm_list

    nlab = len(prm_list_labels)
    prm_list_values = []
    for i in range(0, nlab):

        prm_label_tmp = prm_list_labels[i]
        prm_value_tmp = mdl_prm_opt[prm_label_tmp]
        prm_list_values.append(prm_value_tmp)


    return prm_list_values


def set_mdl_calib_results(df_swp_data, rf_curve_dict, mdl_flag, shift_ref, mdl_prm_opt):


    mdl_prm_list = mdl_prms_from_dict_to_list(mdl_prm_opt, mdl_flag)

    df_swp_data_n = df_swp_data.copy()

    for i in range(0, len(df_swp_data)):

        #print('df_swp_data_n: ', df_swp_data_n)
        #FQ(777)
        exp_tmp = df_swp_data_n.iloc[i]['EXPIRY']
        mat_tmp = df_swp_data_n.iloc[i]['MATURITY']
        strike_tmp = df_swp_data_n.iloc[i]['STRIKE']
        shift_tmp = df_swp_data_n.iloc[i]['SHIFT']

        exp_tmp_yy = dm.term_dict[exp_tmp]
        mat_tmp_yy = dm.term_dict[mat_tmp]

        price_tmp = opt_f.compute_swaption_prices_by_model(mdl_prm_list,
                                                           strike_tmp,
                                                           exp_tmp_yy,
                                                           mat_tmp_yy,
                                                           shift_tmp,
                                                           shift_ref,
                                                           rf_curve_dict,
                                                           mdl_flag)

        """
        print('======SET RESULTS =========')
        print('strike_tmp: ', strike_tmp)
        print('exp_tmp: ', exp_tmp_yy)
        print('mat_tmp: ', mat_tmp_yy)
        print('shift_tmp: ', shift_tmp)
        print('price_res: ', price_tmp)
        print('=============>>>>>>')
        """
        #FQ(88888)
        df_swp_data_n.at[i, 'MDL_DATA'] = price_tmp

    df_swp_data_n['MKT_DATA']=df_swp_data_n['PRICE']

    #print(df_swp_data_n)
    x2_out = report_f.computeCHI2(df_swp_data_n)


    return  df_swp_data_n, x2_out








def model_calibration(mkt_prices_df, mkt_prices_dict, swp_data, shift_data, rf_curve_dict, mdl_label_tmp, prm_data_dict, shift_ref, calib_type):


    x0_list     = []
    bound_list  = []

    for prm_tmp in prm_data_dict.keys():

        x0_tmp = prm_data_dict[prm_tmp]['x0']
        min_tmp = prm_data_dict[prm_tmp]['min']
        max_tmp = prm_data_dict[prm_tmp]['max']

        x0_list.append(x0_tmp)
        bound_list.append([min_tmp, max_tmp])


    tenr = 0.5
    call_flag = 1.0



    #ff = optimize.minimize(opt_f.loss_test_model, x0_list, args =(mkt_prices_dict, rf_curve_dict, mdl_label_tmp), method='TNC', bounds=bound_list, options= {'maxiter': 10})
    ff = optimize.minimize(opt_f.loss_calib_model, x0_list, args =(mkt_prices_dict, rf_curve_dict, mdl_label_tmp, shift_ref), method='TNC', bounds=bound_list)

    prm_data_opt = {}

    prm_list = list(prm_data_dict.keys())

    ln = len(prm_list)
    for i in range(0, ln):
        kk = prm_list[i]
        prm_data_opt[kk] = ff.x[i]

    return prm_data_opt


def to_date_str(date_tmp):

    ds = date_tmp.split('/')
    date_label = ds[0] + ds[1] + ds[2]

    return  date_label




def compute_swp_strike(t_exp, t_tenor, rf_curve):


    t_mat = t_exp + t_tenor
    time_ts    = rf_curve['TIME']
    zc_rates   = rf_curve['VALUE']

    zc_exp_rates = np.interp(t_exp, time_ts, zc_rates)
    zc_mat_rates = np.interp(t_mat, time_ts, zc_rates)

    z_exp = np.exp(-zc_exp_rates*t_exp)
    z_mat = np.exp(-zc_mat_rates*t_mat)

    dt_pay = 0.5
    n_pay = (t_mat - t_exp)/dt_pay
    n_pay = int(np.round(n_pay,1))

    annuity = 0.0
    for i in range(1, n_pay + 1):

        t_tmp  = t_exp + i*dt_pay
        zc_rate_tmp = np.interp(t_tmp, time_ts, zc_rates)
        zc_price_tmp = np.exp(-t_tmp*zc_rate_tmp)
        annuity = zc_price_tmp*dt_pay + annuity

    num = z_exp - z_mat
    den = annuity



    swap = num/den

    return swap







def set_mkt_price_dict(df_swp_data, shift_data, rf_curve_dict):


    mkt_price_dict = {}
    exp_list = list(set(df_swp_data['EXPIRY'].tolist()))
    n_exp_list = len(exp_list)
    mat_list_n = []

    for i in range(0, n_exp_list):


        exp_tmp = exp_list[i]
        mkt_price_dict[exp_tmp] = {}

        idx_exp   = df_swp_data['EXPIRY'] == exp_tmp
        idx_exp_s = shift_data['EXPIRY'] == exp_tmp

        df_swp_data_ = df_swp_data[idx_exp]
        shift_data_  = shift_data[idx_exp_s]

        mat_list_ = df_swp_data_['MATURITY'].tolist()

        for mat_tmp in mat_list_:

            #print('mat_tmp: ', mat_tmp)
            idx_exp_mat   = df_swp_data_['MATURITY'] == mat_tmp
            idx_exp_mat_s = shift_data_['MATURITY'] == mat_tmp

            vol_tmp = df_swp_data_[idx_exp_mat]['VALUES'].values[0]
            shift_tmp = shift_data_[idx_exp_mat_s]['VALUES']

            shift_tmp = shift_data_[idx_exp_mat_s]['VALUES'].values[0]

            t_exp_tmp = dm.term_dict[exp_tmp]
            t_mat_tmp = dm.term_dict[mat_tmp]

            shift_tmp = shift_tmp/100.0
            vol_tmp = vol_tmp/100.0
            tenor = 0.5
            call_type = 1.0

            price_tmp = lmm_f.fromVolaATMToPrice(t_exp_tmp, t_mat_tmp, tenor, vol_tmp, rf_curve_dict['TIME'], rf_curve_dict['VALUE'], shift_tmp, call_type, type_curve='rate')
            strike_tmp, annuity_tmp = lmm_f.SwapRate(tenor, rf_curve_dict['TIME'], rf_curve_dict['VALUE'], t_exp_tmp, t_mat_tmp, type_curve = 'rate')

            mkt_price_dict[exp_tmp][mat_tmp] = {'strike': strike_tmp, 'price': price_tmp, 'exp_val': t_exp_tmp, 'mat_val': t_mat_tmp, 'shift_val': shift_tmp}


    return  mkt_price_dict


def compute_zc_price(time_to_interp, rf_curve, regime):

    time_curve     = rf_curve['TIME']
    zc_rates_curve = rf_curve['VALUE']

    zc_rate_out = np.interp(time_to_interp, time_curve, zc_rates_curve)
    if regime == 'c':
        zc_price_out = np.exp(-zc_rate_out*time_to_interp)
    else:
        zc_price_out = np.exp(-zc_rate_out*time_to_interp)

    return zc_price_out, zc_rate_out

def set_mkt_price_df(df_swp_data, shift_data, rf_curve_dict):

    df_swp_data_n = df_swp_data.copy()

    for i in range(0, len(df_swp_data)):

        exp_tmp = df_swp_data_n.iloc[i]['EXPIRY']
        mat_tmp = df_swp_data_n.iloc[i]['MATURITY']
        vol_tmp = df_swp_data_n.iloc[i]['VALUES']

        vol_tmp = vol_tmp/100.0

        idx_exp = shift_data['EXPIRY'] == exp_tmp
        shift_data_ = shift_data[idx_exp]

        idx_mat = shift_data_['MATURITY'] == mat_tmp
        shift_tmp = shift_data_[idx_mat]['VALUES'].values[0]
        shift_tmp = shift_tmp/100.0

        t_exp_tmp = dm.term_dict[exp_tmp]
        t_mat_tmp = dm.term_dict[mat_tmp]

        #strike_tmp   = compute_swp_strike(exp_val_tmp, mat_val_tmp, rf_curve_dict)
        zc_price_tmp, zc_rate_tmp = compute_zc_price(t_exp_tmp, rf_curve_dict, 'c')
        zc_price_1m_tmp, zc_rate_1m_tmp = compute_zc_price(1.0/12, rf_curve_dict, 'c')
        zc_price_10y_tmp, zc_rate_10y_tmp = compute_zc_price(10.0, rf_curve_dict, 'c')
        zc_price_30y_tmp, zc_rate_30y_tmp = compute_zc_price(30.0, rf_curve_dict, 'c')

        slope_10y_tmp = zc_rate_10y_tmp - zc_rate_1m_tmp
        slope_30y_tmp = zc_rate_30y_tmp - zc_rate_1m_tmp

        tenor = 0.5
        call_type = +1

        #price_tmp    = compute_swp_price(val_tmp, exp_val_tmp, zc_rate_tmp, strike_tmp, shift_tmp)
        #price_tmp    = compute_swp_price(val_tmp, exp_val_tmp, mat_val_tmp, tenor, rf_curve_dict, strike_tmp, shift_tmp, call_type)


        #print('t_exp_tmp: ', t_exp_tmp)
        #print('t_mat_tmp: ', t_mat_tmp)
        #print('tenor: ', tenor)
        #print('vol_tmp: ', vol_tmp)
        #print('shift_tmp: ', shift_tmp)

        price_tmp = lmm_f.fromVolaATMToPrice(t_exp_tmp, t_mat_tmp, tenor, vol_tmp, rf_curve_dict['TIME'], rf_curve_dict['VALUE'], shift_tmp, call_type, type_curve='rate')
        strike_tmp, ForwardAnnuityPrice = lmm_f.SwapRate(tenor, rf_curve_dict['TIME'], rf_curve_dict['VALUE'], t_exp_tmp, t_mat_tmp, type_curve='rate')

        #print('price_tmp: ', price_tmp)
        #FQ(999)
        #print('shift_tmp: ', shift_tmp)
        df_swp_data_n.at[i, 'STRIKE']    = strike_tmp
        df_swp_data_n.at[i, 'PRICE']     = price_tmp
        df_swp_data_n.at[i, 'SHIFT']     = shift_tmp
        df_swp_data_n.at[i, 'ZC_PRICE']  = zc_price_tmp
        df_swp_data_n.at[i, 'ZC_RATE']   = zc_rate_tmp
        df_swp_data_n.at[i, 'SLOPE_10Y']   = slope_10y_tmp
        df_swp_data_n.at[i, 'SLOPE_30Y']   = slope_30y_tmp



    return  df_swp_data_n





if __name__ == '__main__':




    #FQ(666)
    #date_list = ['30/04/2020', '30/07/2020', '31/12/2019', '31/12/2018']
    #date_list = ['31/12/2019']
    #date_list = ['31/03/2022']
    #ptf_label_list = ['RU5Y']
    #ptf_label_list = ['30YL20191231_p001_002']


    mdl_list = ['LMM', 'G2++', 'G1++']
    #mdl_list = ['LMM']
    #mdl_list = ['G1++']

    t0 = time.time()
    calib_type = 'Standard'

    flag_plot = True
    flag_save = True

    w_limit_min = 0
    w_limit_max = 100.5

    w_limit = w_limit_min
    flag_exp = False
    #flag_exp = True

    # LOSS FUNCTIONS vedi Brigo pg. 60 e 77
    # CHK BLACK: prima setto STRIKE = FWD, dopo alzo il fwd con lo shift FWD = FWD + shift??
    # GESTIONE DELLA CURVA (i.e. PP) PER I MODELLI PP
    # LOSS FUNCTIONS



    rf_mkt_curves =     pd.read_excel(open('input/dati_swaptions_v8.xlsx', 'rb'), sheet_name='CURVES')
    swp_mkt_data =      pd.read_excel(open('input/dati_swaptions_v8.xlsx', 'rb'), sheet_name='SWAPTIONS_DATA')
    shift_mkt_data_1 =  pd.read_excel(open('input/dati_swaptions_v8.xlsx', 'rb'), sheet_name='SHIFT_DATA_1')
    #mdl_prms_data =     pd.read_excel(open('input/dati_swaptions_v8.xlsx', 'rb'), sheet_name='MDL_DATA')
    ptf_swp_dataset =   pd.read_excel(open('input/dati_swaptions_v8.xlsx', 'rb'), sheet_name='PTF_DATA')

    #ptf_label_list = ptf_swp_dataset['PTF_LABEL'].tolist()
    #shift_mkt_data =    pd.read_excel(open('input/dati_swaptions_v5.xlsx', 'rb'), sheet_name='SHIFT_DATA')

    idx_ = (ptf_swp_dataset['WEIGHT'] > w_limit_min) & (ptf_swp_dataset['WEIGHT'] < w_limit_max)
    ptf_label_list = ptf_swp_dataset[idx_]['PTF_LABEL'].tolist()

    ptf_label_list = list(set(ptf_label_list))

    #FQ(777)
    date_for_report = []
    ptf_for_report  = []
    mdl_for_report  = []
    chi2_for_report = []
    opt_prm_report  = []

    calib_type_for_report = []
    calib_label_for_report = []


    k = 1


    #print('len(ptf_label_list): ', len(ptf_label_list))
    #FQ(88)

    ptf_code_list = []
    vol_mean_list = []
    zc_mean_list = []
    slope_10y_list = []
    slope_30y_list = []

    for ptf_label_tmp in ptf_label_list:


        xx = ptf_label_tmp.split('_')[0]
        ln = len(xx)

        yy = ptf_label_tmp[ln-8:ln-4]
        mm = ptf_label_tmp[ln-4:ln-2]
        dd = ptf_label_tmp[ln-2:ln]



        date_tmp = dd + '/' + mm + '/' + yy

        #print('date_tmp: ', date_tmp)
        #print('rf_mkt_curves: ', rf_mkt_curves)
        rf_curve_dict = select_rf_curve(date_tmp, rf_mkt_curves)

        swp_vols = select_swp_mkt_data(date_tmp, swp_mkt_data)
        swp_shift = select_swp_mkt_data(date_tmp, shift_mkt_data_1)

        date_str_tmp = to_date_str(date_tmp)

        swp_data_to_calib = select_swp_ptf(ptf_label_tmp, ptf_swp_dataset)

        shift_data        = select_data_to_calib(swp_shift, swp_data_to_calib, flag_exp, w_limit_min,w_limit_max )
        swp_data          = select_data_to_calib(swp_vols, swp_data_to_calib, flag_exp, w_limit_min, w_limit_max)

        swp_data          = select_data_to_calib(swp_vols, swp_data_to_calib, flag_exp, w_limit_min, w_limit_max)
        mkt_price_dict    = set_mkt_price_dict(swp_data, shift_data, rf_curve_dict)
        mkt_price_df      = set_mkt_price_df(swp_data, shift_data, rf_curve_dict)

        vol_tmp = mkt_price_df['VALUES'].mean()
        zc_tmp = mkt_price_df['ZC_RATE'].mean()

        slope_10y_tmp = mkt_price_df['SLOPE_10Y'].mean()
        slope_30y_tmp = mkt_price_df['SLOPE_30Y'].mean()

        #print('slope_30y_tmp: ', mkt_price_df['SLOPE_30Y'])

        ptf_code_list.append(ptf_label_tmp)
        vol_mean_list.append(vol_tmp)
        zc_mean_list.append(zc_tmp)
        slope_10y_list.append(slope_10y_tmp)
        slope_30y_list.append(slope_30y_tmp)

        #!!! MKT PRICE DICT

    data_dict = {'ptf_code':ptf_code_list, 'vol_mean':vol_mean_list, 'zc_rate_mean':zc_mean_list, 'slope_10y':slope_10y_list, 'slope_30y':slope_30y_list}
    df_out = pd.DataFrame(data = data_dict)

    print('df_out: ', df_out.head())

    df_out.to_excel('mkt_scenario_n.xlsx', engine='xlsxwriter')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
