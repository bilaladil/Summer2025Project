import sys
import pandas as pd
import numpy as np
import scipy.stats
from numpy import sqrt, log, exp, pi
from scipy import optimize
from datetime import datetime
import time

import opt_functions as opt_f
import report_functions as report_f
import lmm_functions as lmm_f
import data_model as dm
import vsck_functions as vs





# Press the green button in the gutter to run the script.

def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()

#this function calculates the discount factor at p_time by interpolating between the risk free interest rates
#(rf_values) at specific times (rf_times)
def Pt_MKT(p_time, rf_times, rf_values):

    RateTime = np.interp(p_time, rf_times, rf_values)
    p_out = exp(- RateTime * p_time)

    return p_out


#this function generates the black scholes price for call and put options
def black_price(c_p, fwd, strike, shift, r, T, sigma):
#uses the forward price and a shift to make sure there are no negative values because you can't
#take the log of a negative
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


#selects specific swaption portfolios from a dataframe of portfolios
#checks the ones for which PTF_LABEL is whatever you input for it and then returns your dataframe
#with only these specifically labelled ones. 
def select_swp_ptf(ptf_label, df_pf_data):


    idx_ptf = df_pf_data['PTF_LABEL']==ptf_label

    df_ptf = df_pf_data[idx_ptf]
    df_ptf = df_ptf.reset_index()
    df_ptf = df_ptf.drop(['index'], axis = 1)

    return df_ptf


#similarly to above this function selects a model and its parameters from a dataframe of models and parameters
def select_model_prms_data(mdl_label, df_mdl_data):


    idx_ptf = df_mdl_data['MDL_NAME']==mdl_label
    df_mdl = df_mdl_data[idx_ptf]

    prm_dict = {}

#it extracts the name, initial value, min value and max value and stores them in a dictionary
    for i in range(0, len(df_mdl)):
        rec_ = df_mdl.iloc[i]
        prm_name  = rec_.iloc[1]
        prm_x0    = rec_.iloc[2]
        prm_min   = rec_.iloc[3]
        prm_max   = rec_.iloc[4]

        prm_dict[prm_name] = {}
        prm_dict[prm_name]['Name'] = prm_name
        prm_dict[prm_name]['x0'] = prm_x0
        prm_dict[prm_name]['min'] = prm_min
        prm_dict[prm_name]['max'] = prm_max

    return prm_dict, df_mdl


#this function takes market data on a specific date and then extracts the expiry and maturity
#for swaptions on this date
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

    df_mkt = df_mkt.rename(columns={1:'EXPIRY', 0:'MATURITY', 79:'VALUES', 106:'VALUES', 103:'VALUES', 86:'VALUES',
                                    2:'VALUES', 3:'VALUES', 7:'VALUES', 5:'VALUES', 13:'VALUES', 9:'VALUES',
                                    11:'VALUES', 55:'VALUES', 31:'VALUES'})
    df_mkt = df_mkt.drop(['DATES'])


    return df_mkt


#this function creates a dictionary of discount rates from the risk free curve at a specific date (date_tmp)
def select_rf_curve(date_tmp, df_mkt_curves):

    print('date_tmp: ', date_tmp)
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
        df_tmp  = np.exp(-time_tmp*val_tmp) #e^(-r*t) discount factor
        rf_curve_val.append(val_tmp)
        time_curve_val.append(time_tmp)
        rf_curve_disc_val.append(df_tmp)

    time_curve_val = np.asarray(time_curve_val)
    rf_curve_val   = np.asarray(rf_curve_val)

    curve_dict = {'TIME': time_curve_val, 'VALUE': rf_curve_val, 'DISCOUNT': rf_curve_disc_val}

    return curve_dict


#this function takes in a dataframe of swaps and their corresponding volatilities and aims
#to output a dataframe with swap data based on expiries, maturities and weights
def select_data_to_calib(df_swp_vols, df_swp_data_to_calib, flag_exp, w_limit_min, w_limit_max):

    df_out = pd.DataFrame({  'EXPIRY': [], 'MATURITY': [], 'VALUES': []})


    for i in range(0, len(df_swp_data_to_calib)):

#for each swaption in the data to calibrate we record the expiry, maturity and weight
        record_tmp = df_swp_data_to_calib.iloc[i]
        exp_tmp = record_tmp['EXPIRY']
        mat_tmp = record_tmp['MATURITY']
        weight_tmp = record_tmp['WEIGHT']

#each swaption is assigned a weight value depending on how important it is to the calibration of our model.
#we are only interested in the swaptions whose assigned weight values fall between w_limit_min and w_limit_max
        if (weight_tmp > w_limit_min) & (weight_tmp < w_limit_max):

#we have a flag that when true we sort by expiry first and when false we sort by maturity first
#sorting by one or the other can be useful if there are less of one than the other
#eg if there are only 20 expiry dates but 100 maturity dates then sorting by expiry first makes the
#computation much quicker and more efficient
            if (flag_exp == True):

                idx_exp      = df_swp_vols['EXPIRY'] == exp_tmp
                df_swp_vols_ = df_swp_vols[idx_exp]
                idx_mat      = df_swp_vols_['MATURITY'] == mat_tmp

#finds the data in df_swp_vols that has the same expiry and maturity as the entry from
#df_swp_data_to_calib and adds them to the new dataframe
                df_swap = df_swp_vols_[idx_mat]

                if (len(df_swap))> 0:
#if a match is found (a swaption with the same expriy and maturity as an option from df_swp_data_to_calib)
#then it is added to our output dataframe
                    df_out = df_out.append(df_swap, ignore_index=True)

            else:

                idx_mat      = df_swp_vols['MATURITY'] == mat_tmp
                df_swp_vols_ = df_swp_vols[idx_mat]
                idx_exp      = df_swp_vols_['EXPIRY'] == exp_tmp


                df_swap = df_swp_vols_[idx_exp]


                if (len(df_swap))> 0:

                    #df_out = df_out.append(df_swap, ignore_index=True)
                    df_out = pd.concat([df_out, df_swap], ignore_index=True)
        else:

            continue

    return df_out



#this function goes through a dictionary of model parameters and converts the one that we are interested in
#into a list
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

#tracks how many labels for the model 
    nlab = len(prm_list_labels)
    prm_list_values = []
    for i in range(0, nlab):

#stores the value associated with each label and appends the list with the value
        prm_label_tmp = prm_list_labels[i]
        prm_value_tmp = mdl_prm_opt[prm_label_tmp]
        prm_list_values.append(prm_value_tmp)

 
    return prm_list_values


def set_mdl_calib_results(df_swp_data, rf_curve_dict, mdl_flag, shift_ref, mdl_prm_opt):

    mdl_prm_list = mdl_prms_from_dict_to_list(mdl_prm_opt, mdl_flag)

    df_swp_data_n = df_swp_data.copy()

#filtering through each swaption
    for i in range(0, len(df_swp_data)):

        #print('df_swp_data_n: ', df_swp_data_n)
        #FQ(777)
#storing the expiry, maturity, strike and shift for each option
        exp_tmp = df_swp_data_n.iloc[i]['EXPIRY']
        mat_tmp = df_swp_data_n.iloc[i]['MATURITY']
        strike_tmp = df_swp_data_n.iloc[i]['STRIKE']
        shift_tmp = df_swp_data_n.iloc[i]['SHIFT']

#converts expiry and maturity to numerical values
        exp_tmp_yy = dm.term_dict[exp_tmp]
        mat_tmp_yy = dm.term_dict[mat_tmp]

#computes the swaption price
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
        
#stores the calculated price for each swaption 
        df_swp_data_n.at[i, 'MDL_DATA'] = price_tmp
        
#stores the market price for each swaption 
    df_swp_data_n['MKT_DATA']=df_swp_data_n['PRICE']

    #print(df_swp_data_n)
    
#computes the chi squared value between the calculated prices and market prices
    x2_out = report_f.computeCHI2(df_swp_data_n)


    return  df_swp_data_n, x2_out


#this function calibrates the model paramaters so that model prices match market prices as closely as 
#possible
def model_calibration(mkt_prices_dict, rf_curve_dict, mdl_label_tmp, prm_data_dict, shift_ref):


    x0_list     = []
    bound_list  = []

#stores initial guess and bounds from data provided
    for prm_tmp in prm_data_dict.keys():

        x0_tmp = prm_data_dict[prm_tmp]['x0']
        min_tmp = prm_data_dict[prm_tmp]['min']
        max_tmp = prm_data_dict[prm_tmp]['max']

        x0_list.append(x0_tmp)
        bound_list.append([min_tmp, max_tmp])
        
        
#uses scipy function "minimize" to minimise loss_calib_model using the Truncated Newton Conjugate-Gradient
#method
#loss_calib_model is a function that minimises the squared difference between model prices and 
#market prices
    #ff = optimize.minimize(opt_f.loss_test_model, x0_list, args =(mkt_prices_dict, rf_curve_dict, mdl_label_tmp), method='TNC', bounds=bound_list, options= {'maxiter': 10})
    ff = optimize.minimize(opt_f.loss_calib_model, x0_list, args =(mkt_prices_dict, rf_curve_dict, mdl_label_tmp, shift_ref), method='TNC', bounds=bound_list)

    prm_data_opt = {}
    prm_list = list(prm_data_dict.keys())
    ln = len(prm_list)
    
    for i in range(0, ln):
#kk stores the title of each parameter
        kk = prm_list[i]
#maps the optimised value back onto the respective parameter
        prm_data_opt[kk] = ff.x[i]

    return prm_data_opt



#splitting date into form DDMMYYYY (or YYYYMMDD [need to check how it is in the input data])
def to_date_str(date_tmp):

    ds = date_tmp.split('/')
    date_label = ds[0] + ds[1] + ds[2]

    return  date_label



#this function calculates the strike (par swap rate) for a swap that starts in t_exp years
#with a tenor of t_tenor using rf_curve (zero coupon rates)
def compute_swp_strike(t_exp, t_tenor, rf_curve):
 

    t_mat = t_exp + t_tenor
    time_ts    = rf_curve['TIME']
    zc_rates   = rf_curve['VALUE']

#interpolates along rf_curve to find the zc rate at expiry and maturity
    zc_exp_rates = np.interp(t_exp, time_ts, zc_rates)
    zc_mat_rates = np.interp(t_mat, time_ts, zc_rates)

#converts the interpolated rates to discount factors
    z_exp = np.exp(-zc_exp_rates*t_exp)
    z_mat = np.exp(-zc_mat_rates*t_mat)

    dt_pay = 0.5
    n_pay = (t_mat - t_exp)/dt_pay
    n_pay = int(np.round(n_pay,1))

    annuity = 0.0
    
    for i in range(1, n_pay + 1):
#calculates the present value of the fixed leg of payments
        t_tmp  = t_exp + i*dt_pay
        zc_rate_tmp = np.interp(t_tmp, time_ts, zc_rates)
        zc_price_tmp = np.exp(-t_tmp*zc_rate_tmp)
        annuity = zc_price_tmp*dt_pay + annuity

#calculates the present value of the floating leg of payments 
    num = z_exp - z_mat # present value of floating leg
    
    den = annuity #present value of fixed leg
    
    swap = num/den #par swap rate = present value of floating / present value of fixed

    return swap


#this function creates a dictionary of swaption market prices sorted by expiries and maturities
def set_mkt_price_dict(df_swp_data, shift_data, rf_curve_dict):


    mkt_price_dict = {}
#stores all the expiries and the number of expiries
    exp_list = list(set(df_swp_data['EXPIRY'].tolist()))
    n_exp_list = len(exp_list)
    mat_list_n = []

    for i in range(0, n_exp_list):

        exp_tmp = exp_list[i]
        mkt_price_dict[exp_tmp] = {}

        idx_exp   = df_swp_data['EXPIRY'] == exp_tmp
        idx_exp_s = shift_data['EXPIRY'] == exp_tmp

#filters the swaption data and shift data to only apply for the specific expiry we are on
        df_swp_data_ = df_swp_data[idx_exp]
        shift_data_  = shift_data[idx_exp_s]

        mat_list_ = df_swp_data_['MATURITY'].tolist()
        
#iterates through the corresponding maturities for the current expiry
        for mat_tmp in mat_list_:

            #print('mat_tmp: ', mat_tmp)
            idx_exp_mat   = df_swp_data_['MATURITY'] == mat_tmp
            idx_exp_mat_s = shift_data_['MATURITY'] == mat_tmp

#stores the volatility and shift for swaption with the specific expiry and maturity
            vol_tmp = df_swp_data_[idx_exp_mat]['VALUES'].values[0]
            shift_tmp = shift_data_[idx_exp_mat_s]['VALUES'].values[0]

#converts the expiry and maturity labels into numbers
            t_exp_tmp = dm.term_dict[exp_tmp]
            t_mat_tmp = dm.term_dict[mat_tmp]

#converts percentages to decimals
            shift_tmp = shift_tmp/100.0
            vol_tmp = vol_tmp/100.0
            tenor = 0.5
            call_type = 1.0 #payer swaption
            
#uses LMM to compute swaption prices using the at the money volatility 
#also computes par swap rates and annuities for the swap
            price_tmp = lmm_f.fromVolaATMToPrice(t_exp_tmp, t_mat_tmp, tenor, vol_tmp, rf_curve_dict['TIME'], rf_curve_dict['VALUE'], shift_tmp, call_type, type_curve='rate')
            strike_tmp, annuity_tmp = lmm_f.SwapRate(tenor, rf_curve_dict['TIME'], rf_curve_dict['VALUE'], t_exp_tmp, t_mat_tmp, type_curve = 'rate')

#stores the prices in a dictionary based on expiries and then maturities
            mkt_price_dict[exp_tmp][mat_tmp] = {'strike': strike_tmp, 'price': price_tmp, 'exp_val': t_exp_tmp, 'mat_val': t_mat_tmp, 'shift_val': shift_tmp}


    return  mkt_price_dict


def compute_zc_price(time_to_interp, rf_curve, regime):

    time_curve     = rf_curve['TIME']
    zc_rates_curve = rf_curve['VALUE']

#interpolates the ZC rate at time_to_interp
    zc_rate_out = np.interp(time_to_interp, time_curve, zc_rates_curve)

#calculates the ZC price using discount factor
    if regime == 'c':
        zc_price_out = np.exp(-zc_rate_out*time_to_interp)
####these two are doing the same thing?
    else:
        zc_price_out = np.exp(-zc_rate_out*time_to_interp)

    return zc_price_out, zc_rate_out


def set_mkt_price_df(df_swp_data, shift_data, rf_curve_dict):

    df_swp_data_n = df_swp_data.copy()

#goes through each swaption
    for i in range(0, len(df_swp_data)):
#stores expiry, maturity and volatility
        exp_tmp = df_swp_data_n.iloc[i]['EXPIRY']
        mat_tmp = df_swp_data_n.iloc[i]['MATURITY']
        vol_tmp = df_swp_data_n.iloc[i]['VALUES']
        
#converts volatility to decimal
        vol_tmp = vol_tmp/100.0
        
#filters data for the specific expiry
        idx_exp = shift_data['EXPIRY'] == exp_tmp
        shift_data_ = shift_data[idx_exp]

#filters data even further for the specific maturity
        idx_mat = shift_data_['MATURITY'] == mat_tmp
        shift_tmp = shift_data_[idx_mat]['VALUES'].values[0]

#converts shift to decimal
        shift_tmp = shift_tmp/100.0

#converts expiry and maturity to numbers
        t_exp_tmp = dm.term_dict[exp_tmp]
        t_mat_tmp = dm.term_dict[mat_tmp]

#computes ZC bond price and rate at expiry
        zc_price_tmp, zc_rate_tmp = compute_zc_price(t_exp_tmp, rf_curve_dict, 'c')
        tenor = 0.5
        call_type = +1 #payer swaption

#uses LMM to compute swaption prices using the at the money volatility 
#also computes par swap rates and annuities for the swap
        price_tmp = lmm_f.fromVolaATMToPrice(t_exp_tmp, t_mat_tmp, tenor, vol_tmp, rf_curve_dict['TIME'], rf_curve_dict['VALUE'], shift_tmp, call_type, type_curve='rate')
        strike_tmp, ForwardAnnuityPrice = lmm_f.SwapRate(tenor, rf_curve_dict['TIME'], rf_curve_dict['VALUE'], t_exp_tmp, t_mat_tmp, type_curve='rate')

#stores various data in the new dataframe relating to the swaptions
        df_swp_data_n.at[i, 'STRIKE']    = strike_tmp
        df_swp_data_n.at[i, 'PRICE']     = price_tmp
        df_swp_data_n.at[i, 'SHIFT']     = shift_tmp
        df_swp_data_n.at[i, 'ZC_PRICE']  = zc_price_tmp
        df_swp_data_n.at[i, 'ZC_RATE']   = zc_rate_tmp

    return  df_swp_data_n



#this means that the following code will only run in this file
if __name__ == '__main__':


    mdl_list = ['LMM', 'VSCK', 'CIR1++'] #list of interest rate models

    #mdl_list = ['LMM', 'VSCK', 'CIR1++']
    calib_type = 'Standard'
    flag_plot = False # plot yes or no
    flag_save = False # save the plot yes or no

    w_limit_min = 2019
    w_limit_max = 2023

    #label_out = "2013"
    #label_out = "17_18_19"
    label_out = "20_21_22"

    t0 = time.time() #starts a timer

    #w_limit_min = 20.5 #19.5
    #w_limit_max = 21.5 #20.5


    flag_exp = False
    #flag_exp = True

    # LOSS FUNCTIONS see Brigo pg. 60 and 77
    # CHK BLACK: first I set STRIKE = FWD, then I raise the fwd with the shift FWD = FWD + shift??
    # CURVE MANAGEMENT (i.e. PP) FOR PP MODELS
    # LOSS FUNCTIONS

#pulling data from an excel sheet
    rf_mkt_curves =     pd.read_excel(open('make_calibration2/input/dati_swaptions_v12.xlsx', 'rb'), sheet_name='CURVES')
    swp_mkt_data =      pd.read_excel(open('make_calibration2/input/dati_swaptions_v12.xlsx', 'rb'), sheet_name='SWAPTIONS_DATA')
    shift_mkt_data_1 =  pd.read_excel(open('input/dati_swaptions_v12.xlsx', 'rb'), sheet_name='SHIFT_DATA_1')
    mdl_prms_data =     pd.read_excel(open('input/dati_swaptions_v12.xlsx', 'rb'), sheet_name='MDL_DATA')
    ptf_swp_dataset =   pd.read_excel(open('input/dati_swaptions_v12.xlsx', 'rb'), sheet_name='PTF_DATA_N')

    w_limit = w_limit_min

#we want only the portfolios whose weight values fall within our bounds
    idx_ = (ptf_swp_dataset['WEIGHT'] > w_limit_min) & (ptf_swp_dataset['WEIGHT'] < w_limit_max)
#creates a list of all the portfolio names who fall within our bounds
    ptf_label_list = ptf_swp_dataset[idx_]['PTF_LABEL'].tolist()

    ptf_label_list = list(set(ptf_label_list))
    date_for_report = []
    ptf_for_report  = []
    mdl_for_report  = []
    chi2_for_report = []
    opt_prm_report  = []

    calib_type_for_report = []
    calib_label_for_report = []

    k = 1

    n_ptf = len(ptf_label_list)
#prints the number of suitable portfolios
    print('len(ptf_label_list): ', len(ptf_label_list))
    j = 1
    for ptf_label_tmp in ptf_label_list:

#prints what number portfolio we are on out of the total amount of suitable ones
        print('N. ptf elab: %s over %s' %(j,n_ptf))
        j = j + 1
        #print('len(ptf_label_tmp): ', len(ptf_label_tmp))
#prints the relevant portfolio name
        print('ptf_label_tmp: ', ptf_label_tmp)
        #FQ(77)
#takes the date on the portfolio label
        xx = ptf_label_tmp.split('_')[0]
        ln = len(xx)
#splits the date year, month, day
        yy = ptf_label_tmp[ln-8:ln-4]
        mm = ptf_label_tmp[ln-4:ln-2]
        dd = ptf_label_tmp[ln-2:ln]


        date_tmp = dd + '/' + mm + '/' + yy

#selects the risk free curve data, swaption volatility data and swaption shift data on the specified date
        rf_curve_dict = select_rf_curve(date_tmp, rf_mkt_curves)
        swp_vols = select_swp_mkt_data(date_tmp, swp_mkt_data)
        swp_shift = select_swp_mkt_data(date_tmp, shift_mkt_data_1)

        date_str_tmp = to_date_str(date_tmp)

#picks the swaption data to calibrate by filtering using the label 
        swp_data_to_calib = select_swp_ptf(ptf_label_tmp, ptf_swp_dataset)

#extracting relevant shift and volatilities
        shift_data        = select_data_to_calib(swp_shift, swp_data_to_calib, flag_exp, w_limit_min,w_limit_max )
        swp_data          = select_data_to_calib(swp_vols, swp_data_to_calib, flag_exp, w_limit_min, w_limit_max)

#creating a dictionary and dataframe of market price data 
        mkt_price_dict    = set_mkt_price_dict(swp_data, shift_data, rf_curve_dict)
        mkt_price_df      = set_mkt_price_df(swp_data, shift_data, rf_curve_dict)


        for mdl_label_tmp in mdl_list:

#shifting forward rates to avoid negatives
            shift_mean = np.asarray(shift_data['VALUES'].to_list()).mean()
            shift_ref = shift_mean/100.0
            
#printing evaluation number, model name, portfolio name and date
            print('n. eval: ', k, ', mdl: ', mdl_label_tmp, ', ptf: ', ptf_label_tmp, ', date: ', date_tmp)
#loading market data and calibrating our model parameters
            prm_data_dict, df_mdl_prms = select_model_prms_data(mdl_label_tmp, mdl_prms_data)
            mdl_prm_opt = model_calibration(mkt_price_dict, rf_curve_dict, mdl_label_tmp, prm_data_dict, shift_ref)


            t1 = time.time() #ends the timer and prints the time taken
            print('%.2f s.'%(t1-t0))

#computes model price using the calibrated parameters and computes the chi squared value
            df_model_data, chi2_tmp = set_mdl_calib_results(mkt_price_df, rf_curve_dict, mdl_label_tmp, shift_ref, mdl_prm_opt)
#optional plotting           
            report_f.make_plot(df_model_data, ptf_label_tmp, mdl_label_tmp, date_str_tmp, flag_exp, flag_plot, flag_save)
            report_f.dump_report(df_model_data, mdl_prm_opt, df_mdl_prms, ptf_label_tmp, mdl_label_tmp, date_str_tmp, shift_ref)

#saving data to output
            date_for_report.append(date_tmp)
            ptf_for_report.append(ptf_label_tmp)
            mdl_for_report.append(mdl_label_tmp)
            calib_type_for_report.append(calib_type)
            chi2_for_report.append(chi2_tmp)
            opt_prm_report.append(mdl_prm_opt)

            k = k + 1
            #print('mdl_prm_opt: ', mdl_prm_opt)

    report_f.set_final_report(opt_prm_report, shift_ref, date_for_report, ptf_for_report, mdl_for_report, calib_type_for_report, chi2_for_report, label_out)




