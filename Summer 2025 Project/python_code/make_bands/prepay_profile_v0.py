# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np

import datetime
import dateutil


def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def mortgage_with_pp(pay_yy, T, N_0, K, pp, ref_date, prepay_flag):

    n_pay = T*pay_yy
    G = (1.0 + K) ** (-n_pay)
    c = (K * N_0) / (1.0 - G)

    n_list = []
    time_list = []
    date_list = []

    N_o = N_0

    n_list.append(N_o)
    time_list.append(0.0)
    date_list.append(ref_date)

    date_old = ref_date


    for i in range(1, n_pay+1):

        qi_o = K*N_o
        qc_o = c  - qi_o

        if (prepay_flag) and (i < n_pay):
            N_n= (N_o - qc_o) - pp*(N_o - qc_o)
        else:
            N_n= (N_o - qc_o)


        if (prepay_flag) and (i < n_pay):
            G = (1.0 + K) ** (-(n_pay-i))
            c = (K * N_n) / (1.0 - G)

        delta_time = dateutil.relativedelta.relativedelta(months=6)

        date_new = date_old + delta_time
        n_list.append(N_n)
        time_list.append(i*0.5)
        date_list.append(date_new)

        date_old = date_new

        N_o = N_n


    return  n_list, time_list, date_list


def from_date_to_time_df(df_1):

    data_col = df_1['DATE INIZIO']
    time_list = []
    time_list.append(0)

    for i in range(1, len(data_col)):

        delta_date_ = df_1.iloc[i]['DATE INIZIO'] - df_1.iloc[0]['DATE INIZIO']
        time_ = delta_date_.days/365.2425

        time_list.append(time_)


    data_df_1_n = {'Time': time_list, 'Date': df_1['DATE INIZIO'], 'Lower': df_1['LOWER'], 'Upper': df_1['UPPER']}

    df_1_n = pd.DataFrame(data=data_df_1_n)

    return df_1_n


dict_legend = {'0.01': 'Prepay = 1%', '0.02': 'Prepay = 2%', '0.03': 'Prepay = 3%', '0.05': 'Prepay = 5%', '0.07': 'Prepay = 7%', '0.1': 'Prepay = 10%'}


def compute_bands(ref_year, ref_month, ref_day, flag_save, T, K, N0, level_rate, pay_yy):


    date_ref = datetime.date(ref_year, ref_month, ref_day)
    date_str = date_ref.strftime('%Y%m%d')
    date_str_to_plot = date_ref.strftime('%Y/%m/%d')

    label_to_save = str(T) + 'Y' + level_rate + date_str

    pp_1 = 0.01
    pp_2 = 0.02
    pp_3 = 0.03
    pp_4 = 0.05
    pp_5 = 0.07
    pp_6 = 0.10

    #label = 'Rp20Y_nofloor' #OK
    #file_name1 = 'input/bande/valore_bande_20200731_%s.xlsx'%(label)
    #df_1 = pd.read_excel(open(file_name1, 'rb') )
    #df_1_n = from_date_to_time_df(df_1)
    pp = 0.01
    m_list, t_m_list, d_m_list = mortgage_with_pp(pay_yy, T, N0, K, pp, date_ref, False)

    m_list_pp_1, t_m_list_pp_1, d_m_list_pp_1 = mortgage_with_pp(pay_yy, T, N0, K, pp_1, date_ref, True)
    m_list_pp_2, t_m_list_pp_2, d_m_list_pp_2 = mortgage_with_pp(pay_yy, T, N0, K, pp_2, date_ref, True)
    m_list_pp_3, t_m_list_pp_3, d_m_list_pp_3 = mortgage_with_pp(pay_yy, T, N0, K, pp_3, date_ref, True)
    m_list_pp_4, t_m_list_pp_4, d_m_list_pp_4 = mortgage_with_pp(pay_yy, T, N0, K, pp_4, date_ref, True)
    m_list_pp_5, t_m_list_pp_5, d_m_list_pp_5 = mortgage_with_pp(pay_yy, T, N0, K, pp_5, date_ref, True)
    m_list_pp_6, t_m_list_pp_6, d_m_list_pp_6 = mortgage_with_pp(pay_yy, T, N0, K, pp_6, date_ref, True)

    #print('m_list: ',m_list[len(m_list) -1])

    data_prepay = {'Date': d_m_list, 'Time': t_m_list, 'Rate %s'%(pp_1): m_list_pp_1, 'Rate %s'%(pp_2): m_list_pp_2,
        'Rate %s'%(pp_3): m_list_pp_3, 'Rate %s'%(pp_4): m_list_pp_4, 'Rate %s'%(pp_5): m_list_pp_5, 'Rate %s'%(pp_6): m_list_pp_6}


    data_prepay_1 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_2}
    data_prepay_2 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_3}
    data_prepay_3 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_4}
    data_prepay_4 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_5}
    data_prepay_5 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_6}


    df_prepay = pd.DataFrame(data=data_prepay)

    df_prepay_1 = pd.DataFrame(data=data_prepay_1)
    df_prepay_2 = pd.DataFrame(data=data_prepay_2)
    df_prepay_3 = pd.DataFrame(data=data_prepay_3)
    df_prepay_4 = pd.DataFrame(data=data_prepay_4)
    df_prepay_5 = pd.DataFrame(data=data_prepay_5)

    #fig = plt.figure()

    if (flag_save):

        df_prepay_1.to_excel("output\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_2))
        df_prepay_2.to_excel("output\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_3))
        df_prepay_3.to_excel("output\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_4))
        df_prepay_4.to_excel("output\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_5))
        df_prepay_5.to_excel("output\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_6))

    fig = plt.figure()

    #plt.plot(t_m_list, m_list, '-.')


    col_list = df_prepay.columns

    len_c = len(col_list)
    rate_list = []
    for i in range(0, len_c):
        cc_tmp = col_list[i]

        if (cc_tmp[:4] == 'Rate'):

            cc_s_tmp = cc_tmp.split()
            cc_ = dict_legend[cc_s_tmp[1]]

            rate_list.append(cc_)
            plt.plot(df_prepay['Time'], df_prepay[cc_tmp], '-.')

    plt.legend(rate_list)

    #time_ref = df_1_n['Time']
    #upper_   = df_1_n['Upper']
    #lower_   = df_1_n['Lower']

    #plt.plot(time_ref, upper_, 'r')
    #plt.plot(time_ref, lower_, 'r')

    #plt.plot(t_m_list_pp_2, m_list_pp_2, 'k-.')
    #plt.plot(t_m_list_pp_3, m_list_pp_3, 'm-.')
    #plt.plot(t_m_list_pp_4, m_list_pp_4, 'g-.')
    #plt.plot(t_m_list_pp_5, m_list_pp_5, 'y-.')

    #plt.title('Notional mortgage profile, rate =%s, Date=%s'%(K, date_str))
    plt.title('Notional mortgage profile, rate =%s, Date=%s'%(np.round(K,6), date_str_to_plot))

    #plt.legend(['CPR = %.2f'%(0), 'CPR = %.2f'%(pp_1), 'CPR = %.2f'%(pp_2), 'CPR = %.2f'%(pp_3),'CPR = %.2f'%(pp_4),'CPR = %.2f'%(pp_5)])
    plt.xlabel('Years')
    plt.ylabel('Notional')

    #plt.show()

    if (flag_save):
        fig.savefig('output/%s.png' % (label_to_save), dpi=fig.dpi)


if __name__ == '__main__':

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    #N (Ti+1) = N (Ti) − Q(Ti) − Λ (N (Ti) − Q(Ti)) .

    #six_months_later = datetime.date(future_year, future_month, future_day)


    #ptf_to_process =   pd.read_excel(open('input/ptf_to_process_for_bands_v1.xlsx', 'rb'), sheet_name='PTF_TO_PROCESS')
    ptf_to_process =   pd.read_excel(open('input/ptf_to_process_for_bands_v2.xlsx', 'rb'), sheet_name='PTF_TO_PROCESS')

    idx = ptf_to_process['TO_DO'] == True

    ptf_to_process_ = ptf_to_process[idx]
    ptf_to_process_ = ptf_to_process_.reset_index()

    print(ptf_to_process_.head())

    for i in range(0, len(ptf_to_process_)):


        print('record: ', ptf_to_process_.iloc[i])
        T = ptf_to_process_.iloc[i]['MATURITY']
        K = ptf_to_process_.iloc[i]['K']
        level_rate = ptf_to_process_.iloc[i]['LEVEL_RATE']

        date_ = str(ptf_to_process_.iloc[i]['STRING_DATE'])

        ref_year = int(date_[:4])
        ref_month = int(date_[4:6])
        ref_day = int(date_[6:])

        #FQ(666)
        N0      = 100
        pay_yy  = 2
        #level_rate = 'L'
        flag_save = False
        flag_save = True

        print('ref_year: ', ref_year)
        compute_bands(ref_year, ref_month, ref_day, flag_save, T, K, N0, level_rate, pay_yy)


