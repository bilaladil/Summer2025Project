# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import sys
import pandas as pd

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

if __name__ == '__main__':

    #label = 'Ap5Y'
    #label = 'Ap5Y_nofloor' #OK

    #label = 'Au5Y'
    #label = 'Au5Y_nofloor'

    #label = 'R20Y'
    #label = 'R20Y_nofloor'

    #label = 'Rp20Y'
    label = 'Rp20Y_nofloor' #OK

    #label = 'Ru15Y'
    #label = 'Ru15Y_nofloor'


    file_name1 = 'input/bande/valore_bande_20200731_%s.xlsx'%(label)
    #file_name2 = 'output/prepay_profile_15y_v0.xlsx'
    file_name2 = 'output/prepay_profile_30y_v0.xlsx'

    df_1 = pd.read_excel(open(file_name1, 'rb') )
    df_2 = pd.read_excel(open(file_name2, 'rb') )

    data_col = df_1['DATE INIZIO']
    time_list = []
    time_list.append(0)

    for i in range(1, len(data_col)):

        delta_date_ = df_1.iloc[i]['DATE INIZIO'] - df_1.iloc[0]['DATE INIZIO']
        time_ = delta_date_.days/365.2425

        time_list.append(time_)


    data_df_1_n = {'Time': time_list, 'Date': df_1['DATE INIZIO'], 'Lower': df_1['LOWER'], 'Upper': df_1['UPPER']}

    df_1_n = pd.DataFrame(data=data_df_1_n)



    col_list = df_2.columns
    print(df_2.head())

    fig = plt.figure()

    len_c = len(col_list)
    rate_list = []
    for i in range(0, len_c):
        cc_tmp = col_list[i]

        if (cc_tmp[:4] == 'Rate'):

            cc_s_tmp = cc_tmp.split()
            rate_list.append(cc_s_tmp[1])
            plt.plot(df_2['Time'], df_2[cc_tmp], '-.')
            print('cc_tmp: ', cc_tmp.split())

    plt.legend(rate_list)
    #plt.title('Notional mortgage profile, rate =%s'%(K))
    #plt.legend(['CPR = %.2f'%(0), 'CPR = %.2f'%(pp_1), 'CPR = %.2f'%(pp_2), 'CPR = %.2f'%(pp_3),'CPR = %.2f'%(pp_4),'CPR = %.2f'%(pp_5)])
    #plt.xlabel('Years')
    #plt.ylabel('Notional')


    plt.plot(df_1_n['Time'], df_1_n['Upper'], '-')
    plt.plot(df_1_n['Time'], df_1_n['Lower'], '-')


    plt.show()

