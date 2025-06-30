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



##### Only works for semi annual payments because time interval is hardcoded as 6 months and 
##### if changing to work for any interval then need to change K calculation depending on number of payments


#pay_yy = payments per year
#Term in years
#N_0 initial Notional amount
#K = interest rate 
#####Is K per period or annual?
#pp = prepayment rate
#ref_date = start date of mortgage
#prepay_flag = prepayments allowed or not

#c = annuity payment


def mortgage_with_pp(pay_yy, T, N_0, K, pp, ref_date, prepay_flag):

    n_pay = T*pay_yy #total number of payments
    G = (1.0 + K) ** (-n_pay)  #discount factor
    c = (K * N_0) / (1.0 - G) #present value of annuity formula to get repayment amount

    n_list = [] #list of notional values at each step
    time_list = [] #list of time in years at each step 
    date_list = [] #list of date at each step 

    N_o = N_0 #notional outstanding at start of each period

#storing initial values in each list
    n_list.append(N_o)
    time_list.append(0.0)
    date_list.append(ref_date)

#setting current date
    date_old = ref_date


    for i in range(1, n_pay+1):

        qi_o = K*N_o #interest component of repayment
        qc_o = c  - qi_o #principal component of repayment 

#if prepayments are allowed (and not last payment) then the new notional amount at n + 1 is the 
#(amount outstanding at n minus the principal repayment at n) minus the prepayment as a percentage of
# (amount outstanding at n minus the principal repayment at n )
#if no prepayment ot last repayment then new notional is just amount outstanding at n minus the 
#principal repayment at n
        if (prepay_flag) and (i < n_pay):
            N_n= (N_o - qc_o) - pp*(N_o - qc_o)
            
        else:
            N_n= (N_o - qc_o)


        if (prepay_flag) and (i < n_pay):
#calculate sthe new discount factor and repayment if prepayment allowed and not last payment
            G = (1.0 + K) ** (-(n_pay-i))
            c = (K * N_n) / (1.0 - G)


        delta_time = dateutil.relativedelta.relativedelta(months=6) #time between payments 

        date_new = date_old + delta_time

#stores the current values in each of the lists
        n_list.append(N_n)
        time_list.append(i*0.5)
        date_list.append(date_new)

#set the date and notional equal the new values
        date_old = date_new
        N_o = N_n


    return  n_list, time_list, date_list


def from_date_to_time_df(df_1):

#taking the dates from only the "DATE INIZIO" column
    data_col = df_1['DATE INIZIO']
    time_list = []
    time_list.append(0)

    for i in range(1, len(data_col)):

#taking the date at time i and subtracting the start date from it and then calculating the number of days
#between them before turning it to years and appending the list
        delta_date_ = df_1.iloc[i]['DATE INIZIO'] - df_1.iloc[0]['DATE INIZIO']
        time_ = delta_date_.days/365.2425

        time_list.append(time_)

#creates a new dataframe storing the time elapsed, the date, the lower notional and the upper notional
    data_df_1_n = {'Time': time_list, 'Date': df_1['DATE INIZIO'], 'Lower': df_1['LOWER'], 'Upper': df_1['UPPER']}
    df_1_n = pd.DataFrame(data=data_df_1_n)

    return df_1_n


dict_legend = {'0.01': 'Prepay = 1%', '0.02': 'Prepay = 2%', '0.03': 'Prepay = 3%', '0.05': 'Prepay = 5%', '0.07': 'Prepay = 7%', '0.1': 'Prepay = 10%'}


def compute_bands(ref_year, ref_month, ref_day, flag_save, T, K, N0, level_rate, pay_yy):

#creates a date using the input date and stores it as a string with and without / in between
    date_ref = datetime.date(ref_year, ref_month, ref_day)
    date_str = date_ref.strftime('%Y%m%d')
    date_str_to_plot = date_ref.strftime('%Y/%m/%d')

#creating a label for files
    label_to_save = str(T) + 'Y' + level_rate + date_str

#setting the prepayment rates
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
#storing lists of notional values, time in years and dates when no prepay
    m_list, t_m_list, d_m_list = mortgage_with_pp(pay_yy, T, N0, K, pp, date_ref, False)

#storing lists of notional values, time in years and dates for each prepayment rate with 
#prepayments allowed
    m_list_pp_1, t_m_list_pp_1, d_m_list_pp_1 = mortgage_with_pp(pay_yy, T, N0, K, pp_1, date_ref, True)
    m_list_pp_2, t_m_list_pp_2, d_m_list_pp_2 = mortgage_with_pp(pay_yy, T, N0, K, pp_2, date_ref, True)
    m_list_pp_3, t_m_list_pp_3, d_m_list_pp_3 = mortgage_with_pp(pay_yy, T, N0, K, pp_3, date_ref, True)
    m_list_pp_4, t_m_list_pp_4, d_m_list_pp_4 = mortgage_with_pp(pay_yy, T, N0, K, pp_4, date_ref, True)
    m_list_pp_5, t_m_list_pp_5, d_m_list_pp_5 = mortgage_with_pp(pay_yy, T, N0, K, pp_5, date_ref, True)
    m_list_pp_6, t_m_list_pp_6, d_m_list_pp_6 = mortgage_with_pp(pay_yy, T, N0, K, pp_6, date_ref, True)

    #print('m_list: ',m_list[len(m_list) -1])

#creating a data frame to store dates, times and various rates ( for plotting? )
    data_prepay = {'Date': d_m_list, 'Time': t_m_list, 'Rate %s'%(pp_1): m_list_pp_1, 'Rate %s'%(pp_2): m_list_pp_2,
        'Rate %s'%(pp_3): m_list_pp_3, 'Rate %s'%(pp_4): m_list_pp_4, 'Rate %s'%(pp_5): m_list_pp_5, 'Rate %s'%(pp_6): m_list_pp_6}

#creating data frames for each band with the lower bound always 1%
    data_prepay_1 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_2}
    data_prepay_2 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_3}
    data_prepay_3 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_4}
    data_prepay_4 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_5}
    data_prepay_5 = {'DATE INIZIO': d_m_list, 'UPPER': m_list_pp_1, 'LOWER': m_list_pp_6}


#making each data frame into a pandas data frame
    df_prepay = pd.DataFrame(data=data_prepay)

    df_prepay_1 = pd.DataFrame(data=data_prepay_1)
    df_prepay_2 = pd.DataFrame(data=data_prepay_2)
    df_prepay_3 = pd.DataFrame(data=data_prepay_3)
    df_prepay_4 = pd.DataFrame(data=data_prepay_4)
    df_prepay_5 = pd.DataFrame(data=data_prepay_5)

    #fig = plt.figure()

#if saving allowed then it makes excel files to save the information where the name is made
#useing the label from before and the prepayment bands
    if (flag_save):
        df_prepay_1.to_excel("/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_bands/output/\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_2))
        df_prepay_2.to_excel("/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_bands/output/\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_3))
        df_prepay_3.to_excel("/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_bands/output/\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_4))
        df_prepay_4.to_excel("/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_bands/output/\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_5))
        df_prepay_5.to_excel("/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_bands/output/\%s_pp%s_%s.xlsx" %(label_to_save, pp_1, pp_6))

    fig = plt.figure()

    #plt.plot(t_m_list, m_list, '-.')


#creates a list of the columns in the data frame 
    col_list = df_prepay.columns

    len_c = len(col_list)
    rate_list = []
    for i in range(0, len_c):
        cc_tmp = col_list[i]

#if the column name starts with "Rate" then it splits the "Rate" and "0.0X" value
        if (cc_tmp[:4] == 'Rate'):

            cc_s_tmp = cc_tmp.split()
#then uses the dict_legend from above for each prepayment rate 
            cc_ = dict_legend[cc_s_tmp[1]]

#appends the list of rates with the prepayment rate
            rate_list.append(cc_)
#plots the time againt the prepayment
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
        fig.savefig('/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_bands/output/%s.png' % (label_to_save), dpi=fig.dpi)


if __name__ == '__main__':

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    #N (Ti+1) = N (Ti) − Q(Ti) − Λ (N (Ti) − Q(Ti)) .

    #six_months_later = datetime.date(future_year, future_month, future_day)


    #ptf_to_process =   pd.read_excel(open('input/ptf_to_process_for_bands_v1.xlsx', 'rb'), sheet_name='PTF_TO_PROCESS')

#reads excel file with data that needs to be processed
    ptf_to_process =   pd.read_excel(open('/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_bands/input/ptf_to_process_for_bands_v2.xlsx', 'rb'), sheet_name='PTF_TO_PROCESS')

#filters only rows that need to be done 
    idx = ptf_to_process['TO_DO'] == True


    ptf_to_process_ = ptf_to_process[idx]
    ptf_to_process_ = ptf_to_process_.reset_index()

    print(ptf_to_process_.head())

    for i in range(0, len(ptf_to_process_)):
#looping through every row in the filtered set

#prints the row and extracts maturity, interest and level rate
        print('record: ', ptf_to_process_.iloc[i])
        T = ptf_to_process_.iloc[i]['MATURITY']
        K = ptf_to_process_.iloc[i]['K']
        level_rate = ptf_to_process_.iloc[i]['LEVEL_RATE']

#takes the date from the file and stores the year, month, day separately
        date_ = str(ptf_to_process_.iloc[i]['STRING_DATE'])

        ref_year = int(date_[:4])
        ref_month = int(date_[4:6])
        ref_day = int(date_[6:])


        #FQ(666)
        
#setting the initial notional equal = 100 and 2 payments per year
        N0      = 100
        pay_yy  = 2
        #level_rate = 'L'
        flag_save = False
        #flag_save = True

        print('ref_year: ', ref_year)
        compute_bands(ref_year, ref_month, ref_day, flag_save, T, K, N0, level_rate, pay_yy)


