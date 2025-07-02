
import matplotlib.pyplot as plt
import data_model as dm
import numpy as np
import pandas as pd
import sys



def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def make_plot(df_mdl_data, ptf_label, mdl_label, date_label, flag_exp, flag_plot, flag_save):


    label_to_save = ptf_label + '_' + mdl_label + '_' + date_label + '_plot'


    label_y_to_plot = 'Price Swaption'
    l_list = []

    exp_list = list(set(df_mdl_data['EXPIRY'].tolist()))
    n_exp_list = len(exp_list)
    mat_list_n = []


    mat_list = list(set(df_mdl_data['MATURITY'].tolist()))
    n_mat_list = len(mat_list)
    exp_list_n = []

    term_dict_inv = {v: k for k, v in dm.term_dict.items()}


    if (flag_plot):
        # set plot
        fig = plt.figure()


        if (flag_exp == True):

            label_x_to_plot = 'Swap Maturity [year]'

            for i in range(0, n_exp_list):

                mat_list_n = []

                exp_tmp = exp_list[i]
                idx_exp = df_mdl_data['EXPIRY'] == exp_tmp

                mat_list_ = df_mdl_data[idx_exp]['MATURITY'].tolist()
                vol_mkt_mat = df_mdl_data[idx_exp]['MKT_DATA'].tolist()
                vol_mdl_mat = df_mdl_data[idx_exp]['MDL_DATA'].tolist()

                for mm in mat_list_:
                    mm_ = dm.term_dict[mm]
                    mat_list_n.append(mm_)

                vol_mkt_mat_s = [y for _, y in sorted(zip(mat_list_n, vol_mkt_mat), key=lambda x: x[0])]
                vol_mdl_mat_s = [y for _, y in sorted(zip(mat_list_n, vol_mdl_mat), key=lambda x: x[0])]

                sorted_list2 = sorted(mat_list_n)


                sorted_list2_ = np.array(sorted_list2)
                sorted_list2 = np.around(sorted_list2_, 1)
                sorted_list2 = list(sorted_list2)

                mat_list_n2 = []
                for mm2 in sorted_list2:
                    mm_ = term_dict_inv[mm2]
                    mat_list_n2.append(mm_)

                plt.plot(sorted_list2, vol_mkt_mat_s, 'o')
                plt.plot(sorted_list2, vol_mdl_mat_s, '--')

                l_tmp = exp_tmp
                l_list.append(l_tmp + 'Exp' +' '+'Mkt')
                l_list.append(l_tmp + 'Exp' +' '+'Mdl')

        else:

            label_x_to_plot = 'Expiry [year]'

            for i in range(0, n_mat_list):

                exp_list_n = []

                mat_tmp = mat_list[i]
                idx_mat = df_mdl_data['MATURITY'] == mat_tmp

                exp_list_ = df_mdl_data[idx_mat]['EXPIRY'].tolist()
                vol_mkt_exp = df_mdl_data[idx_mat]['MKT_DATA'].tolist()
                vol_mdl_exp = df_mdl_data[idx_mat]['MDL_DATA'].tolist()

                for ee in exp_list_:
                    ee_ = dm.term_dict[ee]
                    exp_list_n.append(ee_)

                vol_mkt_exp_s = [y for _, y in sorted(zip(exp_list_n, vol_mkt_exp), key=lambda x: x[0])]
                vol_mdl_exp_s = [y for _, y in sorted(zip(exp_list_n, vol_mdl_exp), key=lambda x: x[0])]


                sorted_list2 = sorted(exp_list_n)

                sorted_list2_ = np.array(sorted_list2)
                sorted_list2 = np.around(sorted_list2_, 1)
                sorted_list2 = list(sorted_list2)

                exp_list_n2 = []
                for ee2 in sorted_list2:
                    ee_ = term_dict_inv[ee2]
                    exp_list_n2.append(ee_)

                plt.plot(sorted_list2, vol_mkt_exp_s, 'o')
                plt.plot(sorted_list2, vol_mdl_exp_s, '--')

                l_tmp = mat_tmp
                l_list.append(l_tmp + 'Mat.' + ' ' + 'Mkt')
                l_list.append(l_tmp + 'Mat.' + ' ' + 'Mdl')

        plt.xlabel(label_x_to_plot)
        plt.ylabel(label_y_to_plot)
        plt.legend(l_list)
        #plt.show()
        #FQ(22)
        if (flag_save):
            fig.savefig('output/%s.png'%(label_to_save), dpi=fig.dpi)

        plt.close('all')


def dump_report(df_calib_data, dict_opt_prms, df_prms, ptf_label, mdl_out, date_label, shift_ref):

    file_out = 'make_calibration2/output/' + mdl_out + ptf_label + '_' + date_label + '.txt'
    x2_out = computeCHI2(df_calib_data)

    fout = open(file_out, "w")

    fout.write('Model: \t ')
    fout.write(str(mdl_out))
    fout.write("\n")
    fout.write('Ptf: \t ')
    fout.write(str(ptf_label))
    fout.write("\n")
    fout.write('Date: \t ')
    fout.write(str(date_label))
    fout.write('\n ')
    fout.write("\n")


    fout.write('Expiry\t Maturity\t mkt_data\t   model_data\t \n ')

    for i in range(0, len(df_calib_data)):

        exp_data = df_calib_data.iloc[i]['EXPIRY']
        mat_data = df_calib_data.iloc[i]['MATURITY']
        mkt_data = df_calib_data.iloc[i]['MKT_DATA']
        mdl_data = df_calib_data.iloc[i]['MDL_DATA']


        fout.write(exp_data)
        fout.write("\t")
        fout.write(mat_data)
        fout.write("\t")
        fout.write(str(mkt_data))
        fout.write("\t")
        fout.write(str(mdl_data))
        #fout.write("\t")
        fout.write("\n")

    fout.write("---------------------\n")
    fout.write("\n")
    fout.write("Opt params:\t")
    fout.write("\n\n")

    for kk in dict_opt_prms.keys():

        val = dict_opt_prms[kk]
        fout.write(kk)
        fout.write(": \t")
        fout.write(str(val))
        fout.write("\n")

    fout.write("\n")
    fout.write("Shift ref: ")
    fout.write(str(shift_ref))
    fout.write("\n")


    fout.write("------------------------\n")
    fout.write("\n")
    fout.write("\n")

    fout.write('MDL_NAME\t PRM_NAME\t X0\t MIN\t MAX\n')

    for i in range(0, len(df_prms)):

        mdl_name = df_prms.iloc[i]['MDL_NAME']
        prm_name = df_prms.iloc[i]['PRM_NAME']
        x0_data = df_prms.iloc[i]['X0']
        min_data = df_prms.iloc[i]['MIN']
        max_data = df_prms.iloc[i]['MAX']

        fout.write(mdl_name)
        fout.write("\t")
        fout.write(prm_name)
        fout.write("\t")
        fout.write(str(x0_data))
        fout.write("\t")
        fout.write(str(min_data))
        fout.write("\t")
        fout.write(str(max_data))
        fout.write("\n")


    fout.write("Livello chi2:")
    fout.write(str(x2_out))
    fout.write("\n")
    fout.write("------------------------\n")


def set_final_report(opt_prm_report, shift_ref, date_for_report, ptf_for_report, mdl_for_report, calib_type_for_report, chi2_for_report, label_out):

    prm_name_list_n = []
    prm_value_list_n = []
    date_for_report_n = []
    ptf_for_report_n = []
    mdl_for_report_n = []
    chi2_for_report_n = []
    calib_type_for_report_n = []

    for i in range(0, len(opt_prm_report)):

        opt_prm = opt_prm_report[i]
        prm_name_list = opt_prm.keys()

        for prm_name in prm_name_list:

            prm_name_list_n.append(prm_name)
            prm_value_list_n.append(opt_prm[prm_name])
            date_for_report_n.append(date_for_report[i])
            ptf_for_report_n.append(ptf_for_report[i])
            mdl_for_report_n.append(mdl_for_report[i])
            chi2_for_report_n.append(chi2_for_report[i])
            calib_type_for_report_n.append(calib_type_for_report[i])

        if (mdl_for_report[i] == 'LMM'):
            prm_name_list_n.append('shift ref.')
            prm_value_list_n.append(shift_ref)
            date_for_report_n.append(date_for_report[i])
            ptf_for_report_n.append(ptf_for_report[i])
            mdl_for_report_n.append(mdl_for_report[i])
            chi2_for_report_n.append(chi2_for_report[i])
            calib_type_for_report_n.append(calib_type_for_report[i])



    d1 = {'Date': date_for_report, 'Ptf_label': ptf_for_report, 'Mdl': mdl_for_report, 'Calib type': calib_type_for_report,
         'Mdl prm':  mdl_for_report, 'Chi2':  chi2_for_report}

    d2 = {'Date': date_for_report_n, 'Ptf_label': ptf_for_report_n, 'Mdl': mdl_for_report_n, 'Calib type': calib_type_for_report_n,
        'Chi2':  chi2_for_report_n, 'Prm name': prm_name_list_n, 'Prm value': prm_value_list_n}


    df_out_1 = pd.DataFrame(data=d1)
    df_out_2 = pd.DataFrame(data=d2)

    df_out_1.to_excel("output\output_summary_%s.xlsx"%(label_out), sheet_name='summary')
    df_out_2.to_excel("output\output_opt_params_%s.xlsx"%(label_out), sheet_name='opt_prms')



def computeCHI2(df_calib):


    x2TmpSum = 0.0

    i = 1

    for i in range(0, len(df_calib)):

        mktTmp = df_calib.iloc[i]['MKT_DATA']
        mdlTmp = df_calib.iloc[i]['MDL_DATA']


        x2Tmp = abs(float(mktTmp) - float(mdlTmp))
        x2Tmp = x2Tmp / float(mdlTmp)

        x2TmpSum = x2TmpSum + x2Tmp
    x2TmpSum = float(x2TmpSum / float(i))
    return x2TmpSum