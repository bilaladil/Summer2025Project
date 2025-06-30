import sys
import pandas as pd
import numpy as np


def FQ(label):
	print ('------------- FIN QUI TUTTO OK  %s ----------' % (label))
	sys.exit()


import funzioni_base as fb




def Jamshidian(date, upperB, lowerB):
	n = len(date)

	a = []  # ora costruisco a (i termini noti)
	for a_r in range((n - 1)):
		a_row = []
		for a_c in range((n - 1)):
			a_row.append(0.0)
		a.append(a_row)

	B = []  # ora costruisco B (i nozionali); l'indice di riga e' la prima data di esercizio, quello di colonna e' la maturity date dello swap sottostante (cioe' l'ultimo pagamento)
	for b_r in range((n - 1)):
		b_row = []
		for b_c in range(n):
			b_row.append(0.0)
		B.append(b_row)

	# Nota: non si capisce perche' lavoro con due indici (a_row e b_row) e con due liste (a_row e b_row). Il punto e' che se creo le due matrici insieme
	# (cioe' usando un unico for) poi non funziona niente! Succede che a e B, praticamente, sono la stessa cosa (e si sovrascrivono a vicenda durante le assegnazioni).
	# quindi ho rimediato facendo due for (con indici e liste diverse, per sicurezza).

	for c in range((n - 1)):
		for r in range((c + 1)):
			x = upperB[c] - lowerB[r]
			if x < 0:
				a[r][c] = 0.0
			else:
				a[r][c] = x

	# ora scrivo le formule del paper

	for r in range((n - 1)):
		if r == 0:
			B[r][(n - 1)] = a[r][(n - 2)]
		else:
			B[r][(n - 1)] = a[r][(n - 2)] - a[(r - 1)][(n - 2)]

	for c in range((n - 1)):
		for r in range(c):
			if r == 0:
				B[r][c] = a[r][(c - 1)] - a[r][c]
			else:
				B[r][c] = a[r][(c - 1)] - a[r][c] - a[(r - 1)][(c - 1)] + a[(r - 1)][c]

	# e' ora di scrivere le matrici su files
	# Nota: se prendo le righe, le trasformo in stringhe, e poi le unisco (" ".join), poi mi ritrovo con dei numeri
	# le cui cifre decimali sono separate da spazi (e con MatLab non posso poi fare il load)

	# file=open ("date.txt", "w")
	# for c in xrange (n):
	# file.write (date [c])
	# file.write ("\n")
	# file.close ()

	# file=open ("a_matrix.txt", "w")
	# for r in xrange ((n-1)):
	# for c in xrange ((n-1)):
	# file.write (str (a [r][c]))
	# file.write ("  ")
	# file.write ("\n")
	# file.close ()

	# file=open ("B_matrix.txt", "w")
	# for r in xrange ((n-1)):
	# for c in xrange (n):
	# file.write (str (B [r][c]))
	# file.write ("  ")
	# file.write ("\n")
	# file.close ()

	return B


def between_idx(l1, low, high):
	# funzione che inserisce in l2 tutti i numeri all'interno di i1 compresi in un intervallo delimitato
	# da low e high
	l2 = []
	l2_indx = []

	ln = len(l1)
	for i in range(0, ln):

		l2Tmp = l1[i]
		if (l2Tmp > low and l2Tmp <= high):
			l2.append(l2Tmp)
			l2_indx.append(i)

	return l2, l2_indx


def build_new_idx_column(list_ref, list_target, df_ref, name_new_col):
	df_ref[name_new_col] = 999.00

	for i in range(0, len(list_ref)):

		highTmp = list_ref[i]

		if (i == 0):
			lowTmp = -0.00001
			midTmp = lowTmp

		else:
			lowTmp = list_ref[i - 1]
			midTmp = (highTmp + lowTmp) / 2.0

		between_out = between_idx(list_target, lowTmp, highTmp)

		if (i == 0): lowTmp = 0.0

		indx_out = between_out[1]
		# val_out  = between_out[0]

		for idxTmp in indx_out:

			valTmp = list_target[idxTmp]

			if (valTmp >= midTmp):

				df_ref[name_new_col][idxTmp] = highTmp
			else:
				df_ref[name_new_col][idxTmp] = lowTmp

	return df_ref


def save_report(report, outfile, flag_tipo_file):
	"""
    Take a report and save it to a single Excel file
    """
	if (flag_tipo_file == 'xl'):

		sheet_name = 'Foglio3'
		writer = pd.ExcelWriter(outfile)
		report.to_excel(writer, sheet_name)
		writer.save()

	else:

		report.to_csv(outfile, sep=',')


def convert_schedule(mat_to_convert):

	mat_r = np.round(float(mat_to_convert), 1)


	if (mat_r == 0.5):
		mat_n = '6M'
	elif (mat_r == 1.5):
		mat_n = '18M'
	else:
		mat_n = str(int(mat_to_convert))+ 'Y'


	return mat_n

def cmputeMatrixToCalib(file_anag_ptf_swpt, file_ptf_swpt_calib, file_type):
	df_bande = pd.read_csv(file_anag_ptf_swpt, sep='\t')

	list_ref_mat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 60]
	list_ref_exp = [0.08333, 2.0 / 12.0, 0.25, 0.5, 9.0 / 12.0, 1.0, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 50, 60]

	list_target_mat = df_bande['Mat'] - df_bande['Expiry'] # la maturity della swaption e' la durata dello swap, mentre
	# Jamshidian la conta dall'inizio dello swap ammortizing principale
	list_target_exp = df_bande['Expiry']

	name_new_col_1 = 'mat_to_calib'
	name_new_col_2 = 'exp_to_calib'

	df_bande = build_new_idx_column(list_ref_mat, list_target_mat, df_bande, name_new_col_1)
	df_bande = build_new_idx_column(list_ref_exp, list_target_exp, df_bande, name_new_col_2)

	pivot_out = pd.pivot_table(df_bande, index=["exp_to_calib", "mat_to_calib"], values=["Nominal"], aggfunc=[np.sum])

	#print('pivot_out: ', pivot_out)

	pivot_out_ = pivot_out.reset_index()
	#print('pivot_out: ', pivot_out_.columns)
	#print('pivot_out: ', pivot_out_['exp_to_calib'])
	#print('pivot_out: ', pivot_out_['mat_to_calib'])
	#print('pivot_out: ', pivot_out_['sum'])
	#pivot_out_.drop(columns=[('sum', 'Nominal')])

	exp_list =  []
	mat_list =  []

	for i in range(0, len(pivot_out_)):
		exp_ = pivot_out_.iloc[i]['exp_to_calib'].values[0]
		mat_ = pivot_out_.iloc[i]['mat_to_calib'].values[0]

		exp_n = convert_schedule(exp_)
		mat_n = convert_schedule(mat_)

		exp_list.append(exp_n)
		mat_list.append(mat_n)

	df_data_dict = {'exp_to_calib': exp_list, 'mat_to_calib': mat_list}
	#df_ = pd.DataFrame(data = df_data_dict)

	df_to_calib = pd.DataFrame(df_data_dict)

	file_ = file_anag_ptf_swpt.split('.txt')[0]
	file_ = file_.split('/')[2]
	file_ = file_.split('swaptions_')[1]
	file_n =  file_.replace('.', '')

	df_to_calib['PTF_LABEL'] = file_n

	#save_report(pivot_out, file_ptf_swpt_calib, file_type)
	save_report(df_to_calib, file_ptf_swpt_calib, file_type)




if __name__ == "__main__":

	# --------------MARKET FOCUS ----------------------------------------------------





	ptf_to_process = pd.read_excel(open('input/ptf_to_process_jam_v1.xlsx', 'rb'), sheet_name='PTF_TO_PROCESS')

	idx = ptf_to_process['TO_DO'] == True

	ptf_to_process_ = ptf_to_process[idx]
	ptf_to_process_ = ptf_to_process_.reset_index()

	pp_list = ['pp0.01_0.02', 'pp0.01_0.03', 'pp0.01_0.05', 'pp0.01_0.07', 'pp0.01_0.1']
	tenor = 0.5

	for i in range(0, len(ptf_to_process_)):
		band_ = str(ptf_to_process_.iloc[i]['LABEL_TO_SAVE'])

		strikeIn = float(ptf_to_process_.iloc[i]['K'])

		for j in range(0, len(pp_list)):

			band_file = band_ + '_' + pp_list[j] + '.xlsx'
			file_bande_v0 = 'input/bande_per_net/' + band_file





			# -------------- output --------------------------------------------------------

			file_bande_v0_s   = file_bande_v0.split('/')
			file_bande_v0_s   = file_bande_v0_s[2].split('xlsx')[0]
			file_ptf_out 	  = 'ptf_swaptions_' + file_bande_v0_s + 'txt'
			file_ptf_to_calib = 'swaptions_to_calib_' + file_bande_v0_s + 'txt'

			anag_ptf_swpt = 'output/ptf_anag/' + file_ptf_out
			ptf_swpt_calib = 'output/data_to_calib/' + file_ptf_to_calib

			# ------------------------------------------------------------------------------

			ptf_swpt_calib_xl = ptf_swpt_calib + '.xlsx'
			ptf_swpt_calib_csv = ptf_swpt_calib + '.csv'

			df = pd.read_excel(file_bande_v0)

			date_inizio = df['DATE INIZIO']
			lower_bound = df['LOWER']
			upper_bound = df['UPPER']

			B = Jamshidian(date_inizio, upper_bound, lower_bound)
			B_dict = fb.convertBtoDict(B)

			print('Start elab')
			fb.dumpSwaptionPtfAnag(B_dict, anag_ptf_swpt, tenor, strikeIn)
			# funzione che associa ad ogni expiry la somma dei nominali suddivisi per tenor
			cmputeMatrixToCalib(anag_ptf_swpt, ptf_swpt_calib_xl, 'xl')
			print('End elab!!')
