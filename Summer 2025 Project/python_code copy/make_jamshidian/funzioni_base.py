import numpy as np
from os import sys
#import security_data_calib_map as mp

import string
import socket
import time


from datetime import datetime
from dateutil.relativedelta import relativedelta



def FQ(label):
	print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
	sys.exit()


#this function takes a text file and splits it into lines before splitting it into individual values
#depending on where there are spaces. It then puts all the individual values into a new list 
def loadData(inputFile):

	fin = open(inputFile, 'r')
	listInput = fin.readlines()
	
	newDataList = []
	n_lines   = len(listInput)
	
	for i in range(1, n_lines):
		
		line_splitted = listInput[i].split()
		
		for k in line_splitted:
			newDataList.append(k)
			
	return newDataList



def statisticsWord(input_list):
	
	input_list_u = list(set(input_list))
	n_wd_u = len(input_list_u)
	n_wd   = len(input_list)
	
	dict_statistics = {}

	for i in range(0, n_wd_u):

		wd_ref = input_list_u[i]
		n_f = 0


		for j in range(0, n_wd):

			wd_tmp = input_list[j]

			if (wd_tmp == wd_ref):
			
				n_f = n_f + 1


		try:
			dict_statistics[n_f]

			list_ref = dict_statistics[n_f]
			list_ref.append(wd_ref)
			dict_statistics[n_f] = list_ref

		except:

			dict_statistics[n_f] = [wd_ref]


	return dict_statistics


def converToStrMat(mat_value):
	
	dict_mat = {}
	
	dict_mat['0.08333'] = '1M'
	dict_mat['0.166666666667'] = '2M'
	dict_mat['0.25'] = '3M'
	dict_mat['0.50'] = '6M'
	dict_mat['0.5'] = '6M'
	dict_mat['0.75'] = '9M'
	dict_mat['1.0'] = '1Y'
	dict_mat['1.5'] = '18M'
	dict_mat['2.0'] = '2Y'
	dict_mat['3.0'] = '3Y'
	dict_mat['4.0'] = '4Y'
	dict_mat['5.0'] = '5Y'
	dict_mat['6.0'] = '6Y'
	dict_mat['7.0'] = '7Y'
	dict_mat['8.0'] = '8Y'
	dict_mat['9.0'] = '9Y'
	dict_mat['10.0'] = '10Y'
	dict_mat['15.0'] = '15Y'
	dict_mat['20.0'] = '20Y'
	dict_mat['25.0'] = '25Y'
	dict_mat['30.0'] = '30Y'
	dict_mat['50.0'] = '50Y'

	mat_str = dict_mat[mat_value]

	return mat_str



def convertMat(matStr):
	
	dict_convert = {}


	dict_convert['1D'] = 1.0/365.2425
	dict_convert['7D'] = 7.0/365.2425

	dict_convert['1W'] = 1.0/52.0
	dict_convert['2W'] = 2.0/52.0
	dict_convert['3W'] = 3.0/52.0

	dict_convert['0M'] = 0.0

	dict_convert['1M'] = 1.0/12.0
	dict_convert['2M'] = 2.0/12.0
	dict_convert['3M'] = 3.0/12.0
	dict_convert['4M'] = 4.0/12.0
	dict_convert['5M'] = 5.0/12.0
	dict_convert['6M'] = 6.0/12.0
	dict_convert['7M'] = 7.0/12.0
	dict_convert['8M'] = 8.0/12.0
	dict_convert['9M'] = 9.0/12.0
	dict_convert['10M'] = 10.0/12.0
	dict_convert['11M'] = 11.0/12.0
	dict_convert['12M'] = 12.0/12.0
	dict_convert['1Y'] = 1.0
	dict_convert['2Y'] = 2.0
	dict_convert['3Y'] = 3.0
	dict_convert['4Y'] = 4.0
	dict_convert['5Y'] = 5.0
	dict_convert['6Y'] = 6.0
	dict_convert['7Y'] = 7.0
	dict_convert['8Y'] = 8.0
	dict_convert['9Y'] = 9.0
	dict_convert['10Y'] = 10.0
	dict_convert['11Y'] = 11.0
	dict_convert['12Y'] = 12.0
	dict_convert['13Y'] = 13.0
	dict_convert['14Y'] = 14.0
	dict_convert['15Y'] = 15.0
	dict_convert['20Y'] = 20.0
	dict_convert['25Y'] = 25.0
	dict_convert['30Y'] = 30.0
	dict_convert['35Y'] = 35.0
	dict_convert['40Y'] = 40.0
	dict_convert['50Y'] = 50.0
	dict_convert['60Y'] = 60.0
	
	valOut = dict_convert[matStr]
	
	return valOut
	
def dumpSwaptionPtfAnag(Bn, file_to_dump, tenor, strikeIn):

	file_to_dump=open (file_to_dump, "w")
	
	exp_list   = Bn.keys()
	mat_list   = Bn[0].keys()
	
	n_exp    = len(exp_list)
	n_mat    = len(mat_list)


	file_to_dump.write("Strike\tNominal\tExpiry\tMat\n")
	
	
	for i in range(0, n_exp):

		for j in range(0, n_mat):

			expTmp      = i*tenor
			matTmp      = j*tenor
			notionalTmp = float(Bn[i][j])
			strike      = strikeIn

			if (notionalTmp > 0.0001):
				file_to_dump.write("%s\t %.4f\t %s\t %s\n" %(strike, notionalTmp, expTmp, matTmp))
			else:

				continue



def convertBtoDict(B):
	
	nr = len(B)
	nc = len(B[0])
	
	B_dict = {}
	
	for i in range(0, nr):
		B_dict[i] = {}
		for j in range(0, nc):
			B_dict[i][j] = B[i][j]

	return B_dict



def dumpProb(dict_to_plot, file_out, label_x, label_y):

	fout = open(file_out, 'w')
	
	x_ref_list = dict_to_plot.keys()
	y_ref_list = dict_to_plot[x_ref_list[0]].keys()

	ln_x = len(x_ref_list)
	ln_y = len(y_ref_list)

	fout.write(('\t'))

	for i in range(ln_y):
		yTmp = y_ref_list[i]
		fout.write(('%s\t')%yTmp)

	fout.write(('\n'))


	for i in range(ln_x):
		
		xTmp = x_ref_list[i]

		fout.write('%s\t' %(xTmp))
		for j in range(0, ln_y):
			
			yTmp = y_ref_list[j]
			
			valToWrite = dict_to_plot[xTmp][yTmp]
			
			fout.write('%s\t' %(valToWrite))
		fout.write('\n')


def dumpResultsOnFile(dict_res, file_out, flag_call):

	fout = open(file_out, 'w')
	
	securities_list = dict_res.keys()
	ln_sec = len(securities_list)

	if (flag_call == True):
		fout.write(('Security\t CalibType\t ShockType\t PCall\t Price\t Error\n'))
	else:
		fout.write(('Security\t CalibType\t ShockType\t Price\t Error\n'))

	for i in range(0, ln_sec):
		
		secTmp = securities_list[i]
		calibList = dict_res[secTmp].keys()
		ln_calib = len(calibList)
		
		for j in range(0, ln_calib):
			
			calibTmp = calibList[j]
			shockListTmp = dict_res[secTmp][calibTmp].keys()
			ln_shock = len(shockListTmp)
			
			for k in range(0, ln_shock):

				shockTmp = shockListTmp[k]
				resListTmp = dict_res[secTmp][calibTmp][shockTmp]

				priceTmp 	= resListTmp[0]
				errTmp 		= resListTmp[1]
				pCall 		= resListTmp[2]

				fout.write(('%s\t')%secTmp)
				fout.write(('%s\t')%calibTmp)
				fout.write(('%s\t')%shockTmp)
				
				if (flag_call == True):
					fout.write(('%.3f\t')%pCall)
				
				fout.write(('%.3f\t')%priceTmp)
				fout.write(('%.3f\t')%errTmp)
				fout.write('\n')


	fout.close()


def dumpResultsOnFileBase(resList, file_out):

	fout = open(file_out, 'w')
	
	fout.write(('Price\t Error mc\n'))
	
	
	
	priceTmp 	= resList[0]
	errTmp 		= resList[1]

	fout.write(('%s\t')%priceTmp)
	fout.write(('%s\t')%errTmp)


	fout.close()


"""
def converToMat(mat_str):
	
	dict_mat_v = {}
	
	dict_mat_v['0M']  = 0.0
	dict_mat_v['1W']  = 1.0/52.0
	dict_mat_v['2W']  = 2.0/52.0
	dict_mat_v['3W']  = 3.0/52.0
	dict_mat_v['1M']  = 1.0/12.0
	dict_mat_v['2M']  = 2.0/12.0
	dict_mat_v['3M']  = 0.25
	dict_mat_v['4M']  = 4.0/12.0
	dict_mat_v['5M']  = 5.0/12.0
	dict_mat_v['6M']  = 0.5
	dict_mat_v['9M']  = 1.0/12.0
	dict_mat_v['1Y']  = 1.0
	dict_mat_v['2Y']  = 2.0
	dict_mat_v['18M'] = 1.5 
	dict_mat_v['3Y'] = 3.0
	dict_mat_v['4Y'] = 4.0
	dict_mat_v['5Y'] = 5.0
	dict_mat_v['6Y'] = 6.0
	dict_mat_v['7Y'] = 7.0
	dict_mat_v['8Y'] = 8.0
	dict_mat_v['9Y'] = 9.0
	dict_mat_v['10Y'] = 10.0
	dict_mat_v['12Y'] = 12.0
	dict_mat_v['15Y'] = 15.0
	dict_mat_v['20Y'] = 20.0
	dict_mat_v['25Y'] = 25.0
	dict_mat_v['30Y'] = 30.0
	dict_mat_v['35Y'] = 35.0
	dict_mat_v['40Y'] = 40.0
	dict_mat_v['50Y'] = 50.0
	
	
	
	
	mat_val = dict_mat_v[mat_str]

	
	return mat_val
"""

def loadCurveData(file_input, date_ref):

	fin  = open(file_input, 'r')
	listInput = fin.readlines()

	mat_list    = []
	df_list    = []

	for i in range(1, len(listInput)):

		rowTmp   = listInput[i].split(",")

		matTmp   = rowTmp[0]
		rateTmp  = float(rowTmp[1])
		
		expTmp   = convertMat(matTmp)
		
		if (expTmp < 1.0/12):
			
			days_to_add = int(expTmp*365.2425)
			dateTmp  = date_ref + relativedelta(days=days_to_add)

		else:
			n_mths   = int(12.0*expTmp)
			dateTmp  = date_ref + relativedelta(months=n_mths)
		
		dfTmp    = np.exp(-expTmp*rateTmp)
		#dateTmp  = date_ref + relativedelta(months=n_mths)
		dateTmp = dateTmp.strftime("%Y-%m-%d")
		
		mat_list.append(dateTmp)
		df_list.append(dfTmp)
		

	return mat_list, df_list


def load_curve_fromFile(inputCurveFile):

	fin = open(inputCurveFile, 'r')
	listInput = fin.readlines()
	
	dict_curve = {}
	
	zc_rates_list = []
	time_list = []
	df_list = []
	mat_list = []

	for i in range(1, len(listInput)):

		line_splittedTmp = listInput[i].split("\t")        

		matNameTmp    = str(line_splittedTmp[0])
		timeTmp		  = convertMat(matNameTmp)

		rfValueTmp   = float(line_splittedTmp[1])/100.0
		dfTmp 		 = np.exp(-rfValueTmp*timeTmp)
		

		mat_list.append(matNameTmp)
		time_list.append(timeTmp)
		zc_rates_list.append(rfValueTmp)
		df_list.append(dfTmp)
		

	dict_curve['Mat'] = mat_list
	dict_curve['Times'] = time_list
	dict_curve['ZC rates'] = zc_rates_list
	dict_curve['ZC prices'] = df_list

	return dict_curve


def load_curve_fromFileN(inputCurveFile):

	fin = open(inputCurveFile, 'r')
	listInput = fin.readlines()
	
	dict_curve = {}
	
	zc_rates_list = []
	time_list = []
	df_list = []
	mat_list = []

	for i in range(1, len(listInput)):

		line_splittedTmp = listInput[i].split(",")        

		matNameTmp    = str(line_splittedTmp[0])
		timeTmp		  = convertMat(matNameTmp)

		rfValueTmp   = float(line_splittedTmp[1])
		dfTmp 		 = np.exp(-rfValueTmp*timeTmp)
		

		mat_list.append(matNameTmp)
		time_list.append(timeTmp)
		zc_rates_list.append(rfValueTmp)
		df_list.append(dfTmp)
		

	dict_curve['Mat'] = mat_list
	dict_curve['Times'] = time_list
	dict_curve['ZC rates'] = zc_rates_list
	dict_curve['ZC prices'] = df_list

	return dict_curve


def load_calib_fromFile(inputCalibFile):

	fin = open(inputCalibFile, 'r')
	listInput = fin.readlines()
	
	convertPrmsName = {}

	convertPrmsName['r0'] = 'R0'
	convertPrmsName['kappa'] = 'Kappa'
	convertPrmsName['theta'] = 'Theta'
	convertPrmsName['sigma'] = 'Sigma'


	
	dict_out = {}

	for i in range(0, len(listInput)):

		line_splittedTmp = listInput[i].split("\t")        

		prmsNameTmp    = str(line_splittedTmp[0])
		
		prmsNameTmp = convertPrmsName[prmsNameTmp]
		prmsValueTmp   = float(line_splittedTmp[1])

		dict_out[prmsNameTmp] = prmsValueTmp

	return dict_out


def load_securityInfo(inputCfgFile):

	fin = open(inputCfgFile, 'r')
	listInput = fin.readlines()
	return listInput



		
def	updateCFG(calib_info, zc_info_list, trjTagVal, NrBlocksVal, f_sec_cfg_test, f_sec_cfg_test_n):
	
	list_prm = calib_info.keys()

	fin = open(f_sec_cfg_test)
	fout = open(f_sec_cfg_test_n, "wt")
	
	zcTagRef = 'TimesLevelsList'
	trjTag = 'NrTrj'
	NrBlocksTag = 'NrBlocks'
	
	#trjTag = 'NrTrj'

	for line in fin:
		
		for prmsToSub in list_prm:
			prmsVal = calib_info[prmsToSub]
		
			if prmsToSub in line:
				
				line_splitted = line.split()
				toReplace = line_splitted[2]
				line = line.replace(toReplace, str(prmsVal))

		if zcTagRef in line:
			
			line_splitted = line.split('=')
			toReplace = line_splitted[1]
			line = line.replace(toReplace, str(zc_info_list))

		
		
		if (NrBlocksVal != None):
			if trjTag in line:
				
				line_splitted = line.split('=')
				
				toReplace = (line_splitted[1]).strip()
				line = line.replace(toReplace, str(trjTagVal))
	
			
			if NrBlocksTag in line:
				
				line_splitted = line.split('=')
				
				toReplace = (line_splitted[1]).strip()
				line = line.replace(toReplace, str(NrBlocksVal))
		

		fout.write(line)

	fin.close()
	fout.close()
	
	return 100




def	buildZCurveList(rf_curve):

	ln = len(rf_curve['Times'])
	
	strTmp = '0,1.0'
	
	for i in range(0, ln):
		
		timeTmp  = rf_curve['Times'][i]
		priceTmp = rf_curve['ZC prices'][i]

		strTmp = strTmp + ' + '
		strTmp = strTmp + str('%.6f'%(timeTmp)) + ',' + str('%.6f'%(priceTmp))
		

	return strTmp

def setupShockingCalibration(calib_info, rf_curve, calibTypeTmp, shock_ref):


	calibTypeTmp_s = calibTypeTmp.split('_')

	shock_type  = calibTypeTmp_s[0]
	shock_mat = calibTypeTmp_s[1]
	
	if (shock_type == 'CS'):
		
		if (shock_mat == 'UP'):
			
			r0_shocked = calib_info['R0'] + shock_ref
			calib_info['R0'] = r0_shocked
			
		else:

			print ('Tipo shock: %s non contemplato!!!' %calibTypeTmp)
			FQ(998)			
		
	elif (shock_type == 'RFREE'):
		
		if (shock_mat == '10+'):
			
			mat_list = rf_curve['Times']

			for i in range(0, len(mat_list)):
	
				matTmp = mat_list[i]

				if (matTmp > 10):
					
					rateToShock = rf_curve['ZC rates'][i] 
					timeOfShock = rf_curve['Times'][i] 
	
					rateShocked = rateToShock + shock_ref
					priceShocked = np.exp(-rateShocked*timeOfShock)
					
					rf_curve['ZC rates'][i] =rateShocked
					rf_curve['ZC prices'][i] =priceShocked

		else:
		
			mat_list = rf_curve['Mat']
		
			for i in range(0, len(mat_list)):
				
				matTmp = mat_list[i]
				
				if (matTmp == shock_mat):
					
					rateToShock = rf_curve['ZC rates'][i] 
					timeOfShock = rf_curve['Times'][i] 
	
					rateShocked = rateToShock + shock_ref
					priceShocked = np.exp(-rateShocked*timeOfShock)
					
					rf_curve['ZC rates'][i] =rateShocked
					rf_curve['ZC prices'][i] =priceShocked
				
	
	
	else:
		
		print ('Tipo shock: %s non contemplato!!!' %calibTypeTmp)
		FQ(998)


	return	calib_info,  rf_curve

def computeExpiryDate(currentDate, expiry):

	n_expiry = int(expiry*12.0) 

	dateTmp  = currentDate + relativedelta(months=n_expiry)

	return	dateTmp


def	buildZCurveListDate(rf_curve, file_input):

	currentDate = retriveCurrentDate(file_input)

	ln = len(rf_curve['Times'])
	flag_start = 0
	
	strTmp = ''
	
	if ( float(rf_curve['Times'][0]) > 0.00001):

		dateTmp  = currentDate
		strTmp = strTmp  + dateTmp.strftime("%d/%m/%Y") + ' , ' + str(1.0000)
		flag_start = 1
		
	for i in range(0, ln):
		
		
		timeTmp  = rf_curve['Times'][i]
		dfTmp    = rf_curve['ZC prices'][i]

		n_mnthTmp = int(timeTmp*12.0)
		dateTmp  = currentDate + relativedelta(months=n_mnthTmp)
		

		if (flag_start == 0):
			#strTmp = "ListOfDates = "  + dateTmp.strftime("%d/%m/%Y")
			strTmp = strTmp  + dateTmp.strftime("%d/%m/%Y") + ' , ' + str(dfTmp)
			flag_start = 1

		else:
			strTmp = strTmp + " + "  + dateTmp.strftime("%d/%m/%Y") + ' , ' + str(dfTmp)


	return strTmp



def	buildTimetable(op_input, tenor, expiry, freq, file_input):

	currentDate = retriveCurrentDate(file_input)
	expiryDate  = computeExpiryDate(currentDate, expiry)
	
	tenor_m = tenor - freq
	tenor_p = tenor + freq

	if (op_input[0] == 'CURRENT_DAY'):
		
		start_date = currentDate
	
	elif (op_input[0] == 'EXPIRY_DAY'):

		start_date = expiryDate
	
	elif (op_input[0] == 'EXPIRY_DAY+'):

		mnth_to_add = int(freq*12.0)
		start_date = expiryDate + relativedelta(months=mnth_to_add)


	elif (op_input[0] == 'SINGLE'):

		if (op_input[1] == 'CURRENT_DAY'):
			
			start_date = currentDate
		
		elif (op_input[1] == 'EXPIRY_DAY'):
	
			start_date = expiryDate
			
		else:

			print ('CASO NON GESTITO!!!')

	else:

		print('CASO NON GESTITO')

	if (op_input[1] == 'EXPIRY'):
		
		mat = expiry
		
	elif (op_input[1] == 'TENOR'):
		
		mat = tenor

	elif (op_input[1] == 'TENOR_M'):
	
		mat = tenor_m

	elif (op_input[1] == 'EXP+TENOR_P'):
	
		mat = expiry + tenor_p

	elif (op_input[0] == 'SINGLE'):
	
		mat = 0


	else:
		print ('CASO NON GESTITO')


	if (op_input[0] == 'EXPIRY_DAY+'):
		yy_to_sub = freq
		mat = mat - yy_to_sub


	strTmp = buildTimetableBase(start_date, mat, freq)

	return strTmp





def	buildTimetableBase(start_date, mat, freq):

	unit_ref = int(freq*12.0)
	ln = int(mat/freq)
	
	strTmp = ''
	
	for i in range(0, ln + 1):
		
		dateTmp  = start_date + relativedelta(months=i*unit_ref)

		if (i == 0):
			#strTmp = "ListOfDates = "  + dateTmp.strftime("%d/%m/%Y")
			strTmp = strTmp  + dateTmp.strftime("%d/%m/%Y")

		else:
			strTmp = strTmp + " + "  + dateTmp.strftime("%d/%m/%Y")
		

	return strTmp


def retriveCurrentDate(file_input):
	
	import datetime

	fin = open(file_input)

	
	stringToCatch = 'CurrentDate'

	flag_catched = 0

	for line in fin:
		
		if stringToCatch in line and (flag_catched == 0):
			
			line_splitted = line.split()
			flag_catched = 1
			data_tmp = line_splitted[2]
			data_tmp = data_tmp.split('/')


	#print int(data_tmp[2])
	#print int(data_tmp[1])	
	#print int(data_tmp[0])	
	
	data_out_d = datetime.date(int(data_tmp[2]), int(data_tmp[1]), int(data_tmp[0]))
	
	fin.close()

	return data_out_d



def	updateCFGNominalAndStrike(nominalValue, strikeValue, cfg_base, cfg_new):

	fin = open(cfg_base)
	fout = open(cfg_new, "wt")
	
	flag_strike_catched = 0
	flag_nominal_catched = 0
	
	lines   = tuple(open(cfg_base, 'r'))
	n_lines = len(lines)

	k = 0
	
	while (k < n_lines):

		start_indx = k		
		line = lines[k]

		if 'Strike' in line and (flag_strike_catched == 0):

			flag_strike_catched = 1

			target_line = line
			toReplace 	= target_line.split("=")[1]			
			line		= line.replace(toReplace, str(strikeValue) + '\n')



		if 'Nominal' in line and (flag_nominal_catched == 0):

			flag_nominal_catched = 1

			target_line = line
			toReplace 	= target_line.split("=")[1]
			line		= line.replace(toReplace, str(nominalValue) + '\n')

		if (k == start_indx):
			fout.write(line)
			k = k + 1
		else:
			continue
			#k = k + 1

	fin.close()
	fout.close()
	
	return 100


def	updateCFGbyTimetable(timetable, stringToCatch, cfg_base, cfg_new):

	fin = open(cfg_base)
	fout = open(cfg_new, "wt")
	
	flag_catched = 0
	
	lines   = tuple(open(cfg_base, 'r'))
	n_lines = len(lines)

	k = 0
	
	while (k < n_lines):

		start_indx = k		
		line = lines[k]

		if stringToCatch in line and (flag_catched == 0):

			flag_catched = 1

			fout.write(line)

			line_1 = lines[k+1]
			line_2 = lines[k+2]

			fout.write(line_1)
			fout.write(line_2)

			target_line = lines[k+3]
			toReplace 	= target_line.split("=")[1]			
			tLine_n		= lines[k+3].replace(toReplace, timetable)

			fout.write(tLine_n)
			fout.write('\n')

			k = k + 4

		if (k == start_indx):
			fout.write(line)
			k = k + 1
		else:
			continue
			#k = k + 1

	fin.close()
	fout.close()
	
	return 100


def	updateCFGprms(valuePrms, stringToCatch, cfg_base, cfg_new):

	fin  = open(cfg_base)
	fout = open(cfg_new, "wt")
	flag_catched = 0
	
	lines   = tuple(open(cfg_base, 'r'))
	n_lines = len(lines)

	k = 0
	
	
	while (k < n_lines):

		start_indx = k		
		line = lines[k]

		if stringToCatch in line and (flag_catched == 0):

			flag_catched = 1

			target_line = line
			toReplace 	= target_line.split("=")[1]			
			
			
			tLine_n		= line.replace(toReplace, valuePrms)

			fout.write(tLine_n)
			fout.write('\n')

			k = k + 1

		if (k == start_indx):
			fout.write(line)
			k = k + 1
		else:
			continue
			#k = k + 1

	fin.close()
	fout.close()
	
	return 100



def	updateCFGbyTenorTimetable(timetable, stringToCatch, cfg_base, cfg_new):

	fin = open(cfg_base)
	fout = open(cfg_new, "wt")
	
	flag_catched = 0
	
	lines   = tuple(open(cfg_base, 'r'))
	n_lines = len(lines)

	k = 0
	
	while (k < n_lines):

		start_indx = k		
		line = lines[k]

		if stringToCatch in line and (flag_catched == 0):

			flag_catched = 1

			fout.write(line)

			line_1 = lines[k+1]
			line_2 = lines[k+2]
			line_3 = lines[k+3]

			fout.write(line_1)
			fout.write(line_2)
			fout.write(line_3)

			target_line = lines[k+4]

			toReplace 	= target_line.split("=")[1]			
			tLine_n		= lines[k+4].replace(toReplace, timetable)

			fout.write(tLine_n)
			fout.write('\n')

			k = k + 5

		if (k == start_indx):
			fout.write(line)
			k = k + 1
		else:
			continue
			#k = k + 1

	fin.close()
	fout.close()
	
	return 100


def	updateDataCFG(lmm_prms, tenor, expiry, freq, strike, nominal, zc_curve_data, tipo_swpt, cfg_base, cfg_new):
	
	cfg_tmp = cfg_base[:-4] + '_tmp.cfg'
	
	op_input = {}
	op_input['SCAD_TRJ']  = ['CURRENT_DAY', 'EXPIRY']
	op_input['SCAD_FIX']  = ['EXPIRY_DAY', 'TENOR_M']
	op_input['SCAD_PAY']  = ['EXPIRY_DAY+', 'TENOR']
	op_input['SCAD_TEN']  = ['CURRENT_DAY', 'EXP+TENOR_P']
	
	if (tipo_swpt == 'EUROPEA'):
		
		op_input['SCAD_CALL'] = ['SINGLE', 'EXPIRY_DAY'] # opzione europea
	
	elif (tipo_swpt == 'BERMUDA'):
		
		op_input['SCAD_CALL'] = ['CURRENT_DAY', 'EXPIRY'] # opzione bermuda
		
	else:
		
		print ('tipo opzione %s non valutabile!!' %tipo_swpt)
		FQ(9998)


	zcCurvedata = buildZCurveListDate(zc_curve_data, cfg_base)
	
	trjTimeScad = buildTimetable(op_input['SCAD_TRJ'], tenor, expiry, freq, cfg_base)
	fixTimeScad = buildTimetable(op_input['SCAD_FIX'], tenor, expiry, freq, cfg_base)
	payTimeScad = buildTimetable(op_input['SCAD_PAY'], tenor, expiry, freq, cfg_base)
	tenTimeScad = buildTimetable(op_input['SCAD_TEN'], tenor, expiry, freq, cfg_base)
	calTimeScad = buildTimetable(op_input['SCAD_CALL'], tenor, expiry, freq, cfg_base)
	
	valuePrms_a = lmm_prms['a']
	valuePrms_b = lmm_prms['b']
	valuePrms_c = lmm_prms['c']
	valuePrms_d = lmm_prms['d']
	valuePrms_lc = lmm_prms['longCorr']
	valuePrms_be = lmm_prms['beta']
	valuePrms_de = lmm_prms['Delta']

	#updateCFGprms(valuePrms_a, "aaaaa =", cfg_base, cfg_tmp)
	#updateCFGprms(valuePrms_a, "aaaaa =", cfg_tmp, cfg_new)



	
	updateCFGprms(valuePrms_a, "a =", cfg_base, cfg_tmp)
	updateCFGprms(valuePrms_b, "b =", cfg_tmp, cfg_new)
	updateCFGprms(valuePrms_c, "c =", cfg_new, cfg_tmp)
	updateCFGprms(valuePrms_d, "d =", cfg_tmp, cfg_new)
	updateCFGprms(valuePrms_lc, "longCorr =", cfg_new, cfg_tmp)
	updateCFGprms(valuePrms_de, "Delta =", cfg_tmp, cfg_new)
	updateCFGprms(valuePrms_be, "beta =", cfg_new, cfg_tmp)
	
	updateCFGbyTimetable(trjTimeScad, 'trajectory', cfg_tmp, cfg_new)
	updateCFGbyTimetable(fixTimeScad, 'Fixings', cfg_new, cfg_tmp)
	updateCFGbyTimetable(payTimeScad, 'PayDates', cfg_tmp, cfg_new)
	updateCFGbyTenorTimetable(tenTimeScad, 'Tenors', cfg_new, cfg_tmp)
	updateCFGbyTimetable(calTimeScad, 'CallDates', cfg_tmp, cfg_new)
	updateCFGNominalAndStrike(nominal, strike, cfg_new, cfg_tmp)

	#DatesLevelsList
	updateCFGprms(zcCurvedata, "DatesLevelsList = ", cfg_tmp, cfg_new)
	
	#updateCFGprms(valuePrms_a, "aaaaa =", cfg_tmp, cfg_new)

	

	#updateCFGNominalAndStrike(nominal, strike, cfg_new, cfg_tmp)
	
	#updateCFGNominalAndStrike(nominal, strike, cfg_tmp, cfg_new)

	#updateCFGprms(valuePrms_a, "aaaaa =", cfg_base, cfg_tmp)
	#updateCFGprms(valuePrms_a, "aaaaa =", cfg_tmp, cfg_new)


	return cfg_new


def computeSimpleSum(data_out, fieldToSum):	

	list_res = data_out.keys()
	sumTmp = 0.0
	
	for resTmp in list_res:
		sumTmp = sumTmp  + data_out[resTmp][fieldToSum]		
	return sumTmp	

def computeSimpleAvg(data_out, fieldToSum):	

	list_res = data_out.keys()
	sumTmp = 0.0
	n_list = len(list_res)
	
	for resTmp in list_res:
		sumTmp = sumTmp  + data_out[resTmp][fieldToSum]		
	
	sumTmp = float(sumTmp/n_list)
	
	return sumTmp	



def computeWeightedSum(data_out, fieldToSum, fieldWeighted):
	
	
	normValue = computeSimpleSum(data_out, fieldWeighted)	

	list_res = data_out.keys()
	
	sumTmp = 0.0
	
	for resTmp in list_res:
		
		
		if (normValue < 0.00001):
			sumTmp = sumTmp  + 0.0
		else:
			sumTmp = sumTmp  + float(data_out[resTmp][fieldToSum]*data_out[resTmp][fieldWeighted]/normValue)
		
	return sumTmp	


def load_ptf_fromFile(inputListFile):

	fin = open(inputListFile, 'r')
	listInput = fin.readlines()
	dict_ptf_out = {}

	field_list = listInput[0].split("\t")


	
	n_f 	   = len(field_list)
	
	for i in range(1, len(listInput)):
	
		line_splittedTmp = listInput[i].split("\t")
		
		dict_ptf_out[i] = {}
	
		for j in range(0, n_f):
			
			valTmp = line_splittedTmp[j]
			
			fTmp   = field_list[j].split('\n')
			fTmp = fTmp[0] 
			
			dict_ptf_out[i][fTmp] = float(valTmp)
		

	return dict_ptf_out






def	dumpReportResults(file_out, data_out):
		
	
	ValoreTotale 	= computeSimpleSum(data_out, 'MC Price')
	ErrValoreTotale = computeSimpleSum(data_out, 'MC Err')
	NominaleTotale  = computeSimpleSum(data_out, 'Nominal')
	#N.
	FVmedio 	 = computeSimpleAvg(data_out, 'MC Price')
	StrikeMedio  = computeWeightedSum(data_out, 'Strike', 'MC Price')
	MatMedia 	 = computeWeightedSum(data_out, 'Maturity', 'MC Price')
	ExpMedia 	 = computeWeightedSum(data_out, 'Expiry', 'MC Price')
	NominalMedio = computeWeightedSum(data_out, 'Nominal', 'MC Price')
	
	fout = open(file_out, 'w')
	
	fout.write(('FV tot:\t %.4f\n')%(ValoreTotale))
	fout.write(('Err tot:\t %.4f\n')%(ErrValoreTotale))
	fout.write(('Nominale tot:\t %.2f\n')%(NominaleTotale))
	fout.write(('FV medio:\t %.4f\n')%(FVmedio))
	fout.write(('Strike medio:\t %.6f\n')%(StrikeMedio))
	fout.write(('Maturity media:\t %.4f\n')%(MatMedia))
	fout.write(('Expiry media:\t %.4f\n')%(ExpMedia))
	fout.write(('Nominale medio:\t %.4f\n')%(NominalMedio))
	
	fout.write(('------------------------\n'))
	
	swp_list = data_out.keys()
	
	
	#'Strike': strikeTmp, 'Nominal': nominalTmp, 'Expiry': expiryTmp, 'Maturity': matTmp, 'Freq': freqTmp, 'MC Price': mc_prices, 'MC Err': mc_err, 'P Call'

	fout.write(('Dettaglio valutazione swaption\n\n'))
	
	fout.write(('n.\t Expiry\t Maturity\t Strike\t Nominale\t MC Price\t MC Err\t PCall\n'))
	
	nTmp = 0
	for swpTmp in swp_list:
		
		expTmp 		= data_out[swpTmp]['Expiry']
		matTmp 		= data_out[swpTmp]['Maturity']
		strikeTmp 	= data_out[swpTmp]['Strike']
		nominalTmp 	= data_out[swpTmp]['Nominal']
		mcPriceTmp 	= data_out[swpTmp]['MC Price']
		mcErrTmp 	= data_out[swpTmp]['MC Err']
		pCallTmp 	= data_out[swpTmp]['P Call']

		fout.write(('%d\t %.2f\t %.2f\t %.4f\t %.2f\t %.4f\t %.6f\t %.4f\n')%(nTmp, expTmp, matTmp, strikeTmp, nominalTmp, mcPriceTmp, mcErrTmp, pCallTmp))
	
		nTmp = nTmp + 1
	fout.close()






def computeNtot(security_data_calib, shocklist):


	sec_list = security_data_calib.keys()

	n_sec = len(sec_list)
	
	n_tot = 0
	n_shock = len(shocklist)
	
	for i in range(0, n_sec):
		
		secTmp = sec_list[i]
		
		n_calib = len(security_data_calib[secTmp])
		
		for j in range(0, n_calib):

			n_tot = n_tot + n_shock
		
	return n_tot



def read(name):

	fp = open(name, "r")
	msg=""
	while( True ):
		line = fp.readline()
		if len(line) == 0 : break
		if line[0] == ';': continue
		Y = line.strip('\r\n')
		X = Y.strip()
		if len(X) == 0: continue
		if X[0] == ';': continue
		if X[0] == '#': continue
		msg += "%s\n" %X
	return msg



def line_recv(sock):
	msg=""
	while True:
		chunk = sock.recv(1)

		if( chunk == None ):
			raise Exception("Nbytes: ")
		if len(chunk) == 0 : return "No message"

		if    chunk[0] == '\r': count = 1
		elif chunk[0] == '\n': 
			if count == 1: return msg
			else:           msg += '\n'
			count =0
		else:
			msg += chunk
			count = 0


def bare_snd(sock, msg):
		totalsent = 0
		MSGLEN = len(msg)

		while totalsent < MSGLEN:
			sent = sock.send(msg[totalsent:])
			if sent == 0:
				raise RuntimeError("socket connection broken")
			totalsent = totalsent + sent
		return totalsent

def myrecv(sock, ll):
	while True:
		line = line_recv(sock)
		if len(line) == 0: break;

		ls = string.split(line, ':')	

		if len(ls) == 1:
			TAG = "MSG"
			VAL=ls[0]
		elif len(ls) == 2:
			TAG = ls[0]
			VAL = ls[1]
		elif len(ls) > 2:
			TAG = ls[0]
			VAL=ls[1]
			for n in range(2, len(ls)):
				VAL += ":%s" %(ls[n])
		ll[TAG] = VAL

def mysend(sock, msg):
	kk = msg.keys()
	for k in kk:
		pack = k + ":" + msg[k] + "\r\n"
		bare_snd(sock, pack)
	bare_snd(sock, "\r\n")

def sendDataToEval(folder_input, cfg_name, PORT, HOST, verbose):
	
	cfg_file_path = folder_input + '\\' + cfg_name
	cfg=read(cfg_file_path)

	protocol={}
	protocol["cfg_name"] = cfg_name
	protocol["content_length"] =  "%d" %len(cfg)
	# ------------------
	start_time = time.time()

	#create an INET, STREAMing socket
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	#now connect to the web server on port
	sock.connect((HOST, PORT))

	hello = line_recv(sock)

	#print 'Start elaboration '

	# ----------------------------------------------------------------------------

	# Ship the header protocol
	mysend(sock, protocol)

	n = bare_snd(sock, cfg)
	#print "sent tot", n
	# ------------------------

	ll  = {}
	myrecv(sock, ll)

	try:
	
		if (verbose == True):
		
			print ("*"*120)
			print (ll['RunResults'])
			print ("*"*120)
	

	
		splitted_res_v0 =  ll['RunResults'].split('Early_exercise_probability: ')	
		splitted_res_v1 = splitted_res_v0[1].split('\nOutOfSampleValue')
		p_call = float(splitted_res_v1[0])

	except:
		
		print ('ERRORE DI COMPILAZIONE DEL CFG!!!!')
		print (ll)
		p_call = float(splitted_res_v1[0])

	

	try:
		err = ll["Error"]
		print ("Execution completed with error")
		print (err)
	except KeyError:
		
		mc_err 	 = float(ll['McError'])
		mc_price = float(ll['Price'])
		
		#print 'MC Error XX: ', mc_err
		#print 'MC Price XX: ', mc_price
		
	dt_time =  (time.time() - start_time)


	return mc_err, mc_price, p_call, dt_time

