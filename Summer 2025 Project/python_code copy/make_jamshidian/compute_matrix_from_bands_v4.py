import sys
import pandas as pd
import numpy as np


def FQ(label):
	print ('------------- FIN QUI TUTTO OK  %s ----------' % (label))
	sys.exit()



def Jamshidian(date, upperB, lowerB):
    n = len(date)

#using numpy to make zero matrices instead of long code
    a = np.zeros((n-1,n-1))
    B = np.zeros((n-1,n))
    
#r is the first exercise date of the swaption (rows)
#c is the maturity of the swap (columns)
#B_r,c represents the notional value of a swap starting at r and maturing at c


    for c in range((n - 1)):
        for r in range((c + 1)): #we are only looking at r <= c because you cant exercise after maturity
        
#from Jamshidian and Evers paper we know that a_r,c = max(U_c - L_r, 0) so we apply this to get values for a
            x = upperB[c] - lowerB[r] 
            if x < 0:
                a[r][c] = 0.0
            else:
                a[r][c] = x

#we fill the last column of B using B_r,C = a_r,(C-1) - a_(r-1),(C-1)
    for r in range((n - 1)):
        if r == 0:
#we use (n-2) instead of (C-1) because (n-2) represents the last column of a
            B[r][(n - 1)] = a[r][(n - 2)] #
        else:
            B[r][(n - 1)] = a[r][(n - 2)] - a[(r - 1)][(n - 2)] 
   
#we then fill in the rest of the matrix using B_1,c = a_1,(c-1) - a_1,c
#and B_r,c = a_r,(c-1) - a_r,c - a_(r-1),(c-1) + a_(r-1),c

    for c in range((n - 1)):
        for r in range(c):
            if r == 0:
                B[r][c] = a[r][(c - 1)] - a[r][c]
            else:
                B[r][c] = a[r][(c - 1)] - a[r][c] - a[(r - 1)][(c - 1)] + a[(r - 1)][c]
    
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

#this function takes a list of values, l1, and two bounds, low and high.
#it takes the values in l1 that lie between the bounds and store these values along with their indexes in
#l2 and l2_indx respectively
def between_idx(l1, low, high):

	l2 = []
	l2_indx = []

	ln = len(l1)
	for i in range(0, ln):
		l2Tmp = l1[i]
        
		if (l2Tmp > low and l2Tmp <= high):
			l2.append(l2Tmp)
			l2_indx.append(i)

	return l2, l2_indx



#this function takes a list of target values (list_target), checks if they lie within specific intervals
# (list_ref) and then either rounds them up or down depending whcih bound they are closer to
def build_new_idx_column(list_ref, list_target, df_ref, name_new_col):
	df_ref[name_new_col] = 999.00

	for i in range(0, len(list_ref)):

#highTmp represents the upper bound of the interval
		highTmp = list_ref[i]

#if were at the first interval we choose lowTmp such that we dont cut off any values
		if (i == 0):
			lowTmp = -0.00001
			midTmp = lowTmp

#othrewise the lower boundary is just the previous boundary
		else:
			lowTmp = list_ref[i - 1]
			midTmp = (highTmp + lowTmp) / 2.0

#we use the previous function to find the values that lie within our bounds
		between_out = between_idx(list_target, lowTmp, highTmp)

		if (i == 0): lowTmp = 0.0

		indx_out = between_out[1]
		# val_out  = between_out[0]

#for each value that lies within our boundarys we either round it up or down 
		for idxTmp in indx_out:

			valTmp = list_target[idxTmp]

			if (valTmp >= midTmp):

				df_ref[name_new_col][idxTmp] = highTmp
			else:
				df_ref[name_new_col][idxTmp] = lowTmp

	return df_ref


#this function takes a pandas table (report) and saves it to either an excel or csv file with the name (outfile)
def save_report(report, outfile, file_format):

	if (file_format == 'xl'):

		sheet_name = 'Foglio3'
		writer = pd.ExcelWriter(outfile) #pandas excel writer function
		report.to_excel(writer, sheet_name)
		writer.save()

	else:

		report.to_csv(outfile, sep=',') #saves to a csv file separated by commas


#takes a maturity time in numbers and converts it to months or years
def convert_schedule(mat_to_convert):

#makes the input a float and rounds to one decimal place
	mat_r = np.round(float(mat_to_convert), 1)

#if the rounding is 0.5 or 1.5 it sets it to 6 or 18M, respectively but if its anything else then it writes
#it in years (rounded as well). 
#Do we only care for 6m and 18m or should i code all .5 months? 
	if (mat_r == 0.5):
		mat_n = '6M'
	elif (mat_r == 1.5):
		mat_n = '18M'
	else:
		mat_n = str(int(mat_to_convert))+ 'Y'

	return mat_n


#this function takes in some swaptions data (file_anag_ptf_swpt) and will store the calibrated matrix
#in a file called (file_ptf_swpt_calib)
def cmputeMatrixToCalib(file_anag_ptf_swpt, file_ptf_swpt_calib, file_type):
    
#reads the file with data on the swaptions
    df_bande = pd.read_csv(file_anag_ptf_swpt, sep='\t')

#two lists of reference maturities and expiries that we will sort the data into
    list_ref_mat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 50, 60]
    list_ref_exp = [0.08333, 2.0 / 12.0, 0.25, 0.5, 9.0 / 12.0, 1.0, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 50, 60]

#calculates the maturity of the swaption the way that Jamshidian does it (subtracting expiry from maturity)
    list_target_mat = df_bande['Mat'] - df_bande['Expiry'] 
    list_target_exp = df_bande['Expiry']

    name_new_col_1 = 'mat_to_calib'
    name_new_col_2 = 'exp_to_calib'

#uses the function from above to sort the maturities and expiries for the swaptions
#into the reference values we have a few lines above. These are stored in two new columns

    df_bande = build_new_idx_column(list_ref_mat, list_target_mat, df_bande, name_new_col_1)
    df_bande = build_new_idx_column(list_ref_exp, list_target_exp, df_bande, name_new_col_2)

#creates a pivot table to aggregate notionals of swaps by maturity and expiry
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
        
#we go through the pivot table and extract maturity and expiry values
        exp_ = pivot_out_.iloc[i]['exp_to_calib'].values[0]
        mat_ = pivot_out_.iloc[i]['mat_to_calib'].values[0]

#convert these values to strings
        exp_n = convert_schedule(exp_)
        mat_n = convert_schedule(mat_)

#append our lists with these values
        exp_list.append(exp_n)
        mat_list.append(mat_n)

    df_data_dict = {'exp_to_calib': exp_list, 'mat_to_calib': mat_list}
	#df_ = pd.DataFrame(data = df_data_dict)

    df_to_calib = pd.DataFrame(df_data_dict)
    
    
    file_ = file_anag_ptf_swpt.split('.txt')[0]
    print(file_)
    file_ = file_.split('/')[-1] #had to change this because my computer uses different directory finding
    file_ = file_.split('swaptions_')[1]
    file_n =  file_.replace('.', '')

    df_to_calib['PTF_LABEL'] = file_n

	#save_report(pivot_out, file_ptf_swpt_calib, file_type)
    save_report(df_to_calib, file_ptf_swpt_calib, file_type)


def convertBtoDict(B):

#stores the number of rows and columns in B, which we know correspond to exercise dates and maturities.
	nr = len(B)
	nc = len(B[0])
	
	B_dict = {}

	for i in range(0, nr):
		B_dict[i] = {}
		for j in range(0, nc):
			B_dict[i][j] = B[i][j] #storing the notional value
#this new dictionary lets us easily see the value of the notional at each maturity date corresponding
#to each of the exercise dates

	return B_dict


#This function will take the portfolio data and write it into a file separated by tabs
def dumpSwaptionPtfAnag(Bn, file_to_dump, tenor, strikeIn):

#opens the file where it will write
	file_to_dump=open (file_to_dump, "w")
	
#gets a list of all exercise dates and maturities
	exp_list = Bn.keys()
	mat_list = Bn[0].keys()

#gets the number of exercise dates and maturities
	n_exp = len(exp_list)
	n_mat = len(mat_list)

#Writes a line separating strike, notional, expiries and maturities
	file_to_dump.write("Strike\tNominal\tExpiry\tMat\n")
	
	
	for i in range(0, n_exp):
		for j in range(0, n_mat):
#multiplied by tenor to get the exact time since tenor represents the time step 
			expTmp = i*tenor
			matTmp = j*tenor
			notionalTmp = float(Bn[i][j])
			strike = strikeIn

#if the notional is bigger than 0.0001, we write the strike, notional amount, expiry time and maturity time
#into the file
			if (notionalTmp > 0.0001):
				file_to_dump.write("%s\t %.4f\t %s\t %s\n" %(strike, notionalTmp, expTmp, matTmp))
                
			else:
				continue



if __name__ == "__main__":

	# --------------MARKET FOCUS ----------------------------------------------------

#takes a list of portfolios to process from an excel file and smiliary to the prepayments
#file it sorts out only the rows we want to process

	ptf_to_process = pd.read_excel(open('/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_jamshidian/input/ptf_to_process_jam_v1.xlsx', 'rb'), sheet_name='PTF_TO_PROCESS')

	idx = ptf_to_process['TO_DO'] == True

	ptf_to_process_ = ptf_to_process[idx]
	ptf_to_process_ = ptf_to_process_.reset_index()

#list of prepayments
	pp_list = ['pp0.01_0.02', 'pp0.01_0.03', 'pp0.01_0.05', 'pp0.01_0.07', 'pp0.01_0.1']
	tenor = 0.5 #time steps between exercise dates and maturities

	for i in range(0, len(ptf_to_process_)):

#extracts the label to save and K for each portfolio        
		band_ = str(ptf_to_process_.iloc[i]['LABEL_TO_SAVE'])
		strikeIn = float(ptf_to_process_.iloc[i]['K'])

		for j in range(0, len(pp_list)):

#creating file names and destinations
			band_file = band_ + '_' + pp_list[j] + '.xlsx'
			file_bande_v0 = '/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_jamshidian/input/bande_per_net/' + band_file

			# -------------- output --------------------------------------------------------

#creating more file names and destinations
			file_bande_v0_s   = file_bande_v0.split('/')
			file_bande_v0_s   = file_bande_v0_s[-1].split('xlsx')[0] #had to change to -1 because of the way m computer reads files
			file_ptf_out 	  = 'ptf_swaptions_' + file_bande_v0_s + 'txt'
			file_ptf_to_calib = 'swaptions_to_calib_' + file_bande_v0_s + 'txt'

			anag_ptf_swpt = '/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_jamshidian/output/ptf_anag/' + file_ptf_out
			ptf_swpt_calib = '/Users/bilal/Desktop/Summer 2025 Project/python_code copy/make_jamshidian/output/data_to_calib/' + file_ptf_to_calib

			# ------------------------------------------------------------------------------

			ptf_swpt_calib_xl = ptf_swpt_calib + '.xlsx'
			ptf_swpt_calib_csv = ptf_swpt_calib + '.csv'

			df = pd.read_excel(file_bande_v0)

#extracts start dates and bounds from the file read above
			date_inizio = df['DATE INIZIO']
			lower_bound = df['LOWER']
			upper_bound = df['UPPER']

#creates the matrix B of notionals of swaptions using the above Jamshidian function and then converts into
#a dictionary using a function explained below
			B = Jamshidian(date_inizio, upper_bound, lower_bound)
			B_dict = convertBtoDict(B)

			print('Start elab')
			dumpSwaptionPtfAnag(B_dict, anag_ptf_swpt, tenor, strikeIn)
			# function that associates to each expiry the sum of the nominals divided by tenor
            
#creates calibration matrix
			cmputeMatrixToCalib(anag_ptf_swpt, ptf_swpt_calib_xl, 'xl')
			print('End elab!!')





