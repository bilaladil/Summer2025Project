import sys
import pandas as pd
import numpy as np


def FQ(label):
	print ('------------- FIN QUI TUTTO OK  %s ----------' % (label))
	sys.exit()


import funzioni_base as fb




import pandas as pd
import os
import glob

def save_report(report, outfile, flag_tipo_file):
	"""
    Take a report and save it to a single Excel file
    """
	if (flag_tipo_file == 'xl'):

		sheet_name = 'Foglio3'
		print('outfile: ', outfile)
		writer = pd.ExcelWriter(outfile)
		report.to_excel(writer, sheet_name)
		writer.save()

	else:

		report.to_csv(outfile, sep=',')


if __name__ == "__main__":

	# --------------MARKET FOCUS ----------------------------------------------------

	# import necessary libraries

	# use glob to get all the csv files
	# in the folder
	#os.chdir('output/data_to_calib/processed_extra_base')
	#os.chdir('output\\data_to_calib\\last_bands\\')
	#os.chdir('output\\data_to_calib\\31_12_2021\\')
	#os.chdir('output\\data_to_calib\\31_03_2022\\')
	#os.chdir('output\\data_to_calib\\28_08_2022\\')
	#os.chdir('output\\data_to_calib\\30_08_2022\\')
	#os.chdir('output\\data_to_calib\\04_09_2022\\')
	os.chdir('output\\data_to_calib\\27_11_2023\\')

	path = os.getcwd()

	xlsx_files = glob.glob(os.path.join(path, "*.xlsx"))

	# loop over the list of csv files
	i = 0
	frames = []
	for f in xlsx_files:
		# read the csv file
		df_ = pd.read_excel(f)
		df_ = df_.drop('Unnamed: 0', axis=1)

		#i = i + 1

		frames.append(df_)

	master_calib_df = pd.concat(frames)

	outfile = 'master_to_calib.xlsx'
	save_report(master_calib_df, outfile, 'xl')