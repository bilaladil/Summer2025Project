import numpy as np
import pandas as pd
from scipy.stats import norm, ncx2
import sys





def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



def Pt_MKT(p_time, rf_times, rf_values):

    RateTime = np.interp(p_time, rf_times, rf_values)
    p_out = np.exp(- RateTime * p_time)

    return p_out


def SwapRate(tenor, rf_times, rf_values, t_exp, maturity, type_curve = 'rate'):

	tmp_time = np.arange(0., maturity, tenor) + tenor + t_exp
	tmp_time = np.append(t_exp,tmp_time)

	if type_curve == 'rate':
		df_annuity = []
		for t in tmp_time:
			df_annuity.append(Pt_MKT(t, rf_times,rf_values))
		df_annuity = np.asarray(df_annuity)
		#df_annuity = np.exp(-rf_time * np.interp(tmp_time,rf_times, rf_values))
	elif type_curve == 'discount':
		df_annuity = np.exp(np.interp(tmp_time,rf_times, np.log(rf_values)))
	else:
		raise Exception('error: valore tipologia curva non corretto')

	df_exp = df_annuity[0]
	df_mat = df_annuity[-1]

	ForwardAnnuityPrice = np.diff(tmp_time) * df_annuity[1:]

	swap = (df_exp - df_mat) / ForwardAnnuityPrice.sum()

	return swap.copy(), ForwardAnnuityPrice.sum()

def CurveFromDictToList(rf_Curve):
	rf_times = rf_Curve.keys()
	rf_times.sort()
	rf_values = []

	for j in xrange(0, len(rf_times)):
		rf_values.append(rf_Curve[rf_times[j]])
	return np.array(rf_times), np.array(rf_values)



#################################################
#   LMM su Swaptions
#################################################

def LMM_correlation_matrix(param, times):

	beta = float(param[4])
	rho  = float(param[5])

	T_t = times[:, None] - times

	return rho + (1. - rho) * np.exp(- beta * np.absolute(T_t))

def LMM_IntVol_jk_dt(param, t0, t1, t_k, t_j):
	# integrale in forma analitica della volatilita'

	# il metodo ha senso solo per t_j e t_k maggiore uguale di t_exp
	#if np.any(np.logical_or(t_j < t_exp, t_k < t_exp)):
	#    raise Exception("I tempi nell'integrale in forma chiusa sono errati")

	a = float(param[0])
	b = float(param[1])
	c = float(param[2])
	d = float(param[3])


	tmp00 = 1. / (4. * np.power(c, 3))
	tmp01 = np.exp(-c * (t_j + t_k))

	tmp101 = - np.exp(2. * c * t0) * (1. + c * (t_j + t_k - 2. * t0) + 2. * np.power(c,2.) * (t_j - t0) *(t_k - t0))
	tmp102 = np.exp(2. * c * t1) * (1. + c * (t_j + t_k - 2. * t1) + 2. * np.power(c,2.) * (t_j - t1) *(t_k - t1))

	tmp1 = np.power(b, 2) * (tmp101 + tmp102)

	tmp20 = 2. * a * d * (np.exp(c * t_j) + np.exp(c * t_k)) * \
			(np.exp(c * t0) - np.exp(c * t1))
	tmp21 = np.power(a, 2) * (np.exp(2. * c * t0) - np.exp(2. * c * t1))
	tmp22 = 2. * c * np.power(d, 2) * np.exp(c * (t_j + t_k)) * (t0 - t1)

	tmp2 = - 2. * np.power(c, 2) * (tmp20 + tmp21 + tmp22)

	tmp310 = np.exp(2.*c * t0) * (-1. - c * (t_j + t_k - 2. * t0))
	tmp311 = np.exp(2.*c * t1) * (1. + c * (t_j + t_k - 2. * t1))

	tmp31 = a * ( tmp310 + tmp311 )

	tmp320 = np.exp(c * (t_k + t0)) * (-1. - c * t_j + c * t0)
	tmp321 = np.exp(c * (t_j + t0)) * (-1. - c * t_k + c * t0)
	tmp322 = np.exp(c * (t_k + t1)) * ( 1. + c * t_j - c * t1)
	tmp323 = np.exp(c * (t_j + t1)) * ( 1. + c * t_k - c * t1)

	tmp32 = 2. * d * (tmp320 + tmp321 + tmp322 + tmp323)

	tmp3 = 2. * b * c * (tmp31 + tmp32)

	res = tmp00 * tmp01 * (tmp1 + tmp2 + tmp3)

	if type(res) == np.ndarray:
		res[np.logical_or((t_k - t0) < 0., (t_j - t0) < 0.)] = 0.
	else:
		if (t_k - t0) < 0. or (t_j - t0) < 0.:
			res = 0.

	return res

def LMM_vola_swaptions_Rebonato(param, curve, shift, swap_rate, t_exp, maturity_swap, tenor_swap):

	# tempi di riferimento dello swap
	# t_exp + tenor_swap, .... , t_maturity compresi gli estremi
	# non e' incluso t_exp
	tmp_time = np.arange(0., maturity_swap, tenor_swap) + tenor_swap + t_exp
	# sigma_k si riferisce a T_{k-1}
	tmp_time_vola = tmp_time - tenor_swap

	# Calcolo i fattori di sconto
	#df = np.exp(- tmp_time * np.interp(tmp_time, curve['TIME'], np.log(curve['VALUE'])))
	#df_prec = np.exp( - tmp_time_vola * np.interp(tmp_time_vola, curve['TIME'], np.log(curve['VALUE'])))
	#df = np.exp(np.interp(tmp_time, curve[0], np.log(curve[1])))
	df = np.exp(np.interp(tmp_time, curve['TIME'], np.log(curve['DISCOUNT'])))
	df_prec = np.exp(np.interp(tmp_time_vola, curve['TIME'], np.log(curve['DISCOUNT'])))

	# tassi Libor iniziali
	Forward0 = (df_prec/df - 1.) / tenor_swap

	#print('Forward0: ', Forward0)
	#idx_exp = 0
	#idx_mat = self.idx_tenors[t_maturity] - 1

	# dal successivo di t_exp al t_maturity compreso
	#idx = np.arange((idx_exp+1), (idx_mat + 1), 1)
	w_denom = (tenor_swap * df)
	w_0 = (tenor_swap * df) / w_denom.sum()

	#idx_vola = idx - 1
	# formula di Rebonato
	s_t_jk = LMM_IntVol_jk_dt(param = param, t0 = 0., t1=t_exp, t_k=tmp_time_vola[:, None], t_j=tmp_time_vola)
	cov = LMM_correlation_matrix(param= param, times=tmp_time) * s_t_jk

	#shift = shift/100.0
	tmp = (w_0 * (Forward0 + shift))

	#print('swap_rate: ', swap_rate)
	#print('shift: ', shift)
	swap = swap_rate + shift
	#FQ(77)
	var_estimated = np.dot(tmp, np.dot(cov, tmp.transpose())) / (swap * swap * t_exp)

	return np.sqrt(var_estimated)

def loss_LMM_swaptions(list_model_params, curve, mkt_prices, tenor_data,call_type, power, absrel):

	model_tmp = compute_LMM_prices_swaptions(list_model_params, curve, mkt_prices, tenor_data, call_type)

	diff = np.absolute(model_tmp['model price'] - mkt_prices['market price'])
	if absrel == 'rel':
		diff = diff /np.absolute(mkt_prices['market price'])

	diff = np.power(diff,power)

	return diff.sum()

"""
def compute_LMM_prices_swaptions(parameters_list, curve, mkt_data, tenor_data, call_type):

	shift = parameters_list[6]

	model_res = pd.DataFrame(index=mkt_data.index, columns=['model vola','model price'])

	for i in mkt_data.index:
		model_res.at[i,'model vola'] = LMM_vola_swaptions_Rebonato(param = parameters_list,
																	curve = curve,
																	shift = shift,
																	swap_rate = mkt_data.at[i,'swap'] ,
																	t_exp = mkt_data.at[i,'expiry'],
																	maturity_swap = mkt_data.at[i,'maturity'],
																	tenor_swap = tenor_data)

		model_res.at[i,'model price'] = fromVolaATMToPrice(t_exp = mkt_data.at[i,'expiry'],
														   maturity = mkt_data.at[i, 'maturity'],
														   tenor = tenor_data,
														   vol   = model_res.at[i,'model vola'],
														   rf_times  = curve['TIME'].values,
														   rf_values = curve['VALUE'].values,
														   shift = shift,
														   call_type = call_type,
														   type_curve='discount')

	return model_res.copy()
"""

def fromVolaATMToPrice(t_exp, maturity, tenor, vol, rf_times,rf_values, shift, call_type, type_curve='rate'):

	#if (vol <= 0.0) or (vol >= 2.0):
	#	print ("AAA in swaption: it takes 0 < vol < 2; found ", vol)


	srate, ForwardAnnuityPrice =  SwapRate(tenor, rf_times, rf_values, t_exp, maturity, type_curve='rate')

	call = (srate+shift) * call_type * (2.0 * norm.cdf(call_type * 0.5 * vol * np.sqrt(t_exp)) - 1.0)

	price = ForwardAnnuityPrice * call

	return price


# ---------- Funzione di calcolo norma da ottimizzare nel caso ATM ------------------------------------
# power=1 metrica di Manhattan
# power=2 metrica euclidea
# se absrel='rel' viene calcolata la norma delle differenze relative
# mkt_data e' un DataFrame contenente tre series 'time', 'strike' e 'market price'
def loss_LMM_caps(list_model_params, curve, mkt_prices, tenor_cap, shift, power, absrel):
	model_price_tmp = np.array(compute_LMM_caps_prices(list_model_params, curve, mkt_prices, tenor_cap, shift))
	diff = np.absolute(model_price_tmp - mkt_prices['market price'])
	if absrel == 'rel':
		diff = diff / np.absolute(mkt_prices['market price'])

	diff = np.power(diff, power)

	return diff.sum()


# ---------- Lista prezzi dei Cap ATM da modello LMM -----------------------------
# i parametri del modello vengono passati tramite una lista [a,b,c,d,beta,rho]
def compute_LMM_caps_prices(parameters_list, curve, mkt_data, tenor_cap, shift):
	cap_prices = []

	for i in range(0, len(mkt_data['time'])):
		time = mkt_data['time'][i]
		strike = mkt_data['strike'][i]
		cap_tmp = 0.
		for t in np.arange(0, time, step=tenor_cap):
			cap_tmp += LMM_Caplet(parameters_list, curve, t, t + tenor_cap, strike, shift)

		cap_prices.append(cap_tmp)

	return cap_prices

def LMM_Caplet(param, curve, reset_time, exercise_time, strike, shift):

	'''
	The Black implied volatility at time 0 of a caplet expiring at time Ti-1 and paying at time Ti
	hence, whose underlying is Li

	Return the caplet volatility at time t=0,
	computed in terms of the the instantaneous
	volatility function form proposed by Rebonato.
	'''

	s_t = LMM_IntVol_jk_dt(param=param, t0=0., t1=reset_time, t_k=reset_time, t_j=reset_time)
	vol = np.sqrt(s_t / reset_time)

	price_cap = Black_shifted_Caplet(curve, reset_time, exercise_time, 1., strike, shift, vol)

	return price_cap


# ---------- Funzione di calcolo norma da ottimizzare nel caso ATM ------------------------------------
# power=1 metrica di Manhattan
# power=2 metrica euclidea
# se absrel='rel' viene calcolata la norma delle differenze relative
# mkt_data e' un DataFrame contenente tre series 'time', 'strike' e 'market price'
def loss_LMM_caplets(list_model_params, curve, mkt_prices, tenor_cap, shift, power, absrel):
	model_price_tmp = np.array(compute_LMM_caplets_prices(list_model_params, curve, mkt_prices, tenor_cap, shift))
	diff = np.absolute(model_price_tmp - mkt_prices['market price'])
	if absrel == 'rel':
		diff = diff / np.absolute(mkt_prices['market price'])

	diff = np.power(diff, power)

	return diff.sum()


# ---- Lista prezzi dei Caplet da modello LMM -----------------------------
# i parametri del modello vengono passati tramite una lista [a,b,sigma,eta,rho]
def compute_LMM_caplets_prices(parameters_list, curve, market_data, tenor_caplet, shift):
	caplet_prices = []

	for i in range(0, len(market_data)):
		time_tmp = market_data['time'][i]
		strike_tmp = market_data['strike'][i]
		caplet_tmp = LMM_Caplet(parameters_list, curve, time_tmp - tenor_caplet, time_tmp, strike_tmp, shift)

		caplet_prices.append(caplet_tmp)

	return caplet_prices


