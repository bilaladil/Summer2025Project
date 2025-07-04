#Code for using the Vasicek model to price swaptions
import numpy as np
import scipy 

#we know that swaptions can be expressed as a portfolio of zero coupon bonds paying c_i at T_i

#we start by writing a function to compute the price of zero coupon bonds under the Vasicek model
def ZCB_price(r, theta, sigma, k, T, t):
    
    B = (1 / k)*(1 - np.exp((-k) * (T - t)))
    A = np.exp((theta - ((sigma ** 2) / (2 * (k ** 2))) ) * (B - T + t) - ((sigma ** 2) / (4 * k)) * (B ** 2))
    
    P = A * np.exp(- B * r)
    return A, B, P


#we want to use Jamshidians trick to find rstar when sum of ci * ZCB price = K

#we compute the present value of the fixed leg of the swaption and then use Jamshidian's trick to find
#rstar when this equals the sum of ZCBs
def ComputeSwapPV(times, coupons, ZCBprices):
    
    if len(times) == len(coupons):
        
        n = len(times)

        swapPV = 0
        for i in range(0, n):
            swapPV += coupons[i] * ZCBprices[i]
        
    
        return swapPV
    
    else:
        print("Array sizes do not match. len(times) = ",len(times),"len(coupons) = ", len(coupons))

#we find the A and B values under the Vasicek model at all payment dates.
#these are important because they are needed for f and fprime       
def ComputeABValues(times, coupons, r, theta, sigma, k, t):
    
    if len(times) == len(coupons):
        
        n = len(times)
        Avalues = []
        Bvalues = []
    
        for i in range(0, n):
            
            Aval, Bval, Pval = ZCB_price(r, theta, sigma, k, times[i], t) 
            Avalues.append(Aval)
            Bvalues.append(Bval)
            
        
        return Avalues, Bvalues
    
    else:
        print("Array sizes do not match. len(times) = ",len(times),"len(coupons) = ", len(coupons))


def f(r, coupons, A, B, K):
    
    runningsum = 0
    end = len(coupons)
    for i in range(0, end):
        runningsum += coupons[i] * A[i] * np.exp(-B[i] * r)
    
    return runningsum - K 


def fprime(r, coupons, A, B):
    
    runningsum = 0
    end = len(coupons)
    for i in range(0, end):
        runningsum += - B[i] * coupons[i] * A[i] * np.exp(-B[i] * r)
        
    return runningsum


def JamshidiansTrick(f, r0, fprime):
    
    rstar = scipy.optimize.newton(f, r0, fprime)
    
    return rstar

#we write a function to calculate the zero coupon bond prices at t of zero coupon bonds with 
#maturities at each of our payment dates
def ComputeZCBPrices(times, r_0, theta, sigma, k, t):
    
    ZCBprices = []
    
    for T_i in times:
        A, B, ZCB = ZCB_price(r_0, theta, sigma, k, T_i, t)
        ZCBprices.append(ZCB)
        
    return ZCBprices


#we have a function which compute zero coupon bond prices at option expiry for each of our zero coupon bonds.
#we use rstar and let each one work as the strike price for the respective zero coupon bond
def ComputeStrikePrices(times, rstar, theta, sigma, k, T):
    
    strikeprices = []
    
    for T_i in times:
        A, B, K = ZCB_price(rstar, theta, sigma, k, T_i, T)
        strikeprices.append(K)
        
    return strikeprices
            

def ComputeEuropeanSwaptionPrice(cp, times, coupons, r_0, theta, sigma, k, T, t): 
    
    ZCBprices_at0 = ComputeZCBPrices(times, r_0, theta, sigma, k, 0)
    ZCBprices_atT = ComputeZCBPrices(times, r_0, theta, sigma, k, T)
    
    Avalues, Bvalues = ComputeABValues(times, coupons, r_0, theta, sigma, k, t) 
    
    swapPV = ComputeSwapPV(times, coupons, ZCBprices_at0) 
    
    def f1(r):
        return f(r_0, coupons, Avalues, Bvalues, swapPV)
    def fprime1(r):
        return fprime(r_0, coupons, Avalues, Bvalues)

    rstar = JamshidiansTrick(f1, r_0, fprime1)

    strikeprices = ComputeStrikePrices(times, rstar, theta, sigma, k, T)

    if len(coupons) == len(times):
        
        n = len(coupons)    
        runningsum = 0
        
        if cp == 1: #call
            for i in range(0,n):
                runningsum += coupons[i] * max(ZCBprices_atT[i] - strikeprices[i] , 0)
        elif cp == -1: #put
            for i in range(0,n):
                runningsum += coupons[i] * max(strikeprices[i] - ZCBprices_atT[i] , 0)
         
        price = runningsum
        return price
    
    else:
        print("Array sizes do not match.")


#now that we have covered the easier case of a European swaption, we need to tackle the harder problem of
#a Bermudan swaption

#the first step is to simulate values for the short rate, r, using the Vasicek dynamics

def VasicekShortRateSimulations(r_0, nr, n, k, theta, sigma, T, t):
    
    dt = (T - t)/n
    
    r_val = np.zeros((nr,n+1))
    rand = np.random.randn(nr,n)
    
    r_val[:,0] = r_0
    for i in range(nr):
        for j in range(1, n+1):
            mean = r_val[i, j-1] * np.exp( (-k) * dt ) + theta * (1 - np.exp( (-k) * dt ))
            var = ((sigma ** 2) / (2 * k)) * (1 - np.exp( (- 2 * k) * dt ))
            
            r_val[i,j] = mean + np.sqrt(var) * rand[i, j - 1]
            
    return r_val


def ComputeTargetContValues(cp, nr, n, times, coupons, r_0, k, theta, sigma, T, t):
    
    dt = (T - t)/n
    
    r_val = VasicekShortRateSimulations(r_0, nr, n, k, theta, sigma, T, 0)
    
    TargetContVal = np.zeros((nr, n+1))
    
    for i in range(nr):
        TargetContVal[i, -1] = ComputeEuropeanSwaptionPrice(cp, times, coupons, r_val[i, -1], theta, sigma, k, T, T)
    
    for i in range(nr):
        for j in range(n-1,0,-1):
            TargetContVal[i,j] = np.exp(- r_val[i,j] * dt) * TargetContVal[i,j+1] 
    
    return TargetContVal
             
            
def ComputeEarlyExerciseValues(cp, nr, n, times, coupons, r_0, k, theta, sigma, T, t):
    
    dt = (T - t)/n
    
    r_val = VasicekShortRateSimulations(r_0, nr, n, k, theta, sigma, T, 0)
    
    EarlyExerciseVal = np.zeros((nr, n+1))
    
    for i in range(nr):
        EarlyExerciseVal[i, -1] = ComputeEuropeanSwaptionPrice(cp, times, coupons, r_val[i, -1], theta, sigma, k, T, T)

    for i in range(nr):
        for j in range(n-1,0,-1):
            currenttime = j * dt
            EarlyExerciseVal[i,j] = ComputeEuropeanSwaptionPrice(cp, times, coupons, r_val[i,j], theta, sigma, k, T, currenttime) 
        
    return EarlyExerciseVal



def ComputeLLSContValues():

    #need to find swap rates to see when the swaption is ITM
    #use this from main_v2 lines 364 - 401
    
    """#this function calculates the strike (par swap rate) for a swap that starts in t_exp years
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

        return swap"""
    


def ComputeBermudanSwaptionPrice(cp, nr, n, times, coupons, r_0, k, theta, sigma, T, t ):
    
    dt = (T - t)/n
    
    r_val = VasicekShortRateSimulations(r_0, nr, n, k, theta, sigma, T, t)
    T
    
    
               
  
    
    
    
    
    
    
    
    
    
    






    