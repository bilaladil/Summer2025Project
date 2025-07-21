#Code for using the Vasicek model to price swaptions
import numpy as np
from scipy.optimize import newton

#throughout this code, t is the time at which we are valuing and T is the option expiry

#option expiries {1M, 2M, 3M, 6M, 9M, 1Y, 18M, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y, 15Y, 20Y, 25Y, 30Y}
#swap maturities {1Y, 2Y, 3Y,..., 10Y, 15Y, 20Y, 25Y, 30Y}

 

#things to change:
#how variables, maturities and expiries are input
#in our swaps, payments are always 6 months apart so need to fix that
#


#we know that swaptions can be expressed as a portfolio of zero coupon bonds paying c_i at T_i

#we start by writing a function to compute the price of zero coupon bonds under the Vasicek model
def ZCB_price(r, theta, sigma, k, T, t):
    
    B = (1 / k)*(1 - np.exp((-k) * (T - t)))
    A = np.exp((theta - ((sigma ** 2) / (2 * (k ** 2))) ) * (B - T + t) - ((sigma ** 2) / (4 * k)) * (B ** 2))
    
    P = A * np.exp(- B * r) # (Equation 3.39)
    return A, B, P


#we want to use Jamshidians trick to find rstar when sum of ci * ZCB price = 1
#(PV fixed = PV floating, but we let PV floating = 1)

#this function computes the PV of the fixed leg of the swap
def ComputeSwapPV(times, coupons, r_0, theta, sigma, k, t):
    
    ZCBprices = ComputeZCBPrices(times, r_0, theta, sigma, k, t)
    
    if len(times) == len(coupons):
        
        n = len(times)

        swapPV = 0
        for i in range(0, n):
            swapPV += coupons[i] * ZCBprices[i]
        
        return swapPV
    
    else:
        print("Array sizes do not match. len(times) = ",len(times),"len(coupons) = ", len(coupons))

#we find the A and B values under the Vasicek model at all payment dates.
#these are important because they are needed as inputs for f and fprime to find rstar using Jamshidians      
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


#(PVfixed = PVfloating becomes PVfixed - PVfloating = 0 or PVfixed - 1 = 0 and for Jamshidians we want to solve
#this for rstar)
def f(r, coupons, A, B):
    
    runningsum = 0
    end = len(coupons)
    for i in range(0, end):
        runningsum += coupons[i] * A[i] * np.exp(- B[i] * r)
    
    return float(runningsum - 1) 

#derivative of above
def fprime(r, coupons, A, B):
    
    runningsum = 0
    end = len(coupons)
    for i in range(0, end):
        runningsum += - B[i] * coupons[i] * A[i] * np.exp(- B[i] * r)
        
    return float(runningsum)

#value of r when PVfixed - 1 = 0
def JamshidiansTrick(f, r0, fprime):
    
    rstar = newton(f, r0, fprime)
     
    return rstar


#we write a function to calculate the zero coupon bond prices at t of zero coupon bonds with 
#maturities at each of our payment dates
def ComputeZCBPrices(times, r_0, theta, sigma, k, t):
    
    ZCBprices = []
    
    for T_i in times:
        ZCB = ZCB_price(r_0, theta, sigma, k, T_i, t)[2]
        ZCBprices.append(ZCB)
        
    return ZCBprices


#we have a function which compute zero coupon bond prices at option expiry for each of our zero coupon bonds.
#we use rstar and let each one work as the strike price for the respective zero coupon bond
def ComputeStrikePrices(times, rstar, theta, sigma, k, T):
    
    strikeprices = []
    
    for T_i in times:
        K = ZCB_price(rstar, theta, sigma, k, T_i, T)[2]
        strikeprices.append(K)
        
    return strikeprices

#ChatGPT suggested doing this for opion expiry (when t = T) because we know the value r and we cant use
#Jamshidians because T - t becomes 0
#I didnt write any of this function, I took it straight from chatgpt
def ComputeSwaptionIntrinsicValue(cp, times, coupons, r, theta, sigma, k, t):
    """
    Computes the intrinsic value of a swaption at expiry.

    Parameters:
        cp      : 1 for payer, -1 for receiver
        times   : list of payment dates
        coupons : fixed leg cash flows
        r       : short rate at expiry
        theta, sigma, k : Vasicek parameters
        t       : current time (i.e., expiry)

    Returns:
        float: payoff at expiry
    """
   
    
    PV_fixed = ComputeSwapPV(times, coupons, r, theta, sigma, k, t)
    PV_float = ComputeSwapPV(times, np.ones_like(coupons), r, theta, sigma, k, t)
    
    return max(cp * (PV_fixed - PV_float), 0)
            

def ComputeEuropeanSwaptionPrice(cp, times, coupons, r_0, theta, sigma, k, T, t): 
    
    ZCBprices_atT = ComputeZCBPrices(times, r_0, theta, sigma, k, T) #computing ZCB prices at expiry
    Avalues, Bvalues = ComputeABValues(times, coupons, r_0, theta, sigma, k, t)  #getting A and B values to use
    #for Jamshidians 
    
    #swapPV = ComputeSwapPV(times, coupons, r_0, theta, sigma, k, 0) 
    
    def f1(r):
        return f(r, coupons, Avalues, Bvalues)
    def fprime1(r):
        return fprime(r, coupons, Avalues, Bvalues)

    rstar = JamshidiansTrick(f1, r_0, fprime1) #finding rstar 

    strikeprices = ComputeStrikePrices(times, rstar, theta, sigma, k, T) #computing strike prices using rstar

    if len(coupons) == len(times):
        
        n = len(coupons)    
        runningsum = 0
        
        #calculating swaption payoffs
        if cp == 1: #payer
            for i in range(0,n):
                runningsum += coupons[i] * max(ZCBprices_atT[i] - strikeprices[i] , 0)
        elif cp == -1: #receiver
            for i in range(0,n):
                runningsum += coupons[i] * max(strikeprices[i] - ZCBprices_atT[i] , 0)
         
        price = runningsum
        return price
    
    else:
        print("Array sizes do not match.")


#now that we have covered the easier case of a European swaption, we need to tackle the harder problem of
#a Bermudan swaption

#the first step is to simulate values for the short rate, r, using the Vasicek dynamics

def VasicekShortRateSimulations(nr, n, r_0, theta, sigma, k, T, t):
    
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

#we compute target continuation values using backwards discounting that we will aim to hit using Linear Least Squares
def ComputeTargetContValues(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t):
    
    dt = (T - t)/n
    
    r_val = VasicekShortRateSimulations(nr, n, r_0, theta, sigma, k, T, 0)
    
    TargetContVal = np.zeros((nr, n+1))
    
    for i in range(nr):
        
        shiftedtimes = []
        for t_i in times:
            t_i = t_i + T
            shiftedtimes.append(t_i)
        
        #ChatGPT suggesting to use the IntrinsicValue calculations at expiry
        #TargetContVal[i, -1] = ComputeEuropeanSwaptionPrice(cp, shiftedtimes, coupons, r_val[i, -1], theta, sigma, k, T, T)
        TargetContVal[i, -1] = ComputeSwaptionIntrinsicValue(cp, shiftedtimes, coupons, r_val[i, -1], theta, sigma, k, T)
    
    
    #backward discounting for the other steps
    for i in range(nr):
        for j in range(n-1,0,-1):
            TargetContVal[i,j] = np.exp(- r_val[i,j] * dt) * TargetContVal[i,j+1] 
    
    return TargetContVal
           

#computing early exercise values at each time point to see if it is optimal at any point
def ComputeEarlyExerciseValues(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t):
    
    dt = (T - t)/n
    
    r_val = VasicekShortRateSimulations(nr, n, r_0, theta, sigma, k, T, 0)
    
    EarlyExerciseVal = np.zeros((nr, n+1))
    
    for i in range(nr):
        #EarlyExerciseVal[i, -1] = ComputeEuropeanSwaptionPrice(cp, times, coupons, r_val[i, -1], theta, sigma, k, T, T)
        EarlyExerciseVal[i, -1] = ComputeSwaptionIntrinsicValue(cp, times, coupons, r_val[i, -1], theta, sigma, k, T)

    for i in range(nr):
        for j in range(n-1,0,-1):
            currenttime = j * dt
            
            shiftedtimes = []
            for t_i in times:
                t_i = t_i + currenttime
                shiftedtimes.append(t_i)
          
            EarlyExerciseVal[i, j] = ComputeEuropeanSwaptionPrice(cp, shiftedtimes, coupons, r_val[i, j], theta, sigma, k, T, currenttime)
        
    return EarlyExerciseVal

#computing swap rates at every exercise date to check if swaptions are ITM
def ComputeSwapRates(times, coupons, r_0, theta, sigma, k, t, exercisedates, lifetime):
    #swap rate = present value of floating leg / present value of fixed leg
    swaprates = []
    
    lifetime = float(lifetime)
    for e in exercisedates:
        e = float(e)
        shiftedtimes = []
        for t_i in times:
            t_i = t_i + e
            shiftedtimes.append(t_i)
        
        #PVfixed
        fixedleg = ComputeSwapPV(shiftedtimes, coupons, r_0, theta, sigma, k, e)
        
        #PVfloating
        ZCBstart = ZCB_price(r_0, theta, sigma, k, e, t)[2]
        ZCBend = ZCB_price(r_0, theta, sigma, k, (e + lifetime), t)[2]
        
        floatingleg = ZCBstart - ZCBend
        
        swaprate = floatingleg / fixedleg
        swaprates.append(swaprate)
        
    return swaprates


#Using linear least squares regression to calculate the continuation values for ITM paths
def ComputeLLSContValue(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t, exercisedates, lifetime):

    r_val = VasicekShortRateSimulations(nr, n, r_0, theta, sigma, k, T, t)
    swaprates = ComputeSwapRates(times, coupons, r_0, theta, sigma, k, t, exercisedates, lifetime)
    TargetContVal = ComputeTargetContValues(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t)
    
    LLSContVal = np.zeros((nr,n+1))
    
    for j in range(n-1, 0, -1):
        if cp == 1: #payer
           
            itmpaths = r_val[:, j] > swaprates[j] #payer ITM when short rate > swap rate
            
            xvals = r_val[itmpaths, j]
            yvals = TargetContVal[itmpaths, j]
                        
        elif cp == -1: #receiver
    
            itmpaths = r_val[:, j] < swaprates[j] #rceiver ITM when short rate < swap rate
            
            xvals = r_val[itmpaths, j]
            yvals = TargetContVal[itmpaths, j] 
        
        if len(xvals) > 0: #checking if any ITM values so it only does calculations for ITM paths
            coefficients = np.polyfit(xvals, yvals, 2)
            LLSContVal[itmpaths, j] = np.maximum(0, np.polyval(coefficients, r_val[itmpaths, j]))
            
        else: #if no ITM values
            LLSContVal[:, j] = 0                
            #print("This path is not ITM.")
                 
    return LLSContVal                  


#next step is to compare continuation values and early exercise values
    
def ComputeBermudanSwaptionPrice(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t, exercisedates, lifetime):
    
    dt = (T - t) / n
    
    r_val = VasicekShortRateSimulations(nr, n, r_0, theta, sigma, k, T, t)
    EarlyExerciseVal = ComputeEarlyExerciseValues(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t)
    LLSContVal = ComputeLLSContValue(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t, exercisedates, lifetime)
    
    BermudanOptionVal = np.zeros((nr, n+1))
    
    for i in range(nr):
         for j in range(n-1,0,-1):
             #if early exercise is optimal then that value becomes the option value and all subsequent values on that path are 0
             if EarlyExerciseVal[i,j] > LLSContVal[i,j]:
                 BermudanOptionVal[i,j] = EarlyExerciseVal[i,j]
                 BermudanOptionVal[i,j+1:] = 0 
             
            #continuation if not early exercised
             elif LLSContVal[i,j] >= EarlyExerciseVal[i,j]:
                 BermudanOptionVal[i,j] = LLSContVal[i,j]
         
        #discounting back to t = 0 using simulated short rate
         BermudanOptionVal[i,0] = np.exp(- r_val[i,0] * dt) * BermudanOptionVal[i,1] 
    
    
    price = np.mean(BermudanOptionVal[:,0])
    std = np.std(BermudanOptionVal[:,0])
    
    return price, std


if __name__ == '__main__':
    
    times = [1, 2, 3, 4, 5]
    coupons = [0.02, 0.02, 0.02, 0.02, 1.02]
    r_0 = 0.03
    theta = 0.05
    sigma = 0.01
    k = 0.15
    T = 1
    t = 0
    cp = 1  # payer
    nr = 10
    exercisedates = [1, 2, 3]
    n = len(exercisedates)
    lifetime = 2
    
    price, std = ComputeBermudanSwaptionPrice(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t, exercisedates, lifetime)
    print("Bermudan swaption price:", price, "std:", std)




    