#Code for using the Vasicek model to price swaptions
import numpy as np
from scipy.optimize import newton, brentq

#throughout this code, t is the time at which we are valuing and T is the option expiry

#option expiries {1M, 2M, 3M, 6M, 9M, 1Y, 18M, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y, 15Y, 20Y, 25Y, 30Y}
#swap maturities {1Y, 2Y, 3Y,..., 10Y, 15Y, 20Y, 25Y, 30Y}

def GenerateExerciseDates(first, last, step):
    return np.arange(first, last + 1e-8, step)
 



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
    try:
        rstar = newton(f, r0, fprime)
        
    except(RuntimeError, ValueError):
        try:
            rstar = brentq(f, 1e-6, 1)
        except ValueError:
            return np.nan
     
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
"""def ComputeSwaptionIntrinsicValue(cp, times, coupons, r_0, theta, sigma, k, t):
    
    PV_fixed = ComputeSwapPV(times, coupons, r_0, theta, sigma, k, t)
    PV_float = ComputeSwapPV(times, np.ones_like(coupons), r_0, theta, sigma, k, t)
    
    return max(cp * (PV_fixed - PV_float), 0)"""

def ComputeSwaptionIntrinsicValue(cp, times, coupons, r_0, theta, sigma, k, t):
    
    # Fixed leg PV
    PV_fixed = ComputeSwapPV(times, coupons, r_0, theta, sigma, k, t)

    # Floating leg PV

    mat = times[-1]
    P_end = ZCB_price(r_0, theta, sigma, k, mat, t)[2]
    if t == 0:
        
        PV_float = 1.0
    else:
        PV_float = 1.0 - P_end

    return max(cp * (PV_fixed - PV_float), 0.0)
            

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
#I am using exact discretisation, not Euler scheme

def VasicekShortRateSimulations(nr, r_0, theta, sigma, k, T = 30, t = 0, dt = 0.5):
    
    n = int(T / dt)
    
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

def ComputeTargetContValues(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, rates):
    
    n = int(T / dt)
    
    r_val = rates[:, :n+1]
    
    TargetContVal = np.zeros((nr, n+1))
    
    shiftedtimes = []
    for t_i in times:
        t_i = t_i + T
        shiftedtimes.append(t_i)
    
    for i in range(nr):
        
        #ChatGPT suggesting to use the IntrinsicValue calculations at expiry
        #TargetContVal[i, -1] = ComputeEuropeanSwaptionPrice(cp, shiftedtimes, coupons, r_val[i, -1], theta, sigma, k, T, T)
        TargetContVal[i, -1] = ComputeSwaptionIntrinsicValue(cp, shiftedtimes, coupons, r_val[i, -1], theta, sigma, k, T)
    
    
    #backward discounting for the other steps
    for i in range(nr):
        for j in range(n-1,0,-1):
            TargetContVal[i,j] = np.exp(- r_val[i,j] * dt) * TargetContVal[i,j+1] 
    
    return TargetContVal
           
           

#computing early exercise values at each time point to see if it is optimal at any point
def ComputeEarlyExerciseValues(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, rates):
    
    n = int(T / dt)
    
    r_val = rates[:, :n+1]
    
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
          
            #EarlyExerciseVal[i, j] = ComputeEuropeanSwaptionPrice(cp, shiftedtimes, coupons, r_val[i, j], theta, sigma, k, T, t = currenttime)
            #using EuropeanSwaptionPrice does not work because when T = t it breaks down
            EarlyExerciseVal[i, j] = ComputeSwaptionIntrinsicValue(cp, shiftedtimes, coupons, r_val[i, j], theta, sigma, k, currenttime)

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

#def ComputeLLSContValue(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t, exercisedates, lifetime):
"""def ComputeLLSContValue(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, exercisedates, lifetime, rates):
    #r_val = VasicekShortRateSimulations(nr, n, r_0, theta, sigma, k, T, t)
    n = int(T / dt)
    r_val = rates[:, :n+1]
    
    swaprates = ComputeSwapRates(times, coupons, r_0, theta, sigma, k, t, exercisedates, lifetime)
    TargetContVal = ComputeTargetContValues(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, rates)
    
    LLSContVal = np.zeros((nr,n+1))
    
            
    for j in range(n-1, 0, -1):
        # find the closest exercise date index for this timestep
        if j in exercisedates:
            swap_idx = np.where(exercisedates == j)[0][0]
        else:
            continue  # skip if not an exercise date
        
        if cp == 1:  # payer
            itmpaths = r_val[:, j] > swaprates[swap_idx]
            xvals = r_val[itmpaths, j]
            yvals = TargetContVal[itmpaths, j]
                            
        elif cp == -1:  # receiver
            itmpaths = r_val[:, j] < swaprates[swap_idx]
            xvals = r_val[itmpaths, j]
            yvals = TargetContVal[itmpaths, j] 
            
        if len(xvals) > 0: #checking if any ITM values so it only does calculations for ITM paths
            coefficients = np.polyfit(xvals, yvals, 2)
            LLSContVal[itmpaths, j] = np.maximum(0, np.polyval(coefficients, r_val[itmpaths, j]))
        else: #if no ITM values
            LLSContVal[:, j] = 0
                 
    return LLSContVal        """  

def ComputeLLSContValue(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, exercisedates, lifetime, rates):

    n = int(T / dt)
    r_val = rates[:, :n+1]
    
    # We need early exercise payoffs at exercise indices
    EarlyExerciseVal = ComputeEarlyExerciseValues(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, rates)
    
    # Make sure exercise indices are sorted increasing
    #exercisedates = np.array(exercisedates, dtype=int)
    #exercisedates.sort()
    
    LLSContVal = np.zeros((nr, n+1))
    
    # start with the last early exercise payoff
    j_next = exercisedates[-1]
    V_next = EarlyExerciseVal[:, j_next].copy()
    
    # Go backwards over the remaining exercise dates
    for ii in range(len(exercisedates)-2, -1, -1):
        j = exercisedates[ii]
        
        # Pathwise discount factor from j to j_next
        disc = np.ones(nr)
        if j_next > j:
            for i in range(nr):
                runsum = 0.0
                for u in range(j, j_next):
                    runsum += r_val[i, u] * dt
                disc[i] = np.exp(-runsum)
        
        # Regression target: discounted the payofff back to the previous exercise date
        Y = disc * V_next
        
        X = r_val[:, j]
        payoff_j = EarlyExerciseVal[:, j]  # early exercise at j
        
        # ITM selection (early exercise > 0)
        itmpaths = payoff_j > 0.0
        
        # Continuation estimate using only ITM paths
        if np.any(itmpaths):
            xvals = X[itmpaths]
            yvals = Y[itmpaths]
            # quadratic basis via polyfit 
            coefficients = np.polyfit(xvals, yvals, 2)
            contval = np.polyval(coefficients, X)
            # no negative continuation
            for i in range(nr):
                if contval[i] < 0:
                    contval[i] = 0.0
        else:
            contval = np.zeros(nr)
        
        # store continuation estimates at this exercise index
        for i in range(nr):
            LLSContVal[i, j] = contval[i]
        
        # exercise if payoff > continuation, else continue with Y
        V_curr = np.zeros(nr)
        for i in range(nr):
            if payoff_j[i] > contval[i]:
                V_curr[i] = payoff_j[i]
            else:
                V_curr[i] = Y[i]
        
        # prepare for next backward step
        V_next = V_curr
        j_next = j
    
    return LLSContVal

        

    

#next step is to compare continuation values and early exercise values
    
#def ComputeBermudanSwaptionPrice(cp, nr, n, times, coupons, r_0, theta, sigma, k, T, t, exercisedates, lifetime):
def ComputeBermudanSwaptionPrice(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, exercisedates, lifetime, rates):    
    
    n = int(T / dt)
    
    #r_val = VasicekShortRateSimulations(nr, n, r_0, theta, sigma, k, T, t)
    r_val = rates[:, :n+1]
    EarlyExerciseVal = ComputeEarlyExerciseValues(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, rates)
    LLSContVal = ComputeLLSContValue(cp, nr, times, coupons, r_0, theta, sigma, k, T, t, dt, exercisedates, lifetime, rates)
    
    BermudanOptionVal = np.zeros((nr, n+1))
    
    for i in range(nr):
        # find the optimal exercise index among your exercise dates
        ex_idx = -1
        payoff = 0.0
        for j in exercisedates:  # exercisedates are indices (ascending)
            if EarlyExerciseVal[i, j] > LLSContVal[i, j]:
                ex_idx = j
                payoff = EarlyExerciseVal[i, j]
                break
        # if never exercise early, take continuation at last exercise
        if ex_idx == -1:
            ex_idx = exercisedates[-1]
            payoff = LLSContVal[i, ex_idx]
    
        # discount from 0 -> ex_idx using full path integral
        runsum = 0.0
        for u in range(0, ex_idx):
            runsum += r_val[i, u] * dt
        BermudanOptionVal[i, 0] = np.exp(-runsum) * payoff
    
    
    price = np.mean(BermudanOptionVal[:,0])
    std = np.std(BermudanOptionVal[:,0])
    
    return price, std


#



if __name__ == '__main__':
    # === Vasicek parameters (same as first code) ===
    r_0 = 0.03
    theta = 0.05
    sigma = 0.01
    k = 0.1
    T_max = 7
    dt = 0.1
    nr = 10  # number of simulated paths

    # === Portfolio (same as first code) ===
    portfolio = [
        {'strike': 0.04, 'swap_tenor': 3, 'swap_type': 'payer', 'first_exercise': 2.0, 'last_exercise': 4.0,
         'exercise_freq': 1.0},
        {'strike': 0.035, 'swap_tenor': 2, 'swap_type': 'receiver', 'first_exercise': 1.0, 'last_exercise': 3.0,
         'exercise_freq': 0.5},
        {'strike': 0.05, 'swap_tenor': 4, 'swap_type': 'payer', 'first_exercise': 3.0, 'last_exercise': 5.0,
         'exercise_freq': 1.0}
    ]

    # === Simulate short rates once (same paths for all swaptions) ===
    rates = VasicekShortRateSimulations(nr, r_0, theta, sigma, k, T=T_max, dt=dt)

    results = []
    for i, spec in enumerate(portfolio):
        # Build payment times and coupons for fixed leg
        times = np.arange(dt, spec['swap_tenor'] + dt, 0.5)  # semi-annual payments
        coupons = np.full(len(times), spec['strike'] * 0.5)  # semi-annual coupon rate
        coupons[-1] += 1.0  # add notional at maturity
        
        pay_int = 0.5  # semi-annual payments
        n_pay = int(round(spec['swap_tenor'] / pay_int))


        exercisedates = np.arange(spec['first_exercise'], spec['last_exercise'] + 1e-8, spec['exercise_freq'])
        exercisedates = np.round(exercisedates / dt).astype(int)


        cp = 1 if spec['swap_type'] == 'payer' else -1

        price, std = ComputeBermudanSwaptionPrice(cp, nr, times, coupons, r_0, theta, sigma, k,
                                                  T_max, 0.0, dt, exercisedates, spec['swap_tenor'], rates)

        results.append({
            'Swaption': f'Swaption {i + 1}',
            'Type': spec['swap_type'],
            'Strike': spec['strike'],
            'Tenor': spec['swap_tenor'],
            'FirstExercise': spec['first_exercise'],
            'LastExercise': spec['last_exercise'],
            'Frequency': spec['exercise_freq'],
            'Price': price,
            'StdDev': std
        })

    import pandas as pd
    df_results = pd.DataFrame(results)
    print(df_results)

