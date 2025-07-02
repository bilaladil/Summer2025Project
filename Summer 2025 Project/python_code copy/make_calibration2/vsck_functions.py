#Code for using the Vasicek model to price swaptions
import numpy as np
import scipy 



def ZCB_price(r, theta, sigma, k, T, t):
    
    B = (1 / k)*(1 - np.exp((-k) * (T - t)))
    A = np.exp((theta - ((sigma ** 2) / (2 * (k ** 2))) ) * (B - T + t) - ((sigma ** 2) / (4 * k)) * (B ** 2))
    
    P = A * np.exp(- B * r)
    return A, B, P

#we want to use Jamshidians trick to find rstar when sum of ci * ZCB price = K

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


def EuropeanOptionPrice(cp, coupons, ZCBprices, strikeprices):
    
    if len(coupons) == len(ZCBprices) and len(coupons) == len(strikeprices):
        
        n = len(coupons)    
        runningsum = 0
        
        if cp == 1: #call
            for i in range(0,n):
                runningsum += coupons[i] * max(ZCBprices[i] - strikeprices[i] , 0)
        elif cp == -1: #put
            for i in range(0,n):
                runningsum += coupons[i] * max(strikeprices[i] - ZCBprices[i] , 0)
         
        price = runningsum
        return price
    
    else:
        print("Array sizes do not match.")


#this means that the following code will only run in this file
if __name__ == '__main__':
    
#now we need to calculate the price of each zero coupon bond today
#so we need to assume we have a dataset containing coupon payments and dates
#we need to calculate K for each zero coupon bond    

    times = [1.5, 2.0, 2.5, 3.0]
    coupons = [0.025, 0.025, 0.025, 1.025]    
    
    cp = -1
    k = 0.1
    T = 1.0
    t = 0
    sigma = 0.01
    theta = 0.05
    r_0 = 0.03
    r = r_0


    ZCBprices = []
    for T_i in times:
        A, B, ZCB = ZCB_price(r_0, theta, sigma, k, T_i, 0)
        ZCBprices.append(ZCB)


    if len(times) == len(coupons):
        
        n = len(times)
        Avalues = []
        Bvalues = []

        swaptionK = 0
        for i in range(0, n):
            
            Aval, Bval, Pval = ZCB_price(r, theta, sigma, k, times[i], t) 
            Avalues.append(Aval)
            Bvalues.append(Bval)
            
            swaptionK += coupons[i] * ZCBprices[i]
    
    
    def f1(r):
        return f(r, coupons, Avalues, Bvalues, swaptionK)
    
    def fprime1(r):
        return fprime(r, coupons, Avalues, Bvalues)
    
    rstar = JamshidiansTrick(f1, r_0, fprime1)
    
    strikeprices = []
    for T_i in times:
        A, B, K = ZCB_price(rstar, theta, sigma, k, T_i, T)
        strikeprices.append(K)

    price = EuropeanOptionPrice(cp, coupons, ZCBprices, strikeprices)
    print(price)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    