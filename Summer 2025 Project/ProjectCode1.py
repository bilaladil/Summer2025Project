#I started with the Monte Carlo Pricing Code from ACM30110 Tutorial 6 
#We notice that this code only works for European options so we need to change it to work for
#options where early exercise is possible. To do this we first need to start recording option prices at all
#time points. For simplicity I change the time steps to 3. (We know that with 25 steps the European price is 5.83775).
#I also changed the simulations to 10 because I'm doing a first test.

import math as m
import numpy as np
import matplotlib.pyplot as plt

cp = -1 # +1/-1 for Call/Put
S = 60
K = 65
r = 0.08
T = 0.25
sigma = 0.3

# n DEFINES THE TIME STEP

n = 3

# nr is the number of simulations

nr = 10

# Here we refer to the risk-neutral valuation.

nu = r - 0.5 * sigma**2
dt = T / n # <- h


S_val = np.zeros((nr, n+1))
rand = np.random.randn(nr, n)


S_val[:,0] = S
for i in range(nr):
    for j in range(1,n+1):
        S_val[i,j] = S_val[i,j-1] * m.exp(nu*dt + sigma * dt**0.5 * rand[i,j-1])



# instead of discounting all the way back we want to make a matrix that contains the values at all time points

TargetContVal = np.zeros((nr,n+1))
TargetContVal[:,-1] = np.maximum(cp*(S_val[:,-1] - K), 0)

# we store the early exercise values separately because we want to use the TargetContVal matrix as our target values
# when we do our regression

EarlyExerciseVal = np.zeros((nr,n+1))
EarlyExerciseVal[:,-1] = np.maximum(cp*(S_val[:,-1] - K), 0)

for i in range(nr):
    for j in range(n-1,0,-1):
        if cp == -1:
            TargetContVal[i,j] = np.exp(-r * dt) * TargetContVal[i,j+1]
            EarlyExerciseVal[i,j] = np.maximum(cp*(S_val[i,j]-K),0)
                
        elif cp == 1: 
            TargetContVal[i,j] = np.exp(-r * dt) * TargetContVal[i,j+1]
            EarlyExerciseVal[i,j] = np.maximum(cp*(S_val[i,j]-K),0)

# This seems easy and thats because for the LSM method we need to apply some regression knowledge. 

LLSContVal = np.zeros((nr,n+1))
xvals = []
yvals = []

for j in range(n-1,0,-1):
    if cp == -1:
        itmpaths = S_val[:,j] < K
    
        xvals = S_val[itmpaths,j]
        yvals = TargetContVal[itmpaths,j]
        
        coefficients = np.polyfit(xvals,yvals,2)
  
    
    if cp == 1:
        itmpaths = S_val[:,j] > K
       
        xvals = S_val[itmpaths,j]
        yvals = TargetContVal[itmpaths,j]
        
        coefficients = np.polyfit(xvals,yvals,2)
    
    #LLSContVal[itmpaths,j] = np.polyval(coefficients, S_val[itmpaths,j])
           
# I noticed that if the predicted share value is out of the money, then the preceding target option
# value is 0 but then this leads to a negative LLS continuation value. I noticed that its not
# explicitly mentioned in Longstaff and Schwartz (2001) so I asked chatGPT and it said this is a known
# problem so I implement a way to make sure that we don't get negative continuation values.
                
    LLSContVal[itmpaths,j] = np.maximum( 0, np.polyval(coefficients, S_val[itmpaths,j]) )

# A more complex and widely used solution is to use non negative polynomials but at the moment
# I am just trying to gain a basic understanding so I use this temporary solution.     
 
# Now we need to compare the LLS continuation values to the exercise values and store in a new matrix

AOptVal = np.zeros((nr,n+1)) 
for i in range(nr):
    for j in range(n-1,0,-1):
        AOptVal[i,j] = np.maximum(LLSContVal[i,j],EarlyExerciseVal[i,j])
    
    AOptVal[i,0] = np.exp(-r * dt) * AOptVal[i,1]
    
            
# Yes this works BUT if early exercise occurs then the subsequent values should be 0 because
# if the option is exercised then there are no following values 

EarlyExerciseTracker = np.zeros((nr,n+1))
for i in range(nr):
    for j in range(1,n):
        if EarlyExerciseVal[i,j] > LLSContVal[i,j]:
            EarlyExerciseTracker[i,j] = 1
            EarlyExerciseTracker[i,j+1:] = 0

for i in range(nr):
    for j in range(1,n):
         if EarlyExerciseTracker[i,j] == 1:
             EarlyExerciseTracker[i,j+1:] = 0
             AOptVal[i,j+1:] = 0        

price = np.mean(AOptVal[:,0])
std = np.std(AOptVal[:,0])


plt.figure()
for i in range(nr):
    plt.plot(S_val[i,:])
plt.title('Monte Carlo Method simulations - GBM Paths')
    
print('The price of the Monte Carlo Method with', nr, 'simulations is %.5f.' % price)
print('The standard deviation is %.5f.' % std)

