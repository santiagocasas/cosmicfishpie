import numpy as np
from numpy.polynomial import legendre as lg
#from sympy.physics.wigner import wigner_3j
#import warnings

#To approximate the integral by a discrete sum 

def gauss_lobatto_abscissa_and_weights(order):
    lp = np.zeros(order, dtype='float')
    roots = np.zeros(order, dtype='float')
    lp[order-1] = 1
    der = lg.legder(lp)
    roots[0] =  0
    roots[-1]  = 1
    roots[1:-1] = lg.legroots(der)
    weights = 2./(order * (order-1)* lg.legval(roots, lp)**2)
    return roots, weights

############################################################
#Wigner Symbols calculated using the following definition
#You need to import this package
#from sympy.physics.wigner import wigner_3j
#Here, i = l1 and j = l2
#The values hardcoded in the __init__ have been calculated using this function

###

# def sum3ji(l1,l2,l3,l4):
#     sum = 0.0
#     l = min(abs(l1-l2),abs(l3-l4)) 
#     l_fin = max(l1+l2,l3+l4)
#     for i in range(l,l_fin+1):
#        sum += wigner_3j(l1,l2,i,0,0,0)**2 *wigner_3j(l3,l4,i,0,0,0)**2 * (2*i + 1)
#     return sum

#l = []
#Let l1 be 0,2,4
#for i in range(0,5,2):
#Let l2 be 0,2,4
#    for j in range(0,5,2):
#        a = sum3ji(i,j,l3,l4)   #-> you just have to specify your choice of l3,l4 here.
#        l.append(a)
#print(l)

###
#Using the above formulae, we get the following 3x3 arrays 
#with the correct values, having the shape 3x3 (len(l1) x len(l2))
#The first line for each case is the list outputted by the 'for i in range...' code
#This calculates the (len(l1) x len(l2)) for a set of l3,l4 using the sum3ji formula
#We then transform it into an array that is then reshaped into a 3x3 matrix to be used by the code

#l3 = l4 = 0
m00l = [1.00000000000000, 0, 0, 0, 0.200000000000000, 0, 0, 0, 0.111111111111111]
m00a = np.array(m00l)
m00  = m00a.reshape(3,3)

#l3 = 2 l4 = 2
m22l = [0.200000000000000, 2/35, 2/35, 2/35, 0.0857142857142857, 12/385, 2/35, 12/385, 0.0397158397158397]
m22a = np.array(m22l)
m22  = m22a.reshape(3,3)

#l3 = 4 l4 = 4
m44l = [0.111111111111111, 20/693, 18/1001, 20/693, 0.0397158397158397, 20/1001, 18/1001, 20/1001, 0.0310865604983252]
m44a = np.array(m44l)
m44  = m44a.reshape(3,3)

#l3 = 0 l4 = 2
m02l = [0, 0.200000000000000, 0, 0.200000000000000, 2/35, 0.0571428571428571, 0, 0.0571428571428571, 20/693]
m02a = np.array(m02l)
m02  = m02a.reshape(3,3)

#l3 = 2 l4 = 4
m24l = [0, 0.0571428571428571, 20/693, 0.0571428571428571, 12/385, 0.0397158397158397, 20/693, 0.0397158397158397, 20/1001]
m24a = np.array(m24l)
m24  = m24a.reshape(3,3)

#l3 = 0 l4 = 4
m04l = [0, 0, 0.111111111111111, 0, 2/35, 20/693, 0.111111111111111, 20/693, 18/1001]
m04a = np.array(m04l)
m04  = m04a.reshape(3,3)