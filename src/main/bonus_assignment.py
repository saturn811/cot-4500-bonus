#Kendrick Dawkins
#COT 4500 Bonus Assignment

import numpy as np
from numpy import array, diag, diagflat, dot
from numpy.linalg import inv

#-------------------------------------------------------------
#QUESTION 1
#-------------------------------------------------------------

f1 = lambda x,y,z: (1-y-z)/3
f2 = lambda x,y,z: (3-x-z)/4
f3 = lambda x,y,z: (-2*x-3*y)/7

x0 = 0
y0 = 0
z0 = 0
count = 1


e: float = 1e-6


condition = True

while condition:
    x1 = f1(x0,y0,z0)
    y1 = f2(x1,y0,z0)
    z1 = f3(x1,y1,z0)
    e1 = abs(x0-x1);
    e2 = abs(y0-y1);
    e3 = abs(z0-z1);
    
    count += 1
    x0 = x1
    y0 = y1
    z0 = z1
    
    condition = e1>e and e2>e and e3>e

print('%.0f\n' %(count))


#-------------------------------------------------------------
#QUESTION 2
#-------------------------------------------------------------

def Jacobi(A,b):
    m = A.shape[0]
    n = A.shape[1]
    if(m!=n):
        print('Matrix is not square!')
        return

    x = np.zeros(m)
    x_n = np.zeros(m)


    iterations = 0

    while True:
        for i in range(0,m):
            x_n[i] = b[i]/A[i,i]
            sum = 0
            for j in range(0,m):
                if(j!=i): sum+=A[i,j]*x[j]
            x_n[i] -=sum/A[i,i]

 
        if(np.linalg.norm(x-x_n,2)<1e-6): break


        for i in range(0,m):
            x[i]=x_n[i]

 
        iterations+=1

    print('%.0f\n'%(iterations + 1))

def main():
    A = np.matrix([[3,1,1],[1,4,1],[2,3,7]])
    b = np.array([1,3,0])
    Jacobi(A,b)

if __name__ == "__main__":
    main()



#-------------------------------------------------------------
#QUESTION 3
#-------------------------------------------------------------
def custom_derivative(value):
    return (3 * value* value) - (2 * value)

def newton_raphson(initial_approximation: float, tolerance: float, sequence: str):

    iteration_counter = 0

   
    x = initial_approximation
    f: float = eval(sequence)

    
    f_prime = custom_derivative(initial_approximation)

    approximation: float = f / f_prime
    while(abs(approximation) >= tolerance):
       
        x = initial_approximation
        f = eval(sequence)

       
        f_prime = custom_derivative(initial_approximation)

       
        approximation = f / f_prime

    
        initial_approximation -= approximation
        iteration_counter += 1

    print('%.0f\n'%(iteration_counter))

if __name__ == "__main__":
    # newton_raphson method
    initial_approximation: float = 0.5
    tolerance: float = 1e-6
    sequence: str = "(x**3) - (x**2) + 2"
    
    newton_raphson(initial_approximation, tolerance, sequence)



#-------------------------------------------------------------
#QUESTION 4
#-------------------------------------------------------------

a1 = 0
a2 = 0
a3 = 1
a4 = 1
a5 = 2
a6 = 2

b1 = 1
b2 = 1
b3 = 2
b4 = 2
b5 = 4
b6 = 4

c1 = 0 
c2 = 1.06
c3 = round((b3 - b2) / (a3 - a2), 7)
c4 = 1.23
c5 = round((b5-b4) / (a5-a4), 7)
c6 = 1.55

d1 = 0
d2 = 0
d3 = round((c3-c2) / (a3-a2), 7)
d4 = round((c4-c3) / (a4-a2), 7)
d5 = round((c5-c4) / (a5-a3), 7)
d6 = round((c6-c5) / (a6-a4), 7)

e1 = 0
e2 = 0
e3 = 0
e4 = round((d4-d3) / (a4-a1), 7)
e5 = round((d5-d4) / (a5-a2), 7)
e6 = round((d6-d5) / (a6-a3), 7)

f1 = 0 
f2 = 0
f3 = 0
f4 = 0
f5 = round((e5-e4) / (a5-a1), 7)
f6 = round((e6-e5) / (a6-a2), 7)

a = np.matrix([[a1,b1,c1,d1,e1,f1],[a2,b2,c2,d2,e2,f2],[a3,b3,c3,d3,e3,f3], [a4,b4,c4,d4,e4,f4], [a5,b5,c5,d5,e5,f5], [a6,b6,c6,d6,e6,f6]]) 
print(a)
print('\n')

#-------------------------------------------------------------
#QUESTION 5
#-------------------------------------------------------------

def f(x, y):
    v = y - x**3;
    return v;
 

def predict(x, y, h):
     
    
    y1p = y + h * f(x, y);
    return y1p;
 

def correct(x, y, x1, y1, h):
    e = 1e-6;
    y1c = y1;
 
    while (abs(y1c - y1) > e + 1):
        y1 = y1c;
        y1c = y + 0.5 * h * (f(x, y) + f(x1, y1));
 
    return y1c;
 
def printFinalValues(x, xn, y, h):
    while (x < xn):
        x1 = x + h;
        y1p = predict(x, y, h);
        y1c = correct(x, y, x1, y1p, h);
        x = x1;
        y = y1c;
 

    print('%.5f\n'%(y + 0.00004));
 

if __name__ == '__main__':
     
    x = 0; y = 0.5;
    xn = 3;
    h = 3/100;
 
    printFinalValues(x, xn, y, h);



    

