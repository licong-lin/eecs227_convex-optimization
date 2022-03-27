import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt



np.random.seed(10)
A=np.random.normal(size=(20,20))
np.random.seed()

def f(x):
    x=x.reshape(-1,1)
    return np.sum((A@x)**2)

B=np.zeros((20,20))
for i in range(20):
    if i==0:
        B[i,0]=2
        B[i,1]=-1
    elif i<19:
        B[i,i-1]=-1
        B[i,i]=2
        B[i,i+1]=-1
    elif i==19:
        B[i,i-1]=-1
        B[i,i]=2


def f2(x):
    x=x.reshape(-1,1)
    return x.T@B@x





x=np.ones(20).reshape(-1,1)

def gd(f,x_0,num,eta):
    grad_f=grad(f)
    x_c=x_0
    record_optimal=np.zeros(num+1)
    record_optimal[0]=f(x_0)
    for t in range(1,num+1):
        g=grad_f(x_c)
        x_c=x_c-eta*g
        record_optimal[t]=np.min([record_optimal[t-1],f(x_c)])
    return record_optimal

def agd(f,x_0,num,eta): ## when R(x)=x^2/2
    grad_f=grad(f)
    x_c1=x_0
    x_c2=x_0
    record_optimal=np.zeros(num+1)
    record_optimal[0]=f(x_0)
    for t in range(1,num+1):
        y_c=x_c2+t/(t+3)*(x_c2-x_c1)
        g=grad_f(y_c)
        x_c1=x_c2
        x_c2=y_c-eta*g
        record_optimal[t]=np.min([record_optimal[t-1],f(x_c2)])
    return record_optimal

num=10000


'''
record=gd(f,x,num,eta=0.001)
record2=agd(f,x,num,eta=0.001)

print(record)
plt.plot(np.log(np.arange(num)),np.log(record[0:(num)]),label='gd')

plt.plot(np.log(np.arange(num)),np.log(record2[0:(num)]),label='agd')
plt.legend()
plt.title(r'$\|Ax\|^2$')
plt.ylabel('suboptimality')
plt.xlabel('iterations')
plt.show()
'''

record=gd(f2,x,num,eta=.05)
record2=agd(f2,x,num,eta=.05)

print(record2)
plt.plot(np.log(np.arange(num)),np.log(record[0:(num)]),label='gd')

plt.plot(np.log(np.arange(num)),np.log(record2[0:(num)]),label='agd')
plt.legend()
plt.title(r'$x^\top Ax$')
plt.ylabel('suboptimality')
plt.xlabel('iterations')
plt.show()










