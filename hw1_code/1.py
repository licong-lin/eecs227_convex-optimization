import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt


def A(k,i,j):

    if i<j:
        a=np.exp(i/j)*np.cos(i*j)*np.sin(k)
    elif i==j:
        a=i/10*np.abs(np.sin(k))
        for ii in range(1,11):
            if ii!=i:
                a=a+np.abs(A(k,i,ii))
    else:
        a=A(k,j,i)

    return a

def mA(k):   ## matrix A
    A0=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            A0[i,j]=A(k,i+1,j+1)
    return A0
def b(k):
    B=np.arange(1,11).reshape(-1,1)
    B=np.exp(B/k)*np.sin(B*k)
    return B

def one_f(k,x):
    x=x.reshape(-1,1)
    return x.T@mA(k)@x-b(k).T@x

def f(x):
    s=one_f(1,x)
    index=1
    for k in range(2,6):
        s1=one_f(k,x)
        if s1>s:
            index=k
            s=s1
    return s

grad_f=grad(f)
fstar=-0.707

def sub_gd(x_0,num,C=1):
    x_c=x_0
    record_optimal=np.zeros(num+1)
    record_optimal[0]=f(x_0)
    for t in range(1,num+1):
        eta=C/t**0.5
        g=grad_f(x_c)
        x_c=x_c-eta*g/(np.sum(g**2))**0.5
        record_optimal[t]=np.min([record_optimal[t-1],f(x_c)])
    return record_optimal


def sub_gd_polyak(x_0,num):
    x_c=x_0
    record_optimal=np.zeros(num+1)
    record_optimal[0]=f(x_0)
    for t in range(1,num+1):
        g=grad_f(x_c)
        g_norm=(np.sum(g**2))**0.5
        eta=(f(x_c)-fstar)/g_norm
        x_c=x_c-eta*g/g_norm
        record_optimal[t]=np.min([record_optimal[t-1],f(x_c)])
    return record_optimal





x=np.ones(10).reshape(-1,1)
#f(x)=5337.0


num=200





record=sub_gd(x,num,C=1)
record2=sub_gd_polyak(x,num)

print(record)
plt.plot(np.log(np.arange(num+1)),np.log(record-record[num]),label='root_t')

plt.plot(np.log(np.arange(num+1)),np.log(record2-record2[num]),label='polyak')
plt.legend()
plt.ylabel('suboptimality')
plt.xlabel('iterations')
plt.show()






