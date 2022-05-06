import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt



def f(x,eps):
    v= -np.sum(np.log(1-x**2))+np.sum((1+np.arange(10))*x)/eps
    return v
def df(x,eps):
    v= (np.arange(10)+1)/eps+ 2*x/(1-x**2)
    return v
def d2f(x,eps):
    return (2+2*x**2)/(1-x**2)**2

def lam(x,eps):
    return (np.sum(df(x,eps)**2/d2f(x,eps)))**0.5


Mf=1


x_0=np.zeros(10)


ii=0
max_len=(0,0)

all_record=[0,0,0,0]


for eps in [1,0.1,0.01,0.005]:
    x_cur=x_0

    record=np.array([lam(x_cur,eps)])

    while True:
        x_cur=x_cur-df(x_cur,eps)/d2f(x_cur,eps)/(1+Mf*lam(x_cur,eps))
        v=np.array([lam(x_cur,eps)])
        
        record=np.concatenate((record,v),0)
        if v<10**(-6):
            break
    if len(record)>max_len[0]:
        max_len=(len(record),ii)
    all_record[ii]=record
    ii+=1


dic={1:1,2:0.1,3:0.01,4:0.005}

for i in range(4):
    if i!=max_len[1]:

        all_record[i]=np.concatenate((all_record[i],all_record[i][-1]*
            np.ones(max_len[0]-len(all_record[i]))),0)


    plt.plot(np.arange(max_len[0]),all_record[i],label=r'$\epsilon=$'+str(dic[i+1]))

plt.legend()
plt.ylabel(r'$\lambda_f(x)$')
plt.xlabel('steps')
plt.show()











