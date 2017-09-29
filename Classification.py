import numpy as np
import matplotlib.pyplot as plt
def nonlin(x,diff="False"):
    if(diff=="True"):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

x=np.array([[0,1,1,0],
            [1,0,0,1],
            [1,0,1,0],
            [0,1,0,1]])
y=[[1,0],
   [0,1],
   [0,1],
   [1,0]]

np.random.seed(2)

syn0=2*np.random.random((4,8))-1
syn1=2*np.random.random((8,10))-1
syn2=2*np.random.random((10,2))-1

for iter in range(10000):
    l0=x
    l1=nonlin(np.dot(l0,syn0))
    l2=nonlin(np.dot(l1,syn1))
    l3=nonlin(np.dot(l2,syn2))
    l3_error=y-l3
    l3_delta=l3_error*nonlin(l3,True)
    l2_error=np.dot(l3_delta,syn2.T)
    l2_delta=l2_error*nonlin(l2,True)
    l1_error=np.dot(l2_delta,syn1.T)
    l1_delta=l1_error*nonlin(l1,True)
    syn0+=np.dot(l0.T,l1_delta)
    syn1+=np.dot(l1.T,l2_delta)
    syn2+=np.dot(l2.T,l3_delta)

print("Output of the training is:\n")
print(l3)

i=0
a=[]
while i<4:
    b=int(input("Enter the input:\n"))
    a.append(b)
    i=i+1
c=np.array(a)
ol0=c
ol1=nonlin(np.dot(c,syn0))
ol2=nonlin(np.dot(ol1,syn1))
ol3=nonlin(np.dot(ol2,syn2))
if(ol3[0]>ol3[1]):
    print("\nSame\n")
else:
    print("\nDiff\n")
