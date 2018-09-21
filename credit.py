import numpy as np
import random
from math import exp,log10
import matplotlib.pyplot as plt	

def create_input_matrix(filename):
	ifile = open(filename)
	a=np.array([[0,0,0]])
	y=np.array([0])
	for line in ifile:
		token=line.split(',')
		a=np.append(a,[[1,float(token[0]),float(token[1])]],axis=0)
		y=np.append(y,[int(token[2])],axis=0)
	a=np.delete(a,0,0)
	y=np.delete(y,0,0)
	y=np.array(y)[:,np.newaxis]
	return (a,y)

def sigmoid(w,x):
	f=np.matmul(x,w)
	(n,c)=f.shape
	for i in range(n):
		if (f[i]>100):
			f[i]=100
		elif(f[i]<-100):
			f[i]=-100
	f= (1/(1+np.exp(-f)))
	return f


def output(f):
	if(f<0.5):
		return 0
	else:
		return 1

def error(w,x,y,lamda):
	(n,d)=x.shape
	err=0
	f=sigmoid(w,x)
	for i in range(n):
		if (abs(f[i]-1.0)<0.000001):
			f[i]=0.999999
		if (abs(f[i]-0.0)<0.000001):
			f[i]=0.0000001

		err+=(y[i]*log10(f[i])+(1-y[i])*log10(1-f[i]))
	err=-err
	err+=lamda*(np.matmul(w.transpose(),w))[0]	
	return err	
	

def accuracy(w,x,y):
	(n,d)=x.shape
	count=0
	f=sigmoid(w,x)
	for i in range(n):
		o=output(f[i])
		if(o==y[i]):
			count+=1
	return float(count*100.0/n)	

def gradient_descent(w,x,y,alpha,epsilon,lamda,iterations):
	(n,d)=x.shape
	err=error(w,x,y,lamda)
	preverr=0
	itr=0
	while(err>epsilon and abs(err-preverr)>epsilon and itr<iterations):
		f=sigmoid(w,x)
		w=w-alpha*((np.matmul((f-y).transpose(),x)).transpose()+2*lamda*w)
		preverr=err
		err=error(w,x,y,lamda)
		itr+=1
		# print (preverr)
		# print (err)
	return w	

def create_r(w,x):
	(n,d)=x.shape
	r=np.identity(n)
	f=sigmoid(w,x)
	for i in range(n):
		r[i][i]=f[i]*(1-f[i])
	return r

def newton_raphson(w,x,y,epsilon,lamda,iterations):
	(n,d)=x.shape
	preverr=0
	err=error(w,x,y,lamda)
	itr=0
	while(err>epsilon and abs(preverr-err)>epsilon and itr<iterations):
		R=create_r(w,x)
		f=sigmoid(w,x)
		w=w-np.matmul(np.matmul(np.linalg.inv(np.matmul(np.matmul(x.transpose(),R),x)+lamda*np.identity(d)),x.transpose()),(f-y)) - 2*lamda*w
		preverr=err
		err=error(w,x,y,lamda)
		itr+=1
		# print (preverr)
		# print (err)	
	return w	

def create_dataset(x,p):
	(n,d)=x.shape
	t=np.array([0])
	for i in range(1,int(((p+1)*(p+2))/2)):
		t=np.append(t,0)
	X=np.array([t])
	for k in range(n):
		temp=np.array([])
		for i in range(p+1):
			for j in range(p+1-i):
				temp=np.append(temp,(x[k][1]**i)*(x[k][2]**j))
		X=np.append(X,[temp],axis=0)
	X=np.delete(X,0,0)
	w=np.array(t)[:,np.newaxis]
	return (X,w)			

def plot_dataset(x,y):
	(n,d)=x.shape
	m1={}
	m2={}
	for i in range(d):
		m1[i]=[]
		m2[i]=[]
		for j in range(n):
			if(y[j]==1):
				m1[i].append(x[j][i])
			elif(y[j]==0):
				m2[i].append(x[j][i])
	# print (m1[0])
	# print (m1[1])
	plt.scatter(m1[1], m1[2], label= "issued", color= "green", marker= "o", s=30)
	plt.scatter(m2[1], m2[2], label= "denied", color= "red", marker= "^", s=30)  
	plt.title('Dataset Plot!!') 
	plt.legend()  
	plt.show() 			



if __name__=="__main__":
	(x,y) = create_input_matrix("l2/credit.txt")
	inst=4
	if(inst==1):
		plot_dataset(x,y)
	elif (inst==2):
		w=np.array([[0],[0],[0]])
		lamda=0.4
		epsilon=0.00001
		alpha=0.0001
		iterations=25000
		wg = gradient_descent(w,x,y,alpha,epsilon,lamda,iterations)
		wn = newton_raphson(w,x,y,epsilon,lamda,iterations)
		print ('accuracy with gradient descent = '+str(accuracy(wg,x,y)))
		print ('accuracy with newton-raphson = '+str(accuracy(wn,x,y)))
	elif(inst==4):
		lamda=0.0
		epsilon=0.0001
		alpha=0.000001
		iterations=2500
		(X,W)=create_dataset(x,6)
		wg=gradient_descent(W,X,y,alpha,epsilon,lamda,iterations)
		wn=newton_raphson(W,X,y,epsilon,lamda,iterations)
		print ('accuracy with gradient descent = '+str(accuracy(wg,X,y)))
		print ('accuracy with newton-raphson = '+str(accuracy(wn,X,y)))

 