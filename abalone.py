import numpy as np
import random
from math import exp,log
import pickle
import matplotlib.pyplot as plt	
def create_param_matrix_gender(filename):
	ifile = open(filename)
	a=np.array([[0,0,0]])
	for line in ifile:
		token=line.split(',')
		if token[0]=='M':
			a=np.append(a,[[0,0,1]],axis=0)
		elif token[0]=='I':
			a=np.append(a,[[0,1,0]],axis=0)
		elif token[0]=='F':
			a=np.append(a,[[1,0,0]],axis=0)
	a=np.delete(a,0,0)
	return a

def create_param_matrix(filename):
	ifile = open(filename)
	a=np.array([[0,0,0,0,0,0,0,0,0,0,0]])
	y=np.array([0]);
	for line in ifile:
		token=line.split(',')
		if token[0]=='M':
			a=np.append(a,[[1,0,0,1,float(token[1]),float(token[2]),float(token[3]),float(token[4]),float(token[5]),float(token[6]),float(token[7])]],axis=0)
		elif token[0]=='I':
			a=np.append(a,[[1,0,1,0,float(token[1]),float(token[2]),float(token[3]),float(token[4]),float(token[5]),float(token[6]),float(token[7])]],axis=0)
		elif token[0]=='F':
			a=np.append(a,[[1,1,0,0,float(token[1]),float(token[2]),float(token[3]),float(token[4]),float(token[5]),float(token[6]),float(token[7])]],axis=0)
		y=np.append(y,[int(token[8])],axis=0)
	a=np.delete(a,0,0)

	y=np.delete(y,0,0)
	y=np.array(y)[:,np.newaxis]
	return (a,y)

def data_partition(a,y,p):
	(r,c)=a.shape
	n=int(r*p/100)
	rand_list=random.sample(range(0,r),n)
	train=np.array([a[rand_list[0]]])
	trainy=np.array([y[rand_list[0]]])
	for i in range(1,len(rand_list)):
		train=np.append(train,[a[rand_list[i]]],axis=0)
		trainy=np.append(trainy,[y[rand_list[i]]],axis=0)
	x=[z for z in range(0,r) if z not in rand_list]
	if len(x)>0:
		test=np.array([a[x[0]]])
		testy=np.array([y[x[0]]])
		for i in range(1,len(x)):
			test=np.append(test,[a[x[i]]],axis=0)
			testy=np.append(testy,[y[x[i]]],axis=0)
	else:
		test=np.array([0])
		testy=np.array([0])
	return (train,trainy,test,testy)


def standardize(a):
	mean=np.mean(a,axis=0)
	std=np.std(a,axis=0)
	(r,c)=a.shape
	for i in range(0,r):
		for j in range (1,c):
			a[i][j]=(a[i][j]-mean[j])/std[j]
	return a

def mylinridgereg(x,y,lamda):
	(r,c)=x.shape
	w=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)+lamda*np.identity(c)),x.transpose()),y)
	return w


def meansquarederr(w,x,y):
	f=np.matmul(x,w)
	(r,c)=x.shape
	return (0.5*np.matmul((f-y).transpose(),(f-y)))[0]/r




def load(filename):
	file = open(filename,'rb')
	obj=pickle.load(file)
	file.close()
	return obj

def save(obj,filename):
	file=open(filename,'wb')
	pickle.dump(obj,file)
	file.close()


if __name__=="__main__":
	inst=7
	if (inst==1):
		print (create_param_matrix_gender("l2/linregdata"))
	elif (inst==2):		
		(data,y) = create_param_matrix("l2/linregdata")
		data=standardize(data)
		save(y,'y.pkl')
		save(data,'data.pkl')
	elif(inst==4):
		(data,y) = create_param_matrix("l2/linregdata")
		data=standardize(data)		
		(traindata,trainy,testdata,testy)=data_partition(data,y,80)
		save(traindata,'traindata.pkl')
		save(trainy,'trainy.pkl')
		save(testdata,'testdata.pkl')
		save(testy,'testy.pkl')
		f=0.2
		(traindata,trainy,validationdata,validationy)=data_partition(traindata,trainy,f)
		lamda=0.8
		w=mylinridgereg(traindata,trainy,lamda)
		print (w)
	# elif(inst==6):
	# 	data=load('traindata.pkl')
	# 	y=load('trainy.pkl')
	# 	testdata=load('testdata.pkl')
	# 	testy=load('testy.pkl')
	# 	fi=0.2
	# 	lamdainitial=0.0001	
	# 	lamdadiff=0.1
	# 	lamdano=500
	# 	fractionno=100


	# 	errFrac=[]			#list of errors of a particular fraction on training
	# 	errFracTest=[]		#list of errors of a particular fraction on test

	# 	f=fi
	# 	for i in range(fractionno):
	# 		lamda=lamdainitial
	# 		errLamda=[]				#list of errors of a particular lamda on training
	# 		errLamdaTest=[]			#list of errors of a particular lamda on test
	# 		(traindata,trainy,validationdata,validationy)=data_partition(data,y,f)
	# 		f+=(1.0-fi)/(fractionno)
	# 		for j in range(lamdano):
	# 			#print (str(f)+" "+str(lamda))
	# 			w=mylinridgereg(traindata,trainy,lamda)
	# 			errLamda.append(meansquarederr(w,traindata,trainy))
	# 			errLamdaTest.append(meansquarederr(w,testdata,testy))
	# 			lamda+=lamdadiff
	# 		errFrac.append(errLamda)
	# 		errFracTest.append(errLamdaTest)
	# 	l=[]
	# 	for i in range(lamdano):
	# 		l.append(lamdainitial+i*lamdadiff)	
	# 	for i in range(len(errFrac)): 
	# 		x=errFrac[i]
	# 		plt.plot(l,x) 
			
	# 	plt.xlabel('lamda values') 
	# 	plt.ylabel('average mean squareerror') 
	# 	plt.title('error vs lamda for frac = '+str(fi+i*(1.0-fi)/fractionno)) 
	# 	plt.show()


	# 	lamdaminerr=[]
	# 	valueminerr=[]
	# 	fractions=[]
	# 	for i in range(fractionno):
	# 		fractions.append(fi+i*(1.0-fi)/fractionno)
	# 		index=0;
	# 		minimum=errFrac[i][0]
	# 		for j in range(lamdano):
	# 			if (errFrac[i][j]<minimum):
	# 				print (minimum)
	# 				minimum=errFrac[i][j]
	# 				index=j
	# 		lamdaminerr.append(lamdainitial + index*lamdadiff)
	# 		valueminerr.append(minimum)

	# 	plt.plot(fractions,lamdaminerr) 
	# 	plt.xlabel('frac values') 
	# 	plt.ylabel('lamda values fro minimum err') 
	# 	plt.title('lamda vs frac ') 
	# 	plt.show()

	# 	plt.plot(fractions,valueminerr) 
	# 	plt.xlabel('frac values') 
	# 	plt.ylabel('error values fro minimum err') 
	# 	plt.title('min err vs frac ') 
	# 	plt.show()
	elif(inst==7):
		data=load('traindata.pkl')
		y=load('trainy.pkl')
		testdata=load('testdata.pkl')
		testy=load('testy.pkl')
		fi=0.1
		lamdainitial=0.01 	
		lamdadiff=0.1
		lamdano=50
		fractionno=5


		errFrac=[]			#list of errors of a particular fraction on training
		errFracTest=[]		#list of errors of a particular fraction on test

		f=fi
		for i in range(fractionno):
			lamda=lamdainitial
			errLamda=[]				#list of errors of a particular lamda on training
			errLamdaTest=[]			#list of errors of a particular lamda on test
			for j in range(lamdano):
				err=0
				errt=0
				noofitr=50
				for k in range(noofitr):
					(traindata,trainy,validationdata,validationy)=data_partition(data,y,f)
					w=mylinridgereg(traindata,trainy,lamda)
					err+=meansquarederr(w,traindata,trainy)
					errt+=meansquarederr(w,testdata,testy)
				errLamda.append(err/noofitr)
				errLamdaTest.append(errt/noofitr)
				lamda+=lamdadiff
			errFrac.append(errLamda)
			errFracTest.append(errLamdaTest)
			f+=(1.0-fi)/(fractionno)
			# print ("xascs")
		l=[]
		for i in range(lamdano):
			l.append(lamdainitial+i*lamdadiff)	
		for i in range(len(errFrac)): 
			x=errFrac[i]
			axes = plt.gca()
			axes.set_ylim([0,20])
			plt.plot(l,x) 
			plt.xlabel('lamda values') 
			plt.ylabel('average mean squareerror') 
			plt.title('error vs lamda for fraction='+str(fi+i*(1.0-fi)/(fractionno))) 
			plt.show()


		lamdaminerr=[]
		valueminerr=[]
		fractions=[]
		for i in range(fractionno):
			fractions.append(fi+i*(1.0-fi)/fractionno)
			index=0;
			minimum=errFrac[i][0]
			for j in range(lamdano):
				if (errFrac[i][j]<minimum):
					# print (minimum)
					minimum=errFrac[i][j]
					index=j
			lamdaminerr.append(lamdainitial + index*lamdadiff)
			valueminerr.append(minimum)

		plt.plot(fractions,lamdaminerr) 
		plt.xlabel('frac values') 
		plt.ylabel('lamda values fro minimum err') 
		plt.title('lamda vs frac ') 
		plt.show()

		plt.plot(fractions,valueminerr) 
		plt.xlabel('frac values') 
		plt.ylabel('error values fro minimum err') 
		plt.title('min err vs frac ') 
		plt.show()

		l=[]
		for i in range(lamdano):
			l.append(lamdainitial+i*lamdadiff)	
		for i in range(len(errFracTest)): 
			x=errFracTest[i]
			axes = plt.gca()
			axes.set_ylim([0,50])
			plt.plot(l,x) 
			plt.xlabel('lamda values') 
			plt.ylabel('average mean squareerror') 
			plt.title('error vs lamda for fraction='+str(fi+i*(1.0-fi)/(fractionno))) 
			plt.show()



		#part 8
		lamdaminerr=[]
		valueminerr=[]
		fractions=[]
		minfrac=0
		minlamda=0
		globalmin=100000
		for i in range(fractionno):
			fractions.append(fi+i*(1.0-fi)/fractionno)
			index=0;
			minimum=errFracTest[i][0]
			for j in range(lamdano):
				if (errFracTest[i][j]<minimum):
					# print (minimum)
					minimum=errFracTest[i][j]
					index=j
				if(errFracTest[i][j]<globalmin):
					minfrac=fi+i*(1.0-fi)/fractionno
					minlamda=lamdainitial + j*lamdadiff
					globalmin=errFracTest[i][j]
			lamdaminerr.append(lamdainitial + index*lamdadiff)
			valueminerr.append(minimum)

		plt.plot(fractions,lamdaminerr) 
		plt.xlabel('frac values') 
		plt.ylabel('lamda values fro minimum err') 
		plt.title('lamda vs frac ') 
		plt.show()

		plt.plot(fractions,valueminerr) 
		plt.xlabel('frac values') 
		plt.ylabel('error values fro minimum err') 
		plt.title('min err vs frac ') 
		plt.show()	


		#part 9
		print ("minfrac is :"+str(minfrac))
		print ("minlabda is :"+str(minlamda))
		print ("minerr is :"+str(globalmin))
		(traindata,trainy,validationdata,validationy)=data_partition(data,y,minfrac)
		w=mylinridgereg(traindata,trainy,minlamda)
		f=np.matmul(traindata,w)
		(r,c)=f.shape
		fx=[]
		yx=[]
		l=[]
		for i in range(r):
			l.append(i)
			fx.append(f[i][0])
			yx.append(trainy[i][0])


		plt.scatter(yx,fx)
		plt.xlabel('actual value') 
		plt.ylabel('predicted value') 
		plt.title('predicted vs actual for min err lamda and fraction on training data') 
		plt.show()

		f=np.matmul(testdata,w)
		(r,c)=f.shape
		fx=[]
		yx=[]
		for i in range(r):
			fx.append(f[i][0])
			yx.append(testy[i][0])	

		plt.scatter(yx,fx)
		plt.xlabel('actual value') 
		plt.ylabel('predicted value') 
		plt.title('predicted and actual for min err lamda and fraction on test data') 
		plt.show()	


