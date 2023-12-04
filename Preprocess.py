import sys
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

def process(path):
	data = pd.read_csv(path)
	print(data.head())
	names=list(data.columns)
	correlations = data.corr()


	# plot correlation matrix
	fig = plt.figure()
	fig.canvas.set_window_title('Correlation Matrix')
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0,42,1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names)
	ax.set_yticklabels(names)
	plt.savefig('results/Correlation Matrix.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()	
	
	ncols=3
	plt.clf()
	f = plt.figure(1)
	f.suptitle(" Data Histograms", fontsize=12)
	vlist = list(data.columns)
	nrows = len(vlist) // ncols
	if len(vlist) % ncols > 0:
		nrows += 1
	for i, var in enumerate(vlist):
		plt.subplot(nrows, ncols, i+1)
		plt.hist(data[var].values, bins=15)
		plt.title(var, fontsize=10)
		plt.tick_params(labelbottom='off', labelleft='off')
	plt.tight_layout()
	plt.subplots_adjust(top=0.88)
	plt.savefig("results/Histogram.png") 
	plt.pause(5)
	plt.show(block=False)
	plt.close()


	dos=[]
	u2r=[]
	r2l=[]
	probe=[]
	normal= []
	unknown=[]
	
	dataset=pd.read_csv('KDDTrain+.csv').values
	dataset1=pd.read_csv('KDDTest+.csv').values
	
	
	
	X_train = dataset[:,0:41]
	y_train = dataset[:,41] 
	
	X_test = dataset1[:,0:41]
	y_test = dataset1[:,41]

	for c in y_train:
		if c==1.0:
			dos.append(c)
		if c==2.0:
			u2r.append(c)
		if c==3.0:
			r2l.append(c)
		if c==4.0:
			probe.append(c)
		if c==5.0:
			normal.append(c)
	
	for c in y_test:
		if c==1.0:
			dos.append(c)
		if c==2.0:
			u2r.append(c)
		if c==3.0:
			r2l.append(c)
		if c==4.0:
			probe.append(c)
		if c==5.0:
			normal.append(c)
	
	tot=len(dos)+len(u2r)+len(r2l)+len(probe)+len(normal)
	d=(len(dos)/tot)*100
	u=(len(u2r)/tot)*100
	r=(len(r2l)/tot)*100
	p=(len(probe)/tot)*100
	n=(len(normal)/tot)*100
	
	print(len(dos))
	print(len(u2r))
	print(len(r2l))
	print(len(probe))
	print(len(normal))
	
	print(d)
	print(u)
	print(r)
	print(p)
	print(n)
	
	colors = ["#FF0000", "#FF0000", "#FF0000", "#FF0000", "#008000"]
	explode = (0.1, 0, 0, 0, 0)  
	
	
	alc = ["dos","U2R","R2L","PROBE","NORMAL"]
	acc = [d,u,r,p,n]
	
	
	fig = plt.figure(0)    
	plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
	plt.xlabel('Methods')
	plt.ylabel('Rate in %')
	plt.title('Detection Rate')
	plt.savefig("results/Detection Rate.png") 
	plt.pause(5)
	plt.show(block=False)
	plt.close()
