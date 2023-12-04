import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def process():
	dataset_train = pd.read_csv('KDDTrain+.csv')
	dataset_test = pd.read_csv('KDDTest+.csv')
	X_train = dataset_train.iloc[:, :-2].values
	Y_train = dataset_train.iloc[:, 41].values

	X_test = dataset_test.iloc[:, :-2].values
	Y_test = dataset_test.iloc[:, 41].values

	print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

	print(X_test[:, 3])

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
	X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

	x_train = np.array(X_train)
	x_test = np.array(X_test)
	y_train1 = np.array(Y_train)
	y_test1 = np.array(Y_test)

	y_train= to_categorical(y_train1)
	y_test= to_categorical(y_test1)
	
	batch_size = 32

	model = Sequential()
	model.add(LSTM(80,input_dim=40))
	model.add(Dropout(0.1))
	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(6))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	history=model.fit(x_train, y_train, batch_size=batch_size, epochs=20, validation_data=(x_test, y_test))

	loss, accuracy = model.evaluate(x_test, y_test)
	print(accuracy)


	history_dict = history.history
	print(history_dict.keys())

	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, loss, 'bo', label='Training loss')
	# b is for "solid blue line"
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig('results/RNNloss.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()	

	plt.clf()   # clear figure
	acc_values = history_dict['accuracy']
	val_acc_values = history_dict['val_accuracy']
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('results/RNNAccuracy.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()	
	
	y_pred1 = model.predict(x_test)
	y_pred=np.argmax(y_pred1,axis=1)
	print(y_pred)
	print(Y_test)
	
	mse=mean_squared_error(Y_test, y_pred)
	mae=mean_absolute_error(Y_test, y_pred)
	r2=r2_score(Y_test, y_pred)
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR RNN IS %f "  % mse)
	print("MAE VALUE FOR RNN IS %f "  % mae)
	print("R-SQUARED VALUE FOR RNN IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(Y_test, y_pred))
	print("RMSE VALUE FOR RNN IS %f "  % rms)
	ac=accuracy_score(Y_test,y_pred)
	print ("ACCURACY VALUE RNN IS %f" % ac)
	print("---------------------------------------------------------")

	result2=open("results/resultRNN.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()

	result2=open('results/RNNMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/RNNMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title('RNN Metrics Value')
	fig.savefig('results/RNNMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()


