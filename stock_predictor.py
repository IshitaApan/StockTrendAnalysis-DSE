import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
import numpy.random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Dropout
from keras import backend as K
import os
# import preprocessing 

folder_path = "OHLC_prediction/GRU/companyWiseData/"
print("printing files...")
for filename in os.listdir(folder_path):
	np.random.seed(0)
	if not filename.endswith(".csv"):
		continue
	filepath = os.path.join(folder_path,filename)
	filepath.replace("\\","")
	print("Reading....", filepath)
	# model_name = "closing_price_prediction/models/" + filename.replace(".csv","") + ".model"
	# close_price_image = "closing_price_prediction/images/" + filename.replace(".csv","") + ".png"
	model_name = "OHLC_prediction/GRU/models/" + filename.replace(".csv","") + ".model"
	# OHLC_image = "OHLC_prediction/images/" + filename.replace(".csv","") + ".png"
	OHLC_image = "OHLC_prediction/GRU/images/" + filename.replace(".csv","") + ".png"
	# # print(filepath, model_name, close_price_image, ohlc_image)
	#feature_columns = ["opening_price","High","Low","Close","OHLC_avg"]
	feature_columns = ["opening_price" , "high", "low", "closing_price", "OHLC_avg" ]
	# feature_columns = ["opening_price","high","low","closing_price","OHLC_avg"]
	columns_to_be_predicted="OHLC_avg"
	#columns_to_be_predicted="closing_price"
	# inputDirectory = 'companyWiseData//'
	# filename = '1JANATAMF.csv'
	# #filename = '3RDICB.csv'
	dataset = pd.read_csv(filepath)
	# print(dataset.head(20))
	# print(dataset.info())
	# print(dataset.dtypes)
	OHLC_avg = dataset[['yesterdays_closing_price', 'high', 'low', 'closing_price']].mean(axis = 1)
	HLC_avg = dataset[['high', 'low', 'closing_price']].mean(axis = 1)
	OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1))
	scaler = MinMaxScaler(feature_range=(0, 1))
	OHLC_avg = scaler.fit_transform(OHLC_avg)
	# OHLC_avg = OHLC_avg.reshape(1,len(OHLC_avg))
	OHLC_avg = pd.DataFrame(OHLC_avg, columns=["OHLC_avg"])

	dataset = pd.concat([dataset, OHLC_avg], axis = 1)
	print("dataset",dataset.info())

	original_y = dataset[[columns_to_be_predicted]]

	# classifier_df = dataset.drop(columns_to_drop,axis=1)
	classifier_df = dataset[feature_columns].copy()

	print("classifier",classifier_df.info())

	train_size = int(len(classifier_df)*0.75)
	test_size = len(classifier_df) - train_size
	train_df, test_df = classifier_df[0:train_size], classifier_df[train_size:]
	print("train_df", train_df)
	print("test_df", test_df)

	# train_x = train_df.iloc[:,train_df.columns!=columns_to_be_predicted]
	# train_y = train_df.iloc[:,train_df.columns==columns_to_be_predicted]

	#train_x = train_df.iloc[:len(train_df)-1,train_df.columns!=columns_to_be_predicted]
	train_x = train_df.iloc[:len(train_df)-1,train_df.columns==columns_to_be_predicted]
	train_y = train_df.iloc[1:,train_df.columns==columns_to_be_predicted]
	# print("train_x", train_x)
	# print("train_y", train_y)
	#test_x = test_df.iloc[:,test_df.columns!=columns_to_be_predicted]
	# test_x = test_df.iloc[:,test_df.columns==columns_to_be_predicted]
	# test_y = test_df.iloc[:,test_df.columns==columns_to_be_predicted]
	test_x = test_df.iloc[:len(test_df)-1,test_df.columns==columns_to_be_predicted]
	test_y = test_df.iloc[1:,test_df.columns==columns_to_be_predicted]

	# original_y = train_y + test_y
	# print('original y', original_y)

	train_x = np.reshape(train_x.values, (train_x.shape[0], 1, train_x.shape[1]))
	test_x = np.reshape(test_x.values, (test_x.shape[0], 1, test_x.shape[1]))

	model = Sequential()
	model.add(GRU(64, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences = True))
	# model.add(LSTM(16))
	model.add(GRU(32))
	# model.add(Activation('linear'))
	model.add(Dense(1))
	model.add(Activation('linear'))


	# model = Sequential () 
	# d=0.3
	# model.add(LSTM(256, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
	# model.add(Dropout(d))
	# model.add(LSTM(256, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=False))
	# model.add(Dropout(d)) 
	# model.add(Dense(32,kernel_initializer='uniform', activation='relu'))
	# model.add(Dense(1,kernel_initializer='uniform', activation = 'linear'))

	model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['accuracy']) # Try SGD, adam, adagrad and compare!!!
	history = model.fit(train_x, train_y, epochs=50, batch_size=10, validation_data=(test_x, test_y),verbose=2, shuffle=False)

	# plt.plot(history.history['loss'], label='train loss')
	# plt.plot(history.history['val_loss'], label='test loss')
	# plt.legend()
	# plt.show()
	model.save(model_name)
	trainPredict = model.predict(train_x)
	testPredict = model.predict(test_x)
	print("testPredict", len(trainPredict), len(testPredict))
	print(testPredict)

	step_size=1
	# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
	trainPredictPlot = np.empty_like(OHLC_avg)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

	# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
	testPredictPlot = np.empty_like(OHLC_avg)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict):len(trainPredict)+len(testPredict), :] = testPredict

	
	plt.plot(original_y, 'g', label ='original '+ columns_to_be_predicted)
	plt.plot(trainPredictPlot, 'r', label = 'predicted training set')
	plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
	plt.legend(loc = 'upper right')
	plt.xlabel('Time in Days')
	plt.ylabel(columns_to_be_predicted + ' Value of ' + filename)
	print("Saving... "+ OHLC_image)
	plt.savefig(OHLC_image)
	plt.clf()
	#plt.show()
	trainPredict = scaler.inverse_transform(trainPredict)
	testPredict = scaler.inverse_transform(testPredict)
	trainScore = math.sqrt(mean_squared_error(train_y, trainPredict))
	print('Train RMSE: %.2f' % (trainScore))
	testScore = math.sqrt(mean_squared_error(test_y, testPredict))
	print('Test RMSE: %.2f' % (testScore))
	write_rmse = pd.DataFrame([[filename.replace(".csv",""), trainScore, testScore]], columns=['Company Code', 'Training RMSE', 'Testing RMSE'])
	#
	write_rmse.to_csv('OHLC_prediction/GRU/GRU_RMSE.csv', mode='a', index=False, header = False)
	K.clear_session()
