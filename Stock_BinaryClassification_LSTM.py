#https://github.com/srivatsan88/End-to-End-Time-Series/blob/master/Multivariate_Time_Series_Modeling_using_LSTM.ipynb
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from YahooData import *
import ta

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False
'''
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
df.info()
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df.set_index('date')[['Appliances', 'lights','T_out', 'RH_1', 'Visibility']].plot(subplots=True)
'''
ticker='XLK'
interval='1d'
df=GetYahooData_v2(ticker,1000,interval)
df['flag']=np.where(df['Close'].diff(7)>0, 1, 0)
'''
binary classification
# 1d data
diff(7)/window_len=13: accuracy:79.56/70.166
Test accuracy of the best model: 0.8729282021522522
Test results - Loss: 1.2673572301864624 - Accuracy: 80.66298365592957%

Test accuracy of the best model: 0.8729282021522522
Test results - Loss: 1.0065486431121826 - Accuracy: 81.76795840263367%

'''
#df['flag2']=np.where(df['Close'].diff(2)>0, True, False)
#df['flag5']=np.where(df['Close'].diff(5)>0, True, False)
#https://github.com/bukosabino/ta
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")

print (df.head)
#df_input=df[['Appliances','T_out', 'RH_1', 'Visibility']]
in_cols=['flag','Close','High', 'Low','momentum_kama', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',  'trend_ema_fast', 'trend_ema_slow','trend_sma_fast', 'trend_sma_slow' ]
#in_cols=['flag','Close','High', 'Low','momentum_kama', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',  'trend_ema_fast', 'trend_ema_slow','volatility_bbh', 'volatility_bbl','volatility_bbhi', 'volatility_bbli' ]
df_input=df[in_cols]
df_input=df_input.dropna()
df_input.describe()
#df_input.query("Appliances > 500")

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_input)
     
data_scaled
features=data_scaled
target=data_scaled[:,0]
print('features=', features)
print('target:',target )
TimeseriesGenerator(features, target, length=2, sampling_rate=1, batch_size=1)[0]


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=123, shuffle = False)
     

x_train.shape
x_test.shape
win_length=13
batch_size=32
num_features=len(in_cols)
train_generator = TimeseriesGenerator(x_train, y_train, length=win_length, sampling_rate=1, batch_size=batch_size)
test_generator = TimeseriesGenerator(x_test, y_test, length=win_length, sampling_rate=1, batch_size=batch_size)
     

train_generator[0]

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128, input_shape= (win_length, num_features), return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5)) 
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5)) 
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.summary()


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    patience=10,
                                                    mode='max')

modelfilepath='./{}_{}_best_model.keras'.format(ticker, interval)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=modelfilepath, 
                                      monitor='val_accuracy', 
                                      save_best_only=True,
                                      mode='max')
'''
model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])
'''
''' BinaryCrossentropy '''
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss,metrics=['accuracy'])
'''
history = model.fit(train_generator, epochs=500  ,
                    validation_data=test_generator,
                    shuffle=False,
                    callbacks=[early_stopping])
'''
history = model.fit(train_generator, epochs=500  ,
                    validation_data=test_generator,
                    shuffle=True,
                    callbacks=[checkpoint_callback]
                    )

# Load the best model
best_model = tf.keras.models.load_model(modelfilepath)

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(test_generator, verbose=1)
print('Test accuracy of the best model:', test_acc)


predictions=model.predict(test_generator)
binary_predictions = (predictions > 0.5)
df_final=df_input[predictions.shape[0]*-1:]
df_final['App_Pred']=binary_predictions
print('df_final:',df_final)

test_results=model.evaluate(test_generator, verbose=0)  
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')



'''
predictions.shape[0]
print('predictions:',predictions)

print('x_test,y_test:',x_test, y_test)

x_test

x_test[:,1:][win_length:]

df_pred=pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][win_length:])],axis=1)
     
print('df_pred=',df_pred)
rev_trans=scaler.inverse_transform(df_pred)
print('rev_trans=',rev_trans)
df_final=df_input[predictions.shape[0]*-1:]
print('df_final.count()=',df_final.count())

df_final['App_Pred']=rev_trans[:,0]

print('df_final:',df_final)

df_final[['Close','App_Pred']].plot()
'''
print('Done....')
