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
df=GetYahooData_v2(ticker,2000,interval)
df['flag']=np.where(df['Adj Close'].diff(7)>0, 1, 0)
'''
binary classification
# 1d data
diff(7)/window_len=13: accuracy:79.56/70.166
Test accuracy of the best model: 0.8729282021522522
Test results - Loss: 1.2673572301864624 - Accuracy: 80.66298365592957%

Test accuracy of the best model: 0.8729282021522522
Test results - Loss: 1.0065486431121826 - Accuracy: 81.76795840263367%

with 'trend_stc_high','trend_stc_low','trend_stc'
Test accuracy of the best model: 0.8323699235916138
Test results - Loss: 2.301954984664917 - Accuracy: 75.72254538536072%

with all indicators
Test accuracy of the best model: 0.8989899158477783
Test results - Loss: 3.1546730995178223 - Accuracy: 80.8080792427063%

Test accuracy of the best model: 0.9090909361839294
TTest results - Loss: 3.174992799758911 - Accuracy: 82.45614171028137%

QQQ-2000
Test accuracy of the best model: 0.8815789222717285
Test results - Loss: 2.1134541034698486 - Accuracy: 84.64912176132202%

Test accuracy of the best model: 0.8859649300575256
Test results - Loss: 3.174992799758911 - Accuracy: 82.45614171028137%

XLK 2000:
       Test accuracy of the best model: 0.8938053250312805
       Test results - Loss: 2.5124361515045166 - Accuracy: 76.54867172241211%

'''    
#df['flag2']=np.where(df['Close'].diff(2)>0, True, False)
#df['flag5']=np.where(df['Close'].diff(5)>0, True, False)
#https://github.com/bukosabino/ta
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Adj Close", volume="Volume")

print (df.head)
'''
Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'id',
       'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
       'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
       'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
       'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
       'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
       'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
       'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
       'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
       'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
       'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
       'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
       'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
       'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
       'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
       'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
       'trend_psar_down', 'trend_psar_up_indicator',
       'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
       'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
       'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
       'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
       'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
       'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
       'others_cr'],
'''
df['trend_stc_high']=df['trend_stc']>=75
df['trend_stc_low']=df['trend_stc']<=25
#df_input=df[['Appliances','T_out', 'RH_1', 'Visibility']]
#in_cols=['flag','Close','High', 'Low','trend_stc','trend_ema_fast', 'trend_ema_slow']

in_cols=['flag','Close','High', 'Low','momentum_kama', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',  'trend_ema_fast', 'trend_ema_slow','trend_sma_fast', 'trend_sma_slow','trend_stc_high','trend_stc_low','trend_stc']
in_cols=['flag','Adj Close','High', 'Low','Volume','volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
       'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
       'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
       'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
       'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
       'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
       'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
       'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
       'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
       'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 
       'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
       'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',

       'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',

       'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',


       'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
       'trend_psar_up_indicator',  
       #'trend_psar_down',

       'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',



       'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
       'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',

       'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
       'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',

       'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
       'others_cr']
in_cols=['flag','Adj Close','High', 'Low','Volume']
in_cols=['flag','Adj Close','High', 'Low','Volume','volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
       'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
       'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
       'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
       'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
       'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
       'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
       'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
       'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
       'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 
       'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
       'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',

       'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',

       'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',#good




       'trend_aroon_down', 'trend_aroon_ind',

       #'trend_psar_up',
       #'trend_psar_up_indicator', 
       #'trend_psar_down',

       #'trend_psar_down_indicator', 
       'momentum_rsi', 'momentum_stoch_rsi', #good 2




       'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
       'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',

       'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
       'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',

       'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
       'others_cr']

# add 'trend_stc' reduces accuracy
#in_cols=['flag','Close','High', 'Low','momentum_kama', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',  'trend_ema_fast', 'trend_ema_slow','volatility_bbh', 'volatility_bbl','volatility_bbhi', 'volatility_bbli' ]
df_input=df[in_cols]
df_input=df_input.dropna()
df_input.describe()
print('----------------------------------------')
print(df_input.iloc[-1:].index)
print('----------------------------------------')
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
history = model.fit(train_generator, epochs=500,
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
binary_predictions = predictions > 0.5
df_final=df_input[predictions.shape[0]*-1:]
df_final['App_Pred']=binary_predictions
print('df_final:',df_final)

test_results=model.evaluate(test_generator, verbose=0)  
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

new_data = np.random.randn(1, win_length, len(in_cols))
new_data = np.random.randn(win_length, len(in_cols))

print('doing predictions.....')

for i in range(len(features)-win_length, len(features)):

       #new_data=features[-win_length-i:-i]
       new_data=features[i-win_length+1:i+1]
        
       new_data=np.reshape(new_data, (1,new_data.shape[0],new_data.shape[1]))

       predictions=model.predict(new_data)
       binary_predictions = predictions > 0.5
       print(df_input.iloc[i:i+1].index.to_pydatetime(),binary_predictions)
'''
df_final=df_input[predictions.shape[0]*-1:]
df_final['App_Pred']=binary_predictions
print('df_final:',df_final.tail())
'''
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
