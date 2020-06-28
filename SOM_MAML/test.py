import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = np.load('./Data/Finetune_data_SZtaxi/SZtaxi_finetune.npy')
y_in = np.squeeze(data[:, 5, 5, 0]).tolist()
y_out = np.squeeze(data[:, 5, 5, 1]).tolist()


plt.plot(y_out)
#plt.plot(y_in, label='volume in')
plt.legend()
plt.show()

#def read(path):
#    data = pd.read_csv(path, sep=',', header=None).values[:, 1:]
#    label = pd.read_csv(path, sep=',', header=None).values[:, 0]
#    return [np.expand_dims(data.reshape((-1,10,10)), axis=3), label]


#datain = set(pd.read_csv('./gjj/TaxiInFlow/05-1*/part-00000', sep=',', header=None).values[:, 0].tolist())
#dataout = set(pd.read_csv('./gjj/TaxiOutFlow/05-1*/part-00000', sep=',', header=None).values[:, 0].tolist())

#print(datain.difference(dataout))

#data_files = ['04-1', '04-2*', '04-30','05-0*', '05-1*', '05-2*', '05-3*', '06-0*','06-1*', '06-2*',  '06-3*']
#datatotal = np.zeros(shape=(0, 10, 10, 2))
#labeltotal = np.zeros(shape=(0,))
#for i in data_files:
#    print(i)
#    datain, label = read('./gjj/TaxiInFlow/'+i+'/part-00000')
#    dataout, label = read('./gjj/TaxiOutFlow/'+i+'/part-00000')#


#    data = np.concatenate([datain, dataout], axis=3)
#    print(data.shape)#

#    datatotal = np.concatenate([datatotal, data], axis=0)
#    print(labeltotal)
#    print(label.shape)
#    labeltotal = np.concatenate([labeltotal, label], axis=0)

#print(datatotal.shape)
#print(datatotal.shape)
#print(labeltotal.shape)
#np.save('./Data/Finetune_data_SZtaxi/SZtaxi_test.npy', datatotal)
#np.save('./Data/Finetune_data_SZtaxi/SZtaxi_test_label.npy', labeltotal)
#data = np.load('./Data/Finetune_data_SZtaxi/SZtaxi_test.npy', allow_pickle=True)
#labeltotal = np.load('./Data/Finetune_data_SZtaxi/SZtaxi_test_label.npy', allow_pickle=True)
#print(data.shape)
#print(labeltotal.shape)
#max = np.mean(data)
#print(max)









