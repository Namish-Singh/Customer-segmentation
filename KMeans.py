# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:50:52 2018

@author: Namish Kaushik
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


retail= pd.read_csv('Online+Retail.csv',encoding = 'ISO-8859-1',header=0)
retail['Country'].value_counts()
retail.head()
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],infer_datetime_format = True)
retail.describe()
retail.info()

def null_count(df):
    total_miss = df.isnull().sum()
    perc_miss = (df.isnull().sum())*100/len(df)
    mis_val_table=pd.concat([total_miss,perc_miss],axis =1)
    #renaming cols
    mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[ mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
    return mis_val_table_ren_columns
miss_val_res = null_count(retail)  

retail.isnull().sum()*100/retail.shape[0]

order_wise = retail.dropna()
order_wise.shape()
order_wise.isnull().sum()

#RFM analysis

amount = pd.DataFrame(order_wise.Quantity * order_wise.UnitPrice,columns = ['Amount'])
amount.head()
order_wise.index
order_wise.join(amount)

order_wise = pd.concat(objs= [order_wise,amount],axis =1,ignore_index = False )
monetary = order_wise.groupby('CustomerID').Amount.sum()
monetary = monetary.reset_index()
monetary.head()

frequency = order_wise[['CustomerID','InvoiceNo']]
frequency.head()

k= frequency.groupby('CustomerID').InvoiceNo.count()
k=pd.DataFrame(k)
k = k.reset_index()
k.columns = ['CustomerID','Frequency']
k.head()

# creating master dataset
master = monetary.merge(k,on = 'CustomerID',how = 'inner')
master.head()

# recency calculation
recency = order_wise[['CustomerID','InvoiceDate']]
recency.head() 

maximum = max(recency.InvoiceDate)
maximum = maximum +pd.DateOffset(days=1)
recency['diff'] = maximum - recency.InvoiceDate
recency.head()

df = pd.DataFrame(recency.groupby('CustomerID').diff.min())
df= df.reset_index()
df.columns = ['CustomerID','Recency'] 
df.head()

RFM = k.merge(monetary, on = 'CustomerID')
RFM = RFM.merge(df,on = 'CustomerID')
RFM.head()

plt.boxplot(RFM.Amount)
Q1 = RFM.Amount.quantile(0.25)
Q3 = RFM.Amount.quantile(0.75)
IQR = Q3-Q1
RFM = RFM[(RFM.Amount >= Q1 - 1.5 *IQR)&(RFM.Amount <= Q3 +1.5*IQR)]

plt.boxplot(RFM.Frequency)
Q1 = RFM.Frequency.quantile(0.25)
Q3 = RFM.Frequency.quantile(0.75)
IQR = Q3-Q1
RFM = RFM[(RFM.Frequency >= Q1 - 1.5 *IQR)&(RFM.Frequency <= Q3 +1.5*IQR)]

plt.boxplot(RFM.Recency)
Q1 = RFM.Recency.quantile(0.25)
Q3 = RFM.Recency.quantile(0.75)
IQR = Q3-Q1
RFM = RFM[(RFM.Recency >= Q1 - 1.5 *IQR)&(RFM.Recency <= Q3 +1.5*IQR)]


RFM.head()

RFM_normal = RFM.drop('CustomerID',axis =1)
RFM_normal.Recency = RFM_normal.Recency.dt.days
RFM_normal.Recency.head()


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
standard_scaler.fit_transform(RFM_normal)

# sum of squared distance
ssd= []

for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters,max_iter = 50)
    model_clus.fit(RFM_normal)
    ssd.append(model_clus.inertia_)
plt.plot(ssd)    

model_clus5 = KMeans(n_clusters =5,max_iter =50)
model_clus5.fit(RFM_normal)

#analysis of cluster formed

RFM.index = pd.RangeIndex(len(RFM.index))
RFM_km = pd.concat([RFM,pd.Series(model_clus5.labels_)],axis =1)
RFM_km.columns = ['CustomerID','Frequency','Amount','recency','ClusterID']
RFM_km.columns

RFM_km['Recency'] = RFM_km.recency.dt.days
km_clusters_amount = pd.DataFrame(RFM_km.groupby(['ClusterID']).Amount.mean())
km_clusters_frequency = pd.DataFrame(RFM_km.groupby(['ClusterID']).Frequency.mean())

km_clusters_recency = pd.DataFrame(RFM_km.groupby(['ClusterID']).Recency.mean())
df = pd.concat([pd.Series([0,1,2,3,4]),km_clusters_amount,km_clusters_frequency,km_clusters_recency],axis =1)
df.head()  
df.columns= ['ClusterID','Amount_mean'  , 'Frequency_mean','Recency_mean']

sns.barplot(x = df['ClusterID'],y = df.Amount_mean)
sns.barplot(x = df['ClusterID'],y = df.Frequency_mean)
sns.barplot(x = df['ClusterID'],y = df.Recency_mean)





# 7 steps

# missing value, data transforation,outlier treatment,data standardisation,finding the optimal value of K
#implementing the K menas algorithm, analysisng teh cluster of customers to obtain busniess insights


# now we will do heiracrachical clustering

mergings = linkage(RFM_normal,method = 'single',metric= 'euclidean')
dendrogram(mergings)
plt.show()

mergings = linkage(RFM_normal,method = 'complete',metric= 'euclidean')
dendrogram(mergings)
plt.show()



clusterCut = pd.Series(cut_tree(mergings,n_clusters =5).reshape(-1,))
RFM_hc = pd.concat([RFM,clusterCut],axis=1)
RFM_hc.columns = ['CustomerID','Frequency','Amount','Recency','ClusterID']

RFM_hc.Recency = RFM_hc.Recency.dt.days
km_clusters_amount = pd.DataFrame(RFM_hc.groupby(['ClusterID']).Amount.mean())
km_clusters_frequency = pd.DataFrame(RFM_hc.groupby(['ClusterID']).Frequency.mean())
km_clusters_recency = pd.DataFrame(RFM_hc.groupby(['ClusterID']).Recency.mean())

df_h = pd.concat([pd.Series([0,1,2,3,4]),km_clusters_amount,km_clusters_frequency,
                  km_clusters_recency],axis =1)
df_h.columns = ['ClusterID','Amount_mean','Frequency_mean','Recency_mean']
df_h.head()

sns.barplot(x= df_h['ClusterID'], y = df_h['Amount_mean'])
sns.barplot(x= df_h['ClusterID'], y = df_h['Frequency_mean'])
sns.barplot(x= df_h['ClusterID'], y = df_h['Recency_mean'])
df_h['Amount_mean']
