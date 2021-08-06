#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Getting Data for each sa2
from pysal.model import spreg
from pysal.lib import weights
from pysal.explore import esda
from scipy import stats
import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import seaborn
import folium
df=pd.read_csv('.../sa2_SocioEco.csv')
df=df.dropna()
df['SA2_Code']=df['SA2_Code'].astype(int)

print(df)


# In[2]:


#Getting sa2 Shapefile
df_1=geopandas.read_file('.../1270055001_sa2_2016_aust_shape/SA2_2016_AUST.shp')

variable_names=['varMedian_Income']

print(df_1)


# In[3]:


#Combining the two Dataframes
df_1['SA2_MAIN16']=df_1['SA2_MAIN16'].astype(int)


df_2=df_1.merge(df, left_on='SA2_MAIN16', right_on='SA2_Code',how='right')
print(df_2)


# In[4]:


#Percentage with Heart Stroke or Vascular Disease in each sa2
ax_HSV=df_2.plot(color='k', alpha=0.5,figsize=(24,12))
df_2.plot('Heart_Stroke_Vascular', ax=ax_HSV, legend=True)

plt.show()


# In[5]:


#Nonspatial Regression. To see how Median Income affects Heart Stroke Vascular Disease
m1 = spreg.OLS(
    df[['Heart_Stroke_Vascular']].values, 
    df[variable_names].values,
    name_y='Heart_Stroke_Vascular', 
    name_x=variable_names
)
print(m1.summary)


# In[6]:


#This shows the differences between the actual and predicted values from nonspatial regression for each sa2 in each sa4
df_NSR=df_2.copy()
df_NSR['residual'] = m1.u
medians = df_NSR.groupby(
    "SA4_Name"
).residual.median().to_frame(
    'SA4_residual'
)

f = plt.figure(figsize=(15,3))
ax = plt.gca()
seaborn.boxplot(
    'SA4_Name', 
    'residual', 
    ax = ax,
    data=df_NSR.merge(
        medians, 
        how='left',
        left_on='SA4_Name',
        right_index=True
    ).sort_values(
        'SA4_residual'), palette='bwr'
)
f.autofmt_xdate()


ax_NSR_map=df_NSR.plot(color='k', alpha=0.5,figsize=(20,10))
df_NSR.plot('residual', ax=ax_NSR_map, legend=True)

plt.show()


# In[7]:


#Weighted KNN. From this we can see clusters of sa2 that are over or under predicted
df_KNN=df_2.copy()
knn = weights.KNN.from_dataframe(df_KNN, k=10)
lag_residual = weights.spatial_lag.lag_spatial(knn, m1.u)
ax = seaborn.regplot(
    m1.u.flatten(), 
    lag_residual.flatten(), 
    line_kws=dict(color='orangered'),
    ci=None
)
ax.set_xlabel('Model Residuals - $u$')
ax.set_ylabel('Spatial Lag of Model Residuals - $W u$');


outliers = esda.moran.Moran_Local(m1.u, knn, permutations=9999)
error_clusters = (outliers.q % 2 == 1)
error_clusters &= (outliers.p_sim <= .001) 
df_KNN.assign(
    error_clusters = error_clusters,
    local_I = outliers.Is
).query(
    "error_clusters"
).sort_values(
    'local_I'
).plot(
    'local_I', cmap='bwr', marker='.'
);
plt.show()


# In[8]:


#Using Queen instead of Weighted KNN
df_Queen=df_2.copy()
QWeight = weights.Queen.from_dataframe(df_Queen)
lag_residual = weights.spatial_lag.lag_spatial(QWeight, m1.u)
ax = seaborn.regplot(
    m1.u.flatten(), 
    lag_residual.flatten(), 
    line_kws=dict(color='orangered'),
    ci=None
)
ax.set_xlabel('Model Residuals - $u$')
ax.set_ylabel('Spatial Lag of Model Residuals - $W u$');


outliers = esda.moran.Moran_Local(m1.u, QWeight, permutations=9999)
error_clusters = (outliers.q % 2 == 1)
error_clusters &= (outliers.p_sim <= .001) 
df_Queen.assign(
    error_clusters = error_clusters,
    local_I = outliers.Is
).query(
    "error_clusters"
).sort_values(
    'local_I'
).plot(
    'local_I', cmap='bwr', marker='.'
);
plt.show()


# In[9]:


#Spatial Heterogeneity
df_SH=df_2.copy()
mSH = spreg.OLS_Regimes(
    df_SH[['Heart_Stroke_Vascular']].values, 
    df_SH[variable_names].values,
    df_SH['SA4_Name'].tolist(),
    constant_regi='many',
    cols2regi=[False]*len(variable_names),
    regime_err_sep=False,
    name_y='Heart_Stroke_Vascular', 
    name_x=variable_names
)
df_SH['predicted']=mSH.predy
print(mSH.summary)


axSH = df_SH.plot(color='k', alpha=0.5, figsize=(20,10))
df_SH.plot('predicted', ax=axSH, legend=True)
axSH.set_title("Predicted Percentage of Heart Stroke or Vascular Disease")
plt.show()


# In[10]:


#Plotting Regional Fixed Effects
formula = 'Heart_Stroke_Vascular ~ ' + '+ '.join(variable_names)+ ' + SA4_Name -1'
mSHr = sm.ols(formula, data=df_SH).fit()
Region_effects = mSHr.params.filter(like='SA4_Name')
stripped = Region_effects.index.str.strip('SA4_Name[').str.strip(']')
Region_effects.index = stripped
Region_effects = Region_effects.to_frame('fixed_effect')
axSHfe = df_SH.plot(color='k', alpha=0.5, figsize=(20,10))

df_SH.merge(
    Region_effects, 
    how='left',
    left_on='SA4_Name', 
    right_index=True
).dropna(
    subset=['fixed_effect']
).plot(
    'fixed_effect', ax=axSHfe, legend=True
)
axSHfe.set_title("Victoria Regional Fixed Effects")
plt.show()


# In[11]:


#Spatial Dependence with WeightedKNN
wx = df_KNN.filter(
    like='var'
).apply(
    lambda y: weights.spatial_lag.lag_spatial(knn, y)
).rename(
    columns=lambda c: 'w_'+c
)
slx_exog = df_KNN[variable_names].join(wx)
mKNNsd = spreg.OLS(
    df_KNN[['Heart_Stroke_Vascular']].values, 
    slx_exog.values,
    name_y='l_Heart_Stroke_Vascular', 
    name_x=slx_exog.columns.tolist()
)

df_KNN['predicted_SD']=mKNNsd.predy


axKNNsd = df_KNN.plot(color='k', alpha=0.5, figsize=(20,10))
df_KNN.plot('predicted_SD', ax=axKNNsd, legend=True)
axKNNsd.set_title("Predicted Percentage of Heart Stroke or Vascular Disease")
plt.show()
print(mKNNsd.summary)


# In[12]:


#Spatial Dependence with Queen
wx = df_Queen.filter(
    like='var'
).apply(
    lambda y: weights.spatial_lag.lag_spatial(QWeight, y)
).rename(
    columns=lambda c: 'w_'+c
)
slx_exog = df_Queen[variable_names].join(wx)
mQueensd = spreg.OLS(
    df_Queen[['Heart_Stroke_Vascular']].values, 
    slx_exog.values,
    name_y='l_Heart_Stroke_Vascular', 
    name_x=slx_exog.columns.tolist()
)

df_Queen['predicted_SD']=mQueensd.predy


axQueensd = df_Queen.plot(color='k', alpha=0.5, figsize=(20,10))
df_Queen.plot('predicted_SD', ax=axQueensd, legend=True)
axQueensd.set_title("Predicted Percentage of Heart Stroke or Vascular Disease")
plt.show()
print(mQueensd.summary)
df_Qsd=pd.DataFrame(mQueensd.x)
print(df_Qsd.corr())


# In[36]:


#SR using Queen
df_Queen['SR']=df_Queen['predicted_SD']/df_Queen['Heart_Stroke_Vascular']
print(df_Queen['SR'])
axSRQ = df_Queen.plot(color='k', alpha=0.25, figsize=(20,10))


df_Queen.plot('SR', ax=axSRQ, legend=True,scheme='Quantiles')
axSRQ.set_title("SR of Heart Stroke or Vascular Disease")
plt.show()

