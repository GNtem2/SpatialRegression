#!/usr/bin/env python
# coding: utf-8

# In[29]:


#Getting Data for each LGA
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
from itertools import combinations
df=pd.read_csv('Australia_LGA_Variables_Dataset.csv')
df=df.replace('..', 'NaN')
df=df.replace('n.a.', 'NaN')
df=df.replace('np', 'NaN')
df['Median age (years)'] = np.array(df['Median age (years)'], dtype=float)
df['Code']=df['Code'].astype(int)
df=df.dropna()
#print(df)
stats_summary=[]
#Getting LGA Shapefile
df_1=geopandas.read_file('LGA_2016_AUST.shp')

var_names=['Aboriginal population as proportion of total population (%)','% Australian born',
           '% born in non-English speaking countries','% born overseas who speak English not well or not at all',
           '% Permanent migrants under the Humanitarian Program','ASR People who left school at Year 10 or below, or did not go to school per 100',
           '% single parent families','% persons living in social housing','% private dwellings where Internet accessed',
           '% unemployed','Median Income','ASR Cancer per 100,000',
           'ASR diabetes mellitus per 100','ASR mental and behavioural problems per 100',
           'ASR Heart Stroke or Vascular Disease per 100','ASR asthma per 100','ASR COPD per 100',
           'ASR arthritis per 100','ASR osteoporosis per 100','ASR High Blood Pressure per 100','ASR Obese per 100',
           'ASR Smoking per 100','ASR Alcohol per 100','ASR Low Exercise per 100',
           '% persons with a profound or severe disability','GP Rate per 100,000 people',
           'Specialists Rate per 100,000 people','Nurse Rate per 100,000 people','Dentist Rate per 100,000 people',
           'ASR Nervous System Admissions per 100,000','ASR Circulatory Diseases Admission per 100,000',
           'ASR Ischaemic Heart Disease per 100,000','ASR Stroke per 100,000','ASR Heart Failure per 100,000']

def rSubset(arr, r):
    return list(combinations(arr,r))

for i in range(len(rSubset(var_names,2))):
    variable_names=list(rSubset(var_names,2)[i])
    for j in variable_names:
        df[j]=np.array(df[j],dtype=float)
    df=df.dropna()
    #Nonspatial Regression. 
    #m1 = spreg.OLS(
        #df[['Median age (years)']].values, 
        #df[variable_names].values,
        #name_y='Median age (years)', 
        #name_x=variable_names
    #)
    #print(m1.summary)

    #Combining the two Dataframes
    df_1['LGA_CODE16']=df_1['LGA_CODE16'].astype(int)


    df_2=df_1.merge(df, left_on='LGA_CODE16', right_on='Code',how='right')
    #print(df_2)

    #Using Queen
    df_Queen=df_2.copy()
    QWeight = weights.Queen.from_dataframe(df_Queen)

    #Spatial Dependence with Queen
    variable_names_of_interest=[]
    for i in variable_names:
        df_Queen=df_Queen.rename(columns={i: 'var_of_interest'+i})
        variable_names_of_interest.append('var_of_interest'+i)
    wx = df_Queen.filter(
        like='var_of_interest'
    ).apply(
        lambda y: weights.spatial_lag.lag_spatial(QWeight, y)
    ).rename(
        columns=lambda c: 'w_'+c
    )
    slx_exog = df_Queen[variable_names_of_interest].join(wx)
    mQueensd = spreg.OLS(
        df_Queen[['Median age (years)']].values, 
        slx_exog.values,
        name_y='l_Median age (years)', 
        name_x=slx_exog.columns.tolist()
    )


    stats_summary.append([variable_names,mQueensd.r2,mQueensd.mulColli,mQueensd.betas,len(df_Queen)])


# In[30]:


df_stats=pd.DataFrame(stats_summary, columns=['variables','r2','multicollinearity number','model','number of LGAs'])
print(df_stats)
df_stats.to_csv('two_variables_w_median_income.csv')


# In[ ]:




