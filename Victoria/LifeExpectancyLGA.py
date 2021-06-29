#!/usr/bin/env python
# coding: utf-8

# In[66]:


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
df=pd.read_csv('.../lgadata.csv')
df=df.dropna()
df['Code']=df['Code'].astype(int)

print(df)


# In[67]:


df_1=geopandas.read_file('.../1270055003_lga_2016_aust_shape/LGA_2016_AUST.shp')

variable_names=['GP1000']

print(df_1)


# In[69]:


m1 = spreg.OLS(
    df[['LifeExpectancy']].values, 
    df[variable_names].values,
    name_y='LifeExpectancy', 
    name_x=variable_names
)
print(m1.summary)


# In[70]:


df_1['LGA_CODE16']=df_1['LGA_CODE16'].astype(int)


df_2=df_1.merge(df, left_on='LGA_CODE16', right_on='Code',how='right')
print(df_2)


# In[47]:


df['residual'] = m1.u
medians = df.groupby(
    "Region"
).residual.median().to_frame(
    'region_residual'
)

f = plt.figure(figsize=(15,3))
ax = plt.gca()
seaborn.boxplot(
    'Region', 
    'residual', 
    ax = ax,
    data=df.merge(
        medians, 
        how='left',
        left_on='Region',
        right_index=True
    ).sort_values(
        'region_residual'), palette='bwr'
)
f.autofmt_xdate()
plt.show()


# In[74]:


knn = weights.KNN.from_dataframe(df_2, k=10)
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
df_2.assign(
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


# In[75]:


m4 = spreg.OLS_Regimes(
    df[['LifeExpectancy']].values, 
    df[variable_names].values,
    df['Region'].tolist(),
    constant_regi='many',
    cols2regi=[False]*len(variable_names),
    regime_err_sep=False,
    name_y='LifeExpectancy', 
    name_x=variable_names
)
df_2['predicted']=m4.predy
print(m4.summary)


# In[49]:


f = 'LifeExpectancy ~ ' + '+ '.join(variable_names)+ ' + Region -1'
m3 = sm.ols(f, data=df).fit()
Region_effects = m3.params.filter(like='Region')
stripped = Region_effects.index.str.strip('Region[').str.strip(']')
Region_effects.index = stripped
Region_effects = Region_effects.to_frame('fixed_effect')
ax = df_2.plot(
    color='k', alpha=0.5, figsize=(12,6)
)

df_2.merge(
    Region_effects, 
    how='left',
    left_on='Region', 
    right_index=True
).dropna(
    subset=['fixed_effect']
).plot(
    'fixed_effect', ax=ax
)
ax.set_title("Victoria Regional Fixed Effects")
plt.show()


# In[77]:


m=folium.Map(location=[-37.8,144.97],zoom_start=5)
folium.Choropleth(geo_data=df_2,data=df_2,fill_opacity=0.9,columns=['LGA_NAME16','predicted'], key_on="feature.properties.LGA_NAME16",bins=8, name='predicted',legend_name='predicted').add_to(m)

style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

pugj = folium.features.GeoJson(
    df_2,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(fields=['LGA','predicted','LifeExpectancy']))


m.add_child(pugj)
folium.LayerControl().add_to(m)
m.save('LGALife.html')


# In[ ]:





# In[ ]:




