# -*- coding: utf-8 -*-
"""
This program takes sector-specific model parameters to calibrate individual
partial equilibrium models and then use those models to produce counterfactual 
results without the increased tariffs.

Each year and sector is estimated with an individual perfect competition PE
model. 
"""
### Import libraries ###
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

### Define sectors and regions of interest ###

# NAICS 4-digit sectors
toplist = ["3152","3344","3341","3371","3363","3359","3399","3343","3339","3261"]
sorterIndex = dict(zip(toplist, range(len(toplist))))

# Countries
sourcelist = ['China','United States','Rest of world']
sorterIndex2 = dict(zip(sourcelist, range(len(sourcelist))))

### Import and combine data from Stata ###
df = pd.read_stata('PE_trade_data.dta')
df_dom = pd.read_stata('PE_domestic_data.dta')
df_sigma = pd.read_stata('PE_sigma.dta')

# Use closest domestic naics for apparel and footwear
# (3152 and 3162 are not included, so use the 3-digit aggregation for each)
df_dom.loc[(df_dom['source']=='United States')&(df_dom['naics4']=='3150'),'naics4']='3152'
df_dom.loc[(df_dom['source']=='United States')&(df_dom['naics4']=='3160'),'naics4']='3162'

# Combine data
df = df.append(df_dom)
df = df.merge(df_sigma,on="naics4",how='outer')

# Calculate expenditure (cv times 1+duty)
# (Note: This only includes section 301 duties. All other duties are implicitly
# taken into account in the market share information and are held constant
# in this modeling exercise.)
df['val']=df['cv']*(1+df['duty'])

### Set supply parameters ###
df['supply_elasticity'] = np.inf
df.loc[df['source']=="United States",'supply_elasticity']=1.0

df['supply_shifter'] = 1.0
df.loc[df['source']=="United States",'supply_shifter']=df['val']

### Calculate demand shifters using market shares ###
df['K'] = df.groupby(['year','naics4'])['val'].transform('sum')
df['a'] = df['val']/df['K']

### Calculate initial price indices ###
df['p'] = 1/(1+df['duty'])
df['pduty'] = df['p']*(1+df['duty'])

df['Pin'] = df['a']*(df['pduty'])**(1-df['sigma'])
df['Pin'] = df.groupby(['year','naics4'])['Pin'].transform('sum')
df['Pin'] = df['Pin']**(1/(1-df['sigma']))

### Set nested CES demand function ###
df['q'] = df['a']*df['K']*(df['pduty'])**(-df['sigma'])*df['Pin']**(df['sigma']-1.0)

### Calculate counterfactual price index ###
df['p_cf'] = df['p']

df['PinF_cf'] = df['a']*(df['p_cf'])**(1-df['sigma'])
df['PinF_cf'] = df.groupby(['year','naics4','source'])['PinF_cf'].transform('sum')
df.loc[df['source']=="United States",'PinF_cf']=0.0
df['PinF_cf'] = df.groupby(['year','naics4'])['PinF_cf'].transform('sum')


### Now solve for domestic sector price ###
def dsolve(pd0,Pf,ad,K,sigma,ss,se):
    def xd2(pd):
        # Calculate excess demand for guessed price
        P = (Pf+ad*pd**(1-sigma))**(1/(1-sigma))
        d = ad*K*(pd**(-sigma))*(P**(sigma-1))
        s = ss*pd**se
        return (d-s)**2
    return fsolve(xd2,pd0)

df.loc[df['source']=="United States",'p_cf'] = df.loc[df['source']=="United States"].apply(lambda row : dsolve(row['p'],row['PinF_cf'],row['a'],row['K'],row['sigma'],row['supply_shifter'],row['supply_elasticity'])[0],1)

### Apply those solved prices to find remaining counterfactual stats ###
df['Pin_cf'] = df['a']*(df['p_cf'])**(1-df['sigma'])
df['Pin_cf'] = df.groupby(['year','naics4'])['Pin_cf'].transform('sum')
df['Pin_cf'] = df['Pin_cf']**(1/(1-df['sigma']))

df['q_cf'] = df['a']*df['K']*(df['p_cf'])**(-df['sigma'])*df['Pin_cf']**(df['sigma']-1)
df['val_cf'] = df['q_cf']*df['p_cf']

# Check excess demand to make sure solution is good.
df['supply_cf'] = df['q_cf']
df.loc[df['source']=="United States",'supply_cf'] = df['supply_shifter']*df['p_cf']**df['supply_elasticity']
df['excess_demand_cf']=df['q_cf']-df['supply_cf']

### Results summary ###
results = df[['naics4','source','year','q','q_cf','val','val_cf','Pin','Pin_cf']].loc[df['naics4'].isin(toplist)].copy()
results.loc[(results['source']!="United States")&(results['source']!="China"),'source'] = "Other"
results = results.loc[(results['year']==2021)].groupby(['source','naics4','year']).agg('sum').reset_index()
results['p'] = results['val']/results['q']
results['p_cf'] = results['val_cf']/results['q_cf']
results['p_increase'] = 100*(results['p']-results['p_cf'])/results['p_cf']
results['val_increase'] = 100*(results['val']-results['val_cf'])/results['val_cf']
results['Pin_increase'] = 100*(results['Pin']-results['Pin_cf'])/results['Pin_cf']
results = results[['naics4','source','p_increase','val_increase','Pin_increase']]

results = results.pivot(index='naics4',columns='source',values=['p_increase','val_increase','Pin_increase']).reset_index()

results['naics4_rank'] = results['naics4'].map(sorterIndex)
results.sort_values('naics4_rank',inplace=True)
results.drop('naics4_rank',axis=1,inplace=True)
results.to_excel('Results/results.xlsx',sheet_name="results_summary")

### Sector-specific results tables ###
sresults = df[['naics4','source','year','q','q_cf','val','val_cf']].loc[df['naics4'].isin(toplist)].copy()
sresults.loc[(sresults['source']!="United States")&(sresults['source']!="China"),'source'] = "Rest of world"
sresults = sresults.groupby(['source','naics4','year']).agg('sum').reset_index()
sresults = sresults.loc[sresults['year']!=2022]
sresults['p'] = sresults['val']/sresults['q']
sresults['p_cf'] = sresults['val_cf']/sresults['q_cf']
sresults['p_change']=100*(sresults['p']-sresults['p_cf'])/sresults['p_cf']
sresults['q_change']=100*(sresults['q']-sresults['q_cf'])/sresults['q_cf']
sresults['val_change']=100*(sresults['val']-sresults['val_cf'])/sresults['val_cf']
sresults['naics4_rank'] = sresults['naics4'].map(sorterIndex)
sresults['source_rank'] = sresults['source'].map(sorterIndex2)

sresults_p=sresults[['naics4','naics4_rank','source','source_rank','year','p_change']].pivot(index=['naics4','naics4_rank','source','source_rank'],columns='year',values='p_change').reset_index()
sresults_p.sort_values(['naics4_rank','source_rank'],inplace=True)
sresults_p.drop(['naics4_rank','source_rank'],axis=1,inplace=True)

sresults_val=sresults[['naics4','naics4_rank','source','source_rank','year','val_change']].pivot(index=['naics4','naics4_rank','source','source_rank'],columns='year',values='val_change').reset_index()
sresults_val.sort_values(['naics4_rank','source_rank'],inplace=True)
sresults_val.drop(['naics4_rank','source_rank'],axis=1,inplace=True)

sresults_p.to_excel('Results/results_prices.xlsx',sheet_name="results_prices")
sresults_val.to_excel('Results/results_values.xlsx',sheet_name="results_values")

### Report sigma values ###
sigmas = df.loc[df['naics4'].isin(toplist),['naics4','sigma','sigma_se']].drop_duplicates()
sigmas['naics4_rank']=sigmas['naics4'].map(sorterIndex)
sigmas.sort_values('naics4_rank',inplace=True)
sigmas = sigmas.drop('naics4_rank',axis=1)
sigmas.to_excel('Results/sigmas.xlsx',sheet_name='sigma')