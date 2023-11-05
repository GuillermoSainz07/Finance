# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:45:54 2023

@author: W10 2023
"""

import yfinance as yf
import numpy as np
import pandas as pd


def get_historical_params(tickers, start=None, end=None):
    
    assert start != None, 'Agrega una fecha de inicio'
    if end != None:
        prices = yf.download(tickers=tickers,
                             start=start,
                             end=end,
                             progress=False)[['Adj Close']]
    else:
        prices = yf.download(tickers=tickers,
                             start=start,
                             progress=False)[['Adj Close']]
    
    ret_close = ((prices - prices.shift())/ prices.shift()).dropna()
    ret_close = pd.DataFrame(ret_close)
    
    vols_month = ret_close.groupby(pd.Grouper(freq='M')).std()
    
    
    #v0 theta
    vol_inicial = vols_month.iloc[0]
    vol_final = vols_month.iloc[-1]
    volvol = np.std(vols_month)
    
    if str(vol_final.values[0]) == 'nan':
        vol_final = vols_month.iloc[-2]
    
    # rho
    rent_log_vol = np.log(vols_month / vols_month.shift()).dropna()
    
    means_month = prices.iloc[:-1].groupby(pd.Grouper(freq='M')).mean()
    
    ret_log_month = np.log(means_month / means_month.shift()).dropna()

    
    vol_mean_month = pd.DataFrame(np.column_stack((rent_log_vol.values, ret_log_month.values)))
    
    # rho
    rho = vol_mean_month.corr().iloc[0,1]    
    
    return vol_inicial.values[0], vol_final.values[0], volvol.values[0], rho


v0, vbar, sigma, rho = get_historical_params(tickers='^MXX',
                      start='2018-10-25')


