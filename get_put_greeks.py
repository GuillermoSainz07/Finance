# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 11:00:33 2023

@author: W10 2023
"""
import numpy as np
from scipy.stats import norm

def get_put_greeks(st,k,r,sigma,tau,greek='delta'):
    
    N = norm(loc=0, scale=1)
    d1 = (np.log(st/k) + (r + 0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    
    greeks_formulas = {'delta': -N.cdf(-d1),
                       'rho':(-tau*k*np.exp(-r*tau))*N.cdf(-d2),
                       'theta':((-st*N.pdf(d1)*sigma)/(2*np.sqrt(tau))) \
                           + r*k*np.exp(-r*tau)*N.cdf(-d2) ,
                       'gamma':(N.pdf(d1))/(st*sigma*np.sqrt(tau)),
                       'vega':st*N.pdf(d1)*np.sqrt(tau)}
    
    return greeks_formulas[greek]
    
get_put_greeks(100,100, 0.05, 0.2, 1)

