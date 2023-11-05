# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:25:22 2023

@author: W10 2023
"""

import matplotlib.pyplot as plt
import numpy as np
from heston_functions import ChFHestonModel, CallPutOptionPriceCOSMthd, OptionType
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as implied_vol_fuction


class Sensitivities:
    # Los parametros los defini como un diccionario
    def __init__(self,params={'s0':100,
                             'r':0.05,
                             'tau':2.0,
                             'K':np.linspace(40,200,25),
                             'N':500,
                             'L':5,
                             'kappa':0.1,
                             'gamma':0.1,
                             'vbar':0.1,
                             'rho':-0.75,
                             'v0':0.05}):
        self.s0 = params['s0']
        self.r = params['r']
        self.tau = params['tau']
        self.K = params['K']
        self.kappa = params['kappa']
        self.gamma = params['gamma']
        self.vbar = params['vbar']
        self.rho = params['rho']
        self.v0 = params['v0']
        self.params = params
        self.N = 500
        self.L = 5
        
    def plot_sensitivities(self, type_op='call',variable='kappa', interval=None):
        if type_op == 'put':
            CP  = OptionType.PUT
        else:
            CP = OptionType.CALL
        # La variable intervalo sera una lista compuesta por [min,max,n]
        
        interval = interval
        assert type(interval) is list, 'Agrega un intervalo de variacion para tu variable'
        assert len(interval) == 3,  'Agrege minimo, maximo y cantidad'
            
            
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        
        interval_variable = np.linspace(interval[0],interval[1],interval[2])      
        
        # La volatilidad implicita del eje y esta en terminos de % por eso el *100
        if variable == 'kappa':
            for k in interval_variable:
                cf = ChFHestonModel(self.r, self.tau, k,self.gamma,self.vbar,self.v0,self.rho)
                hp = CallPutOptionPriceCOSMthd(cf,CP,
                                      self.s0,
                                      self.r,
                                      self.tau,
                                      self.K,
                                      self.N,self.L)
                
                imp_vol = implied_vol_fuction(hp, self.s0, self.K, self.tau, self.r, flag='c')
                
                ax[0].plot(self.K,hp, label=f'$\\kappa = ${k:.2f}')
                ax[0].set_xlabel('$K$')
                ax[0].set_ylabel('Price')
                
                ax[1].plot(self.K, imp_vol * 100)
                ax[1].set_ylabel('$\\sigma_{imp}$')
                ax[1].set_xlabel('$K$')
            
        elif variable == 'gamma':
            for g in interval_variable:
                cf = ChFHestonModel(self.r, self.tau, self.kappa, g,self.vbar,self.v0,self.rho)
                hp = CallPutOptionPriceCOSMthd(cf,CP,
                                      self.s0,
                                      self.r,
                                      self.tau,
                                      self.K,
                                      self.N,self.L)
                
                imp_vol = implied_vol_fuction(hp, self.s0, self.K, self.tau, self.r, flag='c')
                
                ax[0].plot(self.K,hp, label=f'$\\gamma = ${g:.2f}')
                ax[0].set_xlabel('$K$')
                ax[0].set_ylabel('Price')
                
                ax[1].plot(self.K, imp_vol * 100)
                ax[1].set_ylabel('$\\sigma_{imp}$')
                ax[1].set_xlabel('$K$')
            
        elif variable == 'vbar':
            for vb in interval_variable:
                cf = ChFHestonModel(self.r, self.tau, self.kappa,self.gamma,vb,self.v0,self.rho)
                hp = CallPutOptionPriceCOSMthd(cf,CP,
                                      self.s0,
                                      self.r,
                                      self.tau,
                                      self.K,
                                      self.N,self.L)
                
                imp_vol = implied_vol_fuction(hp, self.s0, self.K, self.tau, self.r, flag='c')
                
                ax[0].plot(self.K,hp, label=f'$v = ${vb:.2f}')
                ax[0].set_xlabel('$K$')
                ax[0].set_ylabel('Price')
                
                ax[1].plot(self.K, imp_vol * 100)
                ax[1].set_ylabel('$\\sigma_{imp}$')
                ax[1].set_xlabel('$K$')
            
            
        elif variable == 'rho':
            for ro in interval_variable:
                cf = ChFHestonModel(self.r, self.tau,self.kappa,self.gamma,self.vbar,self.v0,ro)
                hp = CallPutOptionPriceCOSMthd(cf,CP,
                                      self.s0,
                                      self.r,
                                      self.tau,
                                      self.K,
                                      self.N,self.L)
                
                imp_vol = implied_vol_fuction(hp, self.s0, self.K, self.tau, self.r, flag='c')
                
                ax[0].plot(self.K,hp, label=f'$\\rho = ${ro:.2f}')
                ax[0].set_xlabel('$K$')
                ax[0].set_ylabel('Price')
                
                ax[1].plot(self.K, imp_vol * 100)
                ax[1].set_ylabel('$\\sigma_{imp}$')
                ax[1].set_xlabel('$K$')
        
        elif variable == 'v0':
            for v in interval_variable:
                cf = ChFHestonModel(self.r, self.tau, self.kappa,self.gamma,self.vbar,v,self.rho)
                hp = CallPutOptionPriceCOSMthd(cf,CP,
                                      self.s0,
                                      self.r,
                                      self.tau,
                                      self.K,
                                      self.N,self.L)
                
                imp_vol = implied_vol_fuction(hp, self.s0, self.K, self.tau, self.r, flag='c')
                
                ax[0].plot(self.K,hp, label=f'$v_0 = ${v:.2f}')
                ax[0].set_xlabel('$K$')
                ax[0].set_ylabel('Price')
                
                ax[1].plot(self.K, imp_vol * 100)
                ax[1].set_ylabel('$\\sigma_{imp}$')
                ax[1].set_xlabel('$K$')
                
        ax[0].grid()
        ax[1].grid()
        ax[0].legend()