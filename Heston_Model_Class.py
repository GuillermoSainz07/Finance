# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:58:37 2023

@author: W10 2023
"""

import numpy as np
import pandas as pd
# Heston Functions
from heston_functions import (ChFHestonModel, 
                              CallPutOptionPriceCOSMthd, 
                              OptionType,
                              clean_data_forward,
                              BS_Call_Put_Option_Price)
from py_vollib_vectorized.implied_volatility import vectorized_implied_volatility as implied_vol_fuction
# Data visualization
import matplotlib.pyplot as plt
# Optimization
from scipy.optimize import minimize, differential_evolution
from time import time
# Data missing handle
from sklearn.impute import KNNImputer
# Error measure
from sklearn.metrics import mean_squared_error
# Historical params
from get_params import get_historical_params



class Heston_model_forward:
    def __init__(self, df):
        self.df = clean_data_forward(df)
        self.data = self.df[['strike_price','free_rate',
                        'maturities','Volatilidad',
                        'valor_mexder','spot_price','CP']]
        
        self.data_call = self.data.query('CP == 0')
        self.data_put = self.data.query('CP == 1')
        
        # Data general
        self.S0 = self.data.spot_price
        self.r = self.data.free_rate
        self.tau = self.data.maturities
        self.K = self.data.strike_price
        self.P = self.data.valor_mexder
        
        self.N = 500
        self.L = 5
        
        # Data call
        self.S0_call = self.data_call.spot_price
        self.r_call = self.data_call.free_rate
        self.tau_call = self.data_call.maturities
        self.K_call = self.data_call.strike_price
        self.P_call = self.data_call.valor_mexder
        
        # Data put
        self.S0_put = self.data_put.spot_price
        self.r_put = self.data_put.free_rate
        self.tau_put = self.data_put.maturities
        self.K_put = self.data_put.strike_price
        self.P_put = self.data_put.valor_mexder
        
           
    def get_optimized_params(self,
                             start='2018-10-25',
                             historical_init=True, handle_nan='knn', optimizer='minimize'):
        
        
        if historical_init == True:
            v0_init, vbar_init, sigma_init, rho_init = get_historical_params(tickers='^MXX',
                                  start=start)
            
            params = {"v0": {"x0": v0_init, "lbub": [1e-3,2]}, 
              "kappa": {"x0": 0.5, "lbub": [1e-3,10]},
              "vbar": {"x0": vbar_init, "lbub": [1e-3,2]},
              "gamma": {"x0": sigma_init, "lbub": [1e-2,3]},
              "rho": {"x0": rho_init, "lbub": [-0.999,0.999]}}
            
        # Naive init
        else:
            params = {"v0": {"x0": 0.05, "lbub": [1e-3,2]}, 
              "kappa": {"x0": 0.5, "lbub": [1e-3,10]},
              "vbar": {"x0": 0.05, "lbub": [1e-3,2]},
              "gamma": {"x0": 0.5, "lbub": [1e-2,3]},
              "rho": {"x0": -0.5, "lbub": [-0.999,0.999]}}
            
        
        def SqErr(x):
            CP  = OptionType.CALL

            v0, kappa, vbar, gamma, rho = [param for param in x]
                
            err = np.sum([(P_i - CallPutOptionPriceCOSMthd(ChFHestonModel(rf, t, kappa, gamma, vbar, v0, rho),CP,
                                                           s0_i,
                                                           rf,
                                                           t,
                                                           [strike],
                                                           self.N,
                                                           self.L))**2 / len(self.P_call) \
                         for P_i, rf, t, s0_i, strike in zip(self.P_call,self.r_call,
                                                             self.tau_call,self.S0_call,
                                                             self.K_call)])
                  
            return err
        
        x0 = [param["x0"] for key, param in params.items()]
        bnds = [param["lbub"] for key, param in params.items()]
        
        if optimizer == 'minimize':
            start = time()
            result = minimize(SqErr, x0, bounds=bnds) 
            self.v0, self.kappa, self.vbar, self.gamma, self.rho = [param for param in result.x]
            end = time()
            
        elif optimizer == 'diff_evol':
            start = time()
            result = differential_evolution(SqErr,bounds=bnds) 
            self.v0, self.kappa, self.vbar, self.gamma, self.rho = [param for param in result.x]
            end = time()
            
        else:
            print('Eliga un optimizador valido')
            
        self.time_run = end - start
        
        heston_prices_call = []
        heston_prices_put = []
        CP_call  = OptionType.CALL
        CP_put = OptionType.PUT
        
        for i in range(len(self.S0_call)):
            instancia = self.data_call.iloc[i]
            
            cf = ChFHestonModel(instancia.free_rate, instancia.maturities,
                                self.kappa,self.gamma,self.vbar,self.v0,self.rho)
            
        
            price_heston_call = CallPutOptionPriceCOSMthd(cf,CP_call,
                                                     instancia.spot_price,
                                                     instancia.free_rate,
                                                     instancia.maturities,
                                                     [instancia.strike_price],
                                                     self.N,self.L)
            
            price_heston_put = CallPutOptionPriceCOSMthd(cf,CP_put,
                                             instancia.spot_price,
                                             instancia.free_rate,
                                             instancia.maturities,
                                             [instancia.strike_price],
                                             self.N,self.L)
            
            
            heston_prices_call.append(price_heston_call[0][0])
            heston_prices_put.append(price_heston_put[0][0])
                
        hestons_prices_call = np.array(heston_prices_call)
        hestons_prices_put = np.array(heston_prices_put)
        
        hestons_prices_call = np.where(hestons_prices_call < 0,0,heston_prices_call)
        hestons_prices_put = np.where(hestons_prices_put < 0,0,heston_prices_put)
        
        imp_vol_call = implied_vol_fuction(hestons_prices_call,
                                           self.S0_call,self.K_call,
                                           self.tau_call,self.r_call, flag='c')
        imp_vol_put = implied_vol_fuction(hestons_prices_put,
                                          self.S0_put,self.K_put,
                                          self.tau_put,self.r_put, flag='p')
        
        heston_optimized_params = self.data.copy()
        heston_optimized_params['imp_vol_heston'] = np.concatenate([imp_vol_call.values,
                                                                    imp_vol_put.values])
        heston_optimized_params['heston_prices'] = np.concatenate([hestons_prices_call,
                                                                   hestons_prices_put])
    
        heston_optimized_params['imp_vol_heston'] = (heston_optimized_params['imp_vol_heston']
                                                     .replace([0, np.nan])
                                                     )
        
        
        #handling nan and zeros values
        if handle_nan == 'knn':
            X = heston_optimized_params[['strike_price','free_rate',
                                         'maturities','heston_prices',
                                         'imp_vol_heston','spot_price']]
            imputer = KNNImputer(n_neighbors=4).set_output(transform='pandas')
            matrix_without_nan = imputer.fit_transform(X)
            
            heston_optimized_params['imp_vol_heston'] = matrix_without_nan.imp_vol_heston
            
        elif handle_nan == 'interpolate':
            heston_optimized_params['imp_vol_heston'] = (heston_optimized_params['imp_vol_heston']
                                                         .interpolate(method='polynomial', order=2))
        else:
            print('Selecciona un metodo para manejo de valores nulos (knn, interpolacion)')
        
        
        new_heston_prices = np.concatenate([BS_Call_Put_Option_Price(CP_call, self.S0_call,self.K_call,
                                 heston_optimized_params.query('CP == 0').imp_vol_heston,
                                 self.tau_call, self.r_call),
                                    BS_Call_Put_Option_Price(CP_put, self.S0_put,self.K_put,
                                                             heston_optimized_params.query('CP == 1').imp_vol_heston,
                                                             self.tau_put, self.r_put)])
        
        heston_optimized_params['heston_prices'] = new_heston_prices
         
        
        self.optimized_data = heston_optimized_params
        
        error = mean_squared_error(heston_optimized_params.valor_mexder,
                                   heston_optimized_params.heston_prices)
        
        
        df_optimized_parms = pd.DataFrame({'$$v_{0}$$': self.v0,
                                           '$$\kappa$$': self.kappa,
                                           '$$\\bar{v}$$': self.vbar,
                                           '$$\gamma$$': self.gamma,
                                           '$$\rho$$': self.rho,
                                           'Time in seconds': self.time_run,
                                           'MSE': error,
                                           'RMSE': np.sqrt(error)},
                                          index=['$$L_{mse}$$'])
        
        
        
        return df_optimized_parms, heston_optimized_params
    
    def plot_surface_volatility(self):
        vol_surf_call = (self.optimized_data.query('CP == 0')
                         .pivot_table(columns='maturities',index='strike_price',
                                      values='imp_vol_heston'))
        
        vol_surf_put = (self.optimized_data.query('CP == 1')
                         .pivot_table(columns='maturities',index='strike_price',
                                      values='imp_vol_heston'))
        
        x_imp_call = vol_surf_call.columns.values
        x_imp_put = vol_surf_put.columns.values
        
        y_imp_call = vol_surf_call.index.values
        y_imp_put =vol_surf_put.index.values
        
        
        X_imp_call, Y_imp_call = np.meshgrid(x_imp_call,y_imp_call)
        X_imp_put, Y_imp_put = np.meshgrid(x_imp_put,y_imp_put)
        Z_imp_call = vol_surf_call.values
        Z_imp_put = vol_surf_put.values
        
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        
        ax.plot_surface(X_imp_call,Y_imp_call,Z_imp_call, cmap='inferno')
        
        ax.set_title('CALL OPTIONS')
        ax.set_xlabel('Maturities')
        ax.set_ylabel('K')
        ax.set_zlabel('$\\sigma_{imp}$')
        
        ax2.plot_surface(X_imp_put,Y_imp_put,Z_imp_put, cmap='inferno')
        ax2.set_title('PUT OPTIONS')
        ax2.set_xlabel('Maturities')
        ax2.set_ylabel('K')
        ax2.set_zlabel('$\\sigma_{imp}$')
        
        fig.suptitle('Volatility Surface')
    
    
# Load data
df = pd.read_csv('data_heston.csv')
# Run model
model = Heston_model_forward(df)
# Get prices and implied vol
df_params, df_data = model.get_optimized_params(historical_init=True, handle_nan='knn') #interpolate
# Plotting vol surface
model.plot_surface_volatility()
        
        
        
        
        