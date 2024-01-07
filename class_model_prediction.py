import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from datetime import date, timedelta

from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Vol_Pred:
    
    def __init__(self, ticker, start=None,window=5):
        
        if start == None:
            self.data = yf.download(ticker,
                               interval='1d',
                               progress=False)[['Adj Close']]
        else:
            self.data = yf.download(ticker,
                               start=start,
                               interval='1d',
                               progress=False)[['Adj Close']]
            
        self.window =  window
        self.ticker =  ticker
        self.data = self.data.rename(columns={'Adj Close':self.ticker})
        
        self.ret = 100 * np.log(self.data/ self.data.shift(1)).dropna()
        self.vol_realizada = self.ret.rolling(window=self.window).std().dropna()
        
    def lag_features(self, df, target_name):
        df = df.copy()
        target_map = df[[target_name]]
      
        df['lag1'] = target_map.shift(1)
        df['lag2'] = target_map.shift(2)
        df['lag3'] = target_map.shift(3)
        df['lag4'] = target_map.shift(4)
        df['lag5'] = target_map.shift(5)
        df['lag6'] = target_map.shift(6)
        df['lag7'] = target_map.shift(7)
        df['lag8'] = target_map.shift(8)
        df['lag9'] = target_map.shift(9)
        df['lag10'] = target_map.shift(10)
          
        return df

    def create_model(self, type_model='randomforest',n_split=252, performance=True,
                     n_split_tss=5, test_size_tss=252, gap_tss=1):
        self.type_model =  type_model.lower()
        
        self.available_models = {'randomforest':RandomForestRegressor(),
                            'xgb':xgb.XGBRegressor(n_estimators=1000,
                                                   early_stopping_rounds = 50)}      
          
        assert type(self.type_model) == str, 'Introduce un modelo'
        
        if self.type_model not in self.available_models.keys():
            raise Exception('Sorry, the model that you typed are not in availables models')
        
        if performance:
            
            time_split =  TimeSeriesSplit(n_splits=n_split_tss,
                                   test_size=test_size_tss,
                                   gap=gap_tss)
            
            sort_data = self.lag_features(self.vol_realizada, self.ticker).dropna().sort_index()
            
            k_fold_errors = []
      
            fig = make_subplots(rows=n_split_tss,
                               cols=1)
            
            for i, (train_idx, test_idx) in enumerate(time_split.split(sort_data)):
                
                train = sort_data.iloc[train_idx]
                test = sort_data.iloc[test_idx]
                
                x_train_k = train.drop(self.ticker, axis=1)
                y_train_k = train[self.ticker]
                
                x_test_k = test.drop(self.ticker, axis=1)
                y_test_k = test[self.ticker]
                    
    
                model_fold = self.available_models[self.type_model]
                if self.type_model == 'xgb':
                    model_fold.fit(x_train_k, y_train_k,
                                   eval_set = [(x_train_k, y_train_k), (x_test_k, y_test_k)],
                                   verbose=False)
                else:
                    model_fold.fit(x_train_k, y_train_k)
                
                predictions_k = model_fold.predict(x_test_k)
                
                k_fold_errors.append(mean_squared_error(y_test_k/100,predictions_k/100))
                
                fig.add_trace(
                    go.Scatter(x=y_train_k.index, y= y_train_k.values/100,
                               name='Train data',
                              line={'color':'royalblue'}),
                row=i+1, col=1)
                
                fig.add_trace(
                    go.Scatter(x=y_test_k.index, y=y_test_k.values/100,
                               name='Test data',
                              line={'color':'green'}),
                row=i+1, col=1)
                
                fig.add_trace(
                    go.Scatter(x=y_test_k.index, y=predictions_k/100,
                              name='predictions',
                              line={'color':'firebrick',
                                    'dash':'dash'}),
                row=i+1, col=1)
                    
            fig.update_layout(title=f'Cross Validation Time Series')
                
            fig.show()
           
             
            
        else:
            pass
        
        
        self.error_kfold = np.sqrt(np.mean(k_fold_errors))

        self.df_kfold_errors = pd.DataFrame({f'fold{key}':np.sqrt(value) for key,value in enumerate(k_fold_errors)},
                                            index=['errors'])

        return fig, self.df_kfold_errors

        
    def predict_future(self):
        
        future = pd.date_range(start=date.today() + timedelta(days=1),
                               end=date.today() +  timedelta(days=1),
                               freq='1d')
        
        df_future = pd.DataFrame(index=future)
        df_future = pd.concat([self.vol_realizada, df_future])
        
        vol_future_date = self.lag_features(df_future, self.ticker)
        day_to_predict = pd.DataFrame(vol_future_date.iloc[-1]).T
        
        X = vol_future_date.dropna().drop(self.ticker, axis=1)
        y = vol_future_date.dropna()[self.ticker]
        
        model = self.available_models[self.type_model]
        if self.type_model == 'xgb':

            model.fit(X,y,
                      eval_set = [(X,y)],
                      verbose=False)
        else:
            model.fit(X,y)
            
        prediction_one_day = model.predict(day_to_predict.drop(self.ticker, axis=1))/100
        prediction_one_day = pd.DataFrame(prediction_one_day, index=future, columns=['prediction'])

        feature_importance = pd.DataFrame(data = model.feature_importances_,
                                          index = model.feature_names_in_,
                                          columns=['Importance'])
            
        return prediction_one_day, feature_importance
        