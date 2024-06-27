import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')

from tqdm import tqdm
import re
import os

from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')

# 考虑到期权和标的数据的处理方式有很多相似之处，我们可以将这些相似之处抽象出来，形成一个类，然后通过继承的方式，分别处理期权和标的数据

# 预处理基类
class preprocess:
    def __init__(self):
        pass

    def datetime_handle(self, data):
        # 按照时间戳（datetime）排序
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.sort_values('datetime', inplace=True)
        data.reset_index(drop=True, inplace=True)
        # 只保留年月日
        data['datetime'] = data['datetime'].dt.strftime('%Y-%m-%d') # 知识点：使用pandas自带的dt.strftime()函数，匹配需要的时间格式（这里是年月日）
        data['datetime'] = pd.to_datetime(data['datetime']) + pd.DateOffset(days=1) # 知识点：使用pd.DateOffset()函数，对时间进行运算（平移）
        return data
    
    def drop_columns(self, data, cols):
        data.drop(cols, axis=1, inplace=True)
        return data
    
    def check_pricing_columns(self, data):
        # 知识点：assert断言，用于判断某个条件是否满足，如果不满足则抛出异常，一般用于对代码/数据进行检查，确保其正确性
        assert data['close'].min() >= 0
        assert data['open'].min() >= 0
        assert data['high'].min() >= 0
        assert data['low'].min() >= 0
        assert data['preclose'].min() >= 0
        assert data['volume'].min() >= 0
        assert data['amount'].min() >= 0

        return data

# 期权预处理类
class option_preprocess(preprocess): # 知识点：通过类（class）的继承，可以将父类的方法和属性传递给子类，进而实现代码的复用
    def __init__(self):
        self.r = 0.025
    
    def extract_from_stockname(self, data):

        formula = r'(\d{2}ETF)(购|沽)(\d{1,2})月(\d{4})(A|B|C)?' # 知识点：正则表达式，用于匹配字符串，这里用于从期权合约名称中提取合约属性

        def parse_stockname(stockname): 
            match = re.match(formula, stockname)
            if match:
                etf_name = match.group(1)
                option_type = match.group(2)
                expiration = int(match.group(3))
                strike_price = float(match.group(4)) / 1000
                dividend_impact = 1 if match.group(5) else 0
                return pd.Series([option_type, expiration, strike_price, dividend_impact])
            else:
                return pd.Series([None, None, None, None])

        data[['option_type', 'expiration_month', 'strike_price', 'dividend_impact']] = data['StockName'].apply(parse_stockname)
        
        return data
    
    def handle_expire_date(self, data):
        # 到期日为到期月的第四个周三
        cur_date = data['datetime'].values[0]
        cur_date = pd.to_datetime(cur_date) 
        cur_month = cur_date.month
        cur_year = cur_date.year

        expire_month = data['expiration_month'].values[0]        
        if cur_month > expire_month:
            expire_year = cur_year + 1
        else:
            expire_year = cur_year

        # 计算第四个周三
        first_day_of_expire_month = datetime(expire_year, expire_month, 1)
        first_wednesday = first_day_of_expire_month + timedelta(days=(2 - first_day_of_expire_month.weekday() + 7) % 7)
        expire_date = first_wednesday + timedelta(weeks=3)
        data['expire_date'] = expire_date
        return data
        
    def compute_trading_days_to_expiration(self, data, calender):

        calender['datetime'] = pd.to_datetime(calender['datetime'])
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['expire_date'] = pd.to_datetime(data['expire_date'])

        trading_days_dict = {} # 知识点： 矢量化计算，避免使用循环，这里利用字典进行缓存，以避免重复计算
        
        # 计算到期日前的交易日天数
        def get_trading_days(start_date, end_date):
            if (start_date, end_date) in trading_days_dict:
                return trading_days_dict[(start_date, end_date)]
            trading_days = calender[(calender['date'] >= start_date) & (calender['date'] <= end_date)].shape[0]
            trading_days_dict[(start_date, end_date)] = trading_days
            return trading_days
        
        data['trading_days_to_expiration'] = data.apply(
            lambda row: get_trading_days(row['datetime'], row['expire_date']),
            axis=1
        )

        return data


    def compute_expiration_order(self, data): # 计算期权合约的期限顺序
        data['expiration_order'] = data.groupby(['option_type','datetime', 'strike_price'])['trading_days_to_expiration'].rank(ascending=True, method='dense') - 1
        return data
    
    def compute_strike_price_order(self, data, und): # 计算期权合约的行权价顺序
        '''
        输入：某日，某期限下的期权合约数据
        输出：将行权价顺序输出到df中的一列，输出总的df
        '''
        und_price = und[und['datetime'] == data['datetime'].values[0]]['close'].values[0]
        maturity = data['trading_days_to_expiration'].values[0]
        fwd_price = und_price * np.exp(self.r * maturity / 252)

        call = data[data['option_type'] == '购']
        put = data[data['option_type'] == '沽']

        call_strike_list = call['strike_price'].unique().tolist()
        put_strike_list = put['strike_price'].unique().tolist()
        call_strike_list.sort() # 知识点：sort函数，对列表进行排序，默认升序
        put_strike_list.sort(reverse=True)

        c_atm_strike = min(call_strike_list, key=lambda x:abs(x-fwd_price)) # 知识点：lambda函数和min函数的嵌套使用，用较少的代码实现了一种功能
        p_atm_strike = min(put_strike_list, key=lambda x:abs(x-fwd_price))
        assert c_atm_strike == p_atm_strike
        c_atm_strike_index = call_strike_list.index(c_atm_strike)
        p_atm_strike_index = put_strike_list.index(p_atm_strike)

        call['strike_price_order'] = call['strike_price'].apply(lambda x: call_strike_list.index(x) - c_atm_strike_index)
        put['strike_price_order'] = put['strike_price'].apply(lambda x: put_strike_list.index(x) - p_atm_strike_index)

        data = pd.concat([call, put], axis=0)
        
        return data
        
    def option_characteristics(self, data, index_data, calendar_data):
        '''
        从期权合约名称（StockName）中提取合约属性，例如认购/认沽，期限，行权价，及是否受分红影响

        例如：50ETF购6月2214A
        认购/认沽：购
        期限：6月
        期限顺序：近月（0）、次近月（1）、……
        行权价：2.214
        行权价顺序：
        '''
        data = self.datetime_handle(data)
        data = self.drop_columns(data, ['StockID', 'code', 'openinterest_change'])
        data = self.extract_from_stockname(data)
        data = data.groupby(['datetime', 'expiration_month']).apply(self.handle_expire_date).reset_index(drop=True)
        data = self.compute_trading_days_to_expiration(data, calendar_data)
        data = self.compute_expiration_order(data)
        data = data.groupby(['datetime', 'expire_date']).apply(self.compute_strike_price_order, index_data).reset_index(drop=True)
        data = self.check_pricing_columns(data)
        return data
    
# 标的预处理类
class index_preprocess(preprocess):
    def __init__(self):
        pass
    
    def index_characteristics(self, data):
        '''
        处理标的数据
        '''
        data = self.datetime_handle(data)
        data = self.drop_columns(data, ['StockID', 'StockName'])
        data = self.check_pricing_columns(data)
        return data

