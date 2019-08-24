#!/usr/bin/env python3

import requests as rq
import pandas as pd
import numpy as np
import time
import os

from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
from requests.exceptions import ConnectTimeout, ReadTimeout
from matplotlib import pyplot as plt

# Agriculture & Forestry: 46
# Cooking & Recipes: 122
# - Fruits and Vegetables: 908

pytrends = TrendReq(hl='en-US')

def fetch_foods():
    df = pd.read_csv('./foods.csv')
    return df.Name

def fetch_countries():
    resp = rq.get('https://restcountries.eu/rest/v2/all')
    return resp.json()

def fetch_keywords(kwords, geo='US'):

    frame = '2005-01-01 2018-12-31'
    pytrends = TrendReq(hl='en-US', retries=5, backoff_factor=0.5)
    dfs = []

    for kw in kwords:

        try:
            print('Fetching keyword:', kw)
            pytrends.build_payload([kw], cat=908, geo=geo, timeframe=frame)
            df = pytrends.interest_over_time()
        except (ConnectTimeout, ReadTimeout):
            continue
        
        if not df.empty: 
            df = df.drop(columns=['isPartial'])
        else:
            print('Not enough data!')
        
        dfs.append(df)
        time.sleep(2)

    return pd.concat(dfs, axis=1)

def fetch_food_trends(kwords, geo='US'):
    frame = '2005-01-01 2018-12-31'
    
    pytrends = TrendReq(hl='en-US')
    pytrends.build_payload(kwords, cat=908, geo=geo, timeframe=frame)
    
    df = pytrends.interest_over_time()
    df = df.drop(columns=['isPartial'])

    months = df.groupby(df.index.month)
    return months.mean()

def polyfit(df, deg=5):
    return [ np.polyfit(df.index, df[k], deg) for k in df ]

def plot_food_poly(column, deg=7):
    z = np.polyfit(column.index, column, deg=deg)
    p = np.poly1d(z)
    x = np.linspace(1, 12, 100)
    y = p(x)

    plt.title(column.name)
    
    plt.subplot(3, 1, 1)
    plt.scatter(column.index, column)
    plt.plot(x, y)

    plt.subplot(3, 1, 2)
    derivate = np.polyder(p, 1)
    y = derivate(x)
    plt.plot(x, y)

    plt.subplot(3, 1, 3)
    derivate = np.polyder(p, 2) * -1
    y  = derivate(x)
    plt.plot(x, y)

def month_data_peaks(column, deg=7):
    z = np.polyfit(column.index, column, deg=deg)
    p = np.poly1d(z)
    x = np.linspace(1, 12.5, 24)
    y = p(x)

    d1 = p.deriv()
    r = d1.roots
    r = r.real
    r = r[(r >= 1) & (r <= 13)]

    peaks = p(r)
    ulimit = max(peaks)
    blimit = min(peaks)

    d2 = p.deriv(2) * -1
    prob = []

    for t in x:
        c = (p(t) - blimit) / (ulimit - blimit) if d2(t) >= 0 else 0
        c = c if c > 0 else 0
        prob.append(c)

    return pd.Series(prob, index=x, name=column.name)

def read_trends(filename):
    return pd.read_csv(filename, index_col='date', parse_dates=['date'])

def fetch_countries_trends(kwords):

    lang = 'en' # default
    kwords = fetch_foods()

    data = fetch_countries()
    data = sorted(data, key=lambda x: x['population'], reverse=True)

    for country in data:
        df = write_trends(country, kwords, lang)
        if df is None: continue
        time.sleep(10)

def write_trends(country, kwords, lang):

    try:
        print('- Fetching', country['name'])
        geo = country['alpha2Code']

        filename = './trends/{}_{}.csv'
        filename = filename.format(lang, geo)

        if os.path.exists(filename):
            print('Already fetched...')
            return

        df = fetch_keywords(kwords, geo=geo)
        df.to_csv(filename)
        return df

    except FileNotFoundError as e:
        print('ERROR!', e)
    
    except ResponseError as e:
        print('ERROR!')
        print(e.response.text)

def main():
    pass

if __name__ == '__main__':
    main()