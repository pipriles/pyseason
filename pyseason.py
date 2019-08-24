#!/usr/bin/env python3

import requests as rq
import pandas as pd
import re
import time
import multiprocessing as mp
import numpy as np
import util

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from functools import wraps
from sklearn import tree
from sklearn.model_selection import cross_val_score

URL = 'https://www.seasonalfoodguide.org/'

MONTHS = ('January', 'February', 'March', 'April', 
    'May', 'June', 'July', 'August', 'September',
    'October', 'November', 'December')

def retry_on_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 2
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print('[ERROR]', e)
                if retries <= 0: break
                retries -= 1
    return wrapper

def fetch_foods(html):
    soup = BeautifulSoup(html, 'html.parser') 
    opts = soup.select('select[name=currentVeg] option')
    food = { o['value'] for o in opts }
    food.discard('blank')
    return food

def fetch_states(html):
    soup = BeautifulSoup(html, 'html.parser') 
    opts = soup.select('select[name=sfgState] option')
    food = { o['value'] for o in opts }
    food.discard('blank')
    return food

def select_option(driver, name, value):
    # Select by food name
    elem = driver.find_element_by_name(name)
    elem = Select(elem)
    elem.select_by_value(value)

def fetch_state_data(driver, st):
    # Select by state name
    select_option(driver, 'sfgState', st)
    driver.implicitly_wait(1)

    html = driver.page_source
    foods = fetch_foods(html)
    print('- Fetching state', st)

    for x in foods:
        print('Fetching', x)
        data = fetch_months(driver, x)
        data['Food']  = x
        data['State'] = st
        yield data

def fetch_months(driver, food):
    # Select by food name
    select_option(driver, 'currentVeg', food)
    driver.implicitly_wait(1)

    # Initialize months
    data = dict.fromkeys(MONTHS, 0)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    elem = soup.find(id='wheel-months-list')
    months = elem.get_text(strip=True).split(', ') if elem else []
    for k in months:
        data[k] = 1

    return data

def merge_months(df):
    
    results = []
    
    for month in MONTHS:
        
        whole = df[month].astype(bool)

        name = 'Early {}'.format(month)
        s = pd.Series(0.0, index=df.index, name=name)
        e = df.get(name, s)
        e = e.fillna(0).astype(bool)
        s = (e | whole).astype(float)
        s.name = name
        results.append(s)

        name = 'Late {}'.format(month)
        s = pd.Series(0.0, index=df.index, name=name)
        l = df.get(name, s)
        l = l.fillna(0).astype(bool)
        s = (l | whole).astype(float)
        s.name = name
        results.append(s)

    return results

def expand_months(months):
    for index in range(len(months)):
        value = months[index]
        next_ = months[(index + 1) % len(months)]
        yield int((value == 2) or (value == 1 and next_ <= 0))
        yield int((value == 2) or (value == 1 and next_  > 0))

def biweek(months):
    weeks = expand_months(months)
    record = {}
    for x in util.MONTHS:
        record['Early %s' % x] = next(weeks)
        record['Late %s'  % x] = next(weeks)
    return record

def fix_months(df):
    cols  = ['Country', 'Food']
    bweek = [ biweek(x) for x in df[util.MONTHS].values ]
    expanded = pd.DataFrame(bweek)
    return pd.concat([df[cols], expanded], axis=1)

def lget(l, index):
    return l[index] if index < len(l) else None

def extract_location(soup):
    path = soup.select('#fake a.redglow')
    path = [ x.get_text(strip=True) for x in path ]
    return { 
        'Region':  lget(path, 1),
        'Country': lget(path, 2),
        'State':   lget(path, 3) 
    }

def fetch_city_weather(url):
    
    print(url)

    resp = rq.get(url, params={'units': 'metric'})
    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')
    
    column = soup.find_all(id='h4font')
    tables = soup.select('table[width="650"]')

    location = extract_location(soup)
    h1 = soup.select_one('.text-shadow-fix > h1')
    city = h1.get_text(strip=True)
    city = city.partition(',')[0]
    
    for name, t in zip(column, tables):
        # Get months values
        values = t.select('tr[bgcolor="white"] > td')
        values = [ x.get_text(strip=True) for x in values ]

        data = { k: v for k, v in zip(MONTHS, values[2:]) }
        data['Data Unit'] = values[0]
        data['Annual'] = values[1]
        data['Column'] = name.get_text(strip=True)
        data['City']   = city

        yield { **data, **location }

def monthly_all(href):
    BASE = 'https://www.weatherbase.com'
    href = re.sub(r'weather\.php3', r'weatherall.php3', href)
    return BASE + href

def fetch_state_weather(url):

    resp = rq.get(url, params={'units': 'metric'})
    if not 'city.php' in resp.url: 
        print('Redirect!')
        return

    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')

    cities = soup.select('a.redglow')
    cities = [ monthly_all(a['href']) for a in cities ]

    # for x in cities:
    #   yield from fetch_city_weather(x)
    yield from fetch_parallel(process_weather, cities)

def fetch_country_weather(url):

    match = re.search(r'/state\.php', url)
    if match is None:
        yield from fetch_state_weather(url)
        return

    resp = rq.get(url, params={'units': 'metric'})
    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')

    BASE = 'https://www.weatherbase.com'

    states = soup.select('a.redglow')
    states = [ BASE + a['href'] for a in states ]

    for s in states:
        yield from fetch_state_weather(s)

def fetch_region_weather(url):

    resp = rq.get(url, params={'units': 'metric'})
    html = resp.text
    soup = BeautifulSoup(html, 'html.parser')

    BASE = 'https://www.weatherbase.com'

    countries = soup.select('a.redglow')
    countries = [ BASE + a['href'] for a in countries ]

    for x in countries:
        yield from fetch_country_weather(x)

@retry_on_error
def process_weather(data):
    return list(fetch_city_weather(data))

def fetch_parallel(func, data):
    with mp.Pool(8) as p:
        yield from p.imap_unordered(func, data)

def process_weather_df(df):
    df = df.replace(r'---', np.nan)
    df = df.astype({ k: 'float64' for k in util.MONTHS })
    df = df.astype({ 'Annual': 'float64' })
    return df

def load_weather():
    df = pd.read_csv('./weather.csv')
    df = process_weather_df(df)
    df['State Code'] = df.State.map(util.US_STATES_CODES)
    return df

def load_seasons():
    se = pd.read_csv('./seasons.csv')
    se = se.set_index(['Food', 'State'])
    se = season_mean(se)
    return se

def weather_season_data():
    df = load_weather()
    se = load_seasons()

    g = df.groupby(['State Code', 'Column'])
    mean = g.mean()
    mean = mean[util.MONTHS].T

    for food, data in se.groupby(level=0):
        
        records = []
        codes = data.index.get_level_values(1)
        
        for k in codes:
            try:
                year = mean[k][util.ALTERNATIVE_DATA].dropna()
            except KeyError:
                continue
            crop = [ data.loc[(food, k)][i] for i in year.index ]
            months = year.values.tolist()
            record = { 'location': k, 'season': crop, 
                'params': months }
            records.append(record)

        yield (food, records)

def extract_parameters(mean, data):
    for code in data.State:
        mean.loc[code].loc[util.WEATHER_DATA]

def season_mean(se):
    indexs = range(0, len(se.columns), 2)
    months = [ se.iloc[:, i:i+2].mean(axis=1) for i in indexs ]
    for x, name in zip(months, util.MONTHS):
        x.name = name
    return pd.concat(months, axis=1)

def load_country_params(df):

    df = process_weather_df(df)
    g = df.groupby(['Country', 'Column'])
    mean = g.mean()
    mean = mean[util.MONTHS]

    for country, group in mean.groupby(level=0):
        try:
            data = group.loc[country]
            data = data.T[util.ALTERNATIVE_DATA]
        except KeyError:
            continue

        data.interpolate(limit_direction='both', inplace=True)
        months = data.values.tolist()
        yield (country, months)

def predict_country_season(clfs, df):

    data = load_country_params(df)
    for country, months in data:
        for food in clfs:
            crop = clfs[food].predict(months)
            d = { k: v for k, v in zip(util.MONTHS, crop) }
            d['Food'] = food
            d['Country'] = country
            yield d

def train_models():

    data = weather_season_data()
    clfs = {}

    for food, locations in data:
        
        X = []
        Y = []
        for l in locations:
            X.extend(l['params'])
            Y.extend(l['season'])
        
        print('-', food)
        print('X size:', len(X))
        print('Y size:', len(Y))

        enc = { 0.0: 0, 0.5: 1, 1.0: 2 }
        Y = [ enc[x] for x in Y ]

        clf = tree.DecisionTreeClassifier(
            max_depth=7, min_samples_leaf=1)
        
        scores = cross_val_score(clf, X, Y, cv=5)
        print('Accuracy:', scores.mean())
        
        clf.fit(X, Y)
        clfs[food] = clf
        
        print('---------')

    return clfs

def main():
    driver = webdriver.Chrome()
    driver.get(URL)
    driver.implicitly_wait(1)

    html = driver.page_source
    states = fetch_states(html)
    results = []
    
    for st in states:
        for x in fetch_state_data(driver, st):
            results.append(x)

    return results

if __name__ == '__main__':
    main()