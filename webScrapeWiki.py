# -*- coding: utf-8 -*-

# packages to load

import request as urr
#from urllib import error
from bs4 import BeautifulSoup
import pandas as pd

#function to scrape wiki table
def wikiscrape():
    req = urr.Request('https://en.wikipedia.org/wiki/List_of_data_breaches')
    content = urr.urlopen(req)
    cs = content.info().get_content_charset()  # get charset of webpage

    html = content.read().decode(cs)  # consists of html source code

    print("Downloading...")

    soup = BeautifulSoup(html, 'html.parser')

    data = []

    tabclasses = []

    tables = soup.findAll("table")  # "class" : "wikitable"

    for index, tab in enumerate(tables):
        data.append([])
        tabclasses.append(tab.attrs)
        for ind, items in enumerate(tab.find_all("tr")):
            cols = items.find_all(["th", "td"])
            cols = [ele.text.strip() for ele in cols]
            data[index].append([ele for ele in cols if (ele != [])])

    df = pd.DataFrame()

    for rows in data:
        df = df.append(rows)
        df = df.replace(r'\n', '', regex=True)

    print('N dimensions of data {}'.format(df.shape))
    print('Function call complete')
    return df