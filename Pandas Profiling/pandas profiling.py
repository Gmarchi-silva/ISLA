# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 22:23:49 2023

@author: Igor Tavares
@subject: Projeto II - PG ADSe (ISLA Gaia)
"""

import pandas as pd

df = pd.read_csv(r'C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/cities.csv')

from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file = r'C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/p_profiling_cities.html')


df = pd.read_csv('C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/product.csv')
prof = ProfileReport(df)
prof.to_file(output_file=r'C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/p_profiling_product.html')


df = pd.read_csv('C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/sales.csv')
prof = ProfileReport(df)
prof.to_file(output_file=r'C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/p_profiling_sales.html')
