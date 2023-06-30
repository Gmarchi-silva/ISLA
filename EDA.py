import pandas as pd

df = pd.read_csv(r'C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/sales.csv')

df['date'].head()
df['date'].tail()
df.describe().transpose()
df.info()
df.duplicated().sum()
df[['sales','store_id', 'product_id', 'date', 'revenue', 'stock', 'price']].query("sales == 0")
df_zero_sales = df[['sales','store_id', 'product_id', 'date', 'revenue', 'stock', 'price']].query("sales == 0")
df_zero_sales['revenue'].sum()
df[['product_id', 'sales', 'stock']].query("(product_id == 'P0015') & (stock < 2)")
df_P0015 = df[['product_id', 'sales', 'stock']].query("product_id == 'P0015'")
df_P0015['stock_valid'] = df_P0015[df_P0015['stock'].index[-1]] - df['sales']
df_P0015

df_na_sales = df[df['sales'].isnull()]
df_na_sales = df_na_sales[['sales','store_id', 'product_id', 'date', 'revenue', 'stock', 'price']]
df_na_sales['revenue'].sum()
df_na_sales.query('revenue != 0')

df_zero_revenue = df[['sales','store_id', 'product_id', 'date', 'revenue', 'stock', 'price']].query("revenue == 0")
df_zero_revenue.query('sales != 0')

df_na_price = df[df['price'].isnull()]
df_na_price = df_na_price[['sales','store_id', 'product_id', 'date', 'revenue', 'stock', 'price']]
df_na_price[df['sales'].isnull()].count()
df_na_price[df['stock'].isnull()].count()
df_na_price[df['revenue'].isnull()].count()
df_na_price.query("(sales==0) & (revenue==0)")
df_na_price.query("revenue==0")
df_na_price.query("(sales!=0) & (revenue==0)").count()
