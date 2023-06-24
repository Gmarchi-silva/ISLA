
"""
Created on Thu Jun 22 19:36:49 2023
@author: Gabriel Marchi
"""

import pandas as pd
import seaborn as sns
import re
import openpyxl
from datetime import datetime


def clean_city_code(city_code):
    return city_code.replace('?', 'i').upper()


def fill_missing_values(df):
    fill_values = {'promo_bin_1': 'NA', 'promo_bin_2': 'NA', 'promo_discount_2': 0, 'promo_discount_type_2': 'NA'}
    return df.fillna(fill_values)


def drop_columns(df, columns):
    return df.drop(columns, axis=1, errors='ignore')


def calculate_continuous_week_number(df):
    df['date'] = pd.to_datetime(df['date'])
    df['week_number'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    min_year = df['year'].min()
    df['continuous_week_number'] = (df['year'] - min_year) * 52 + df['week_number']
    return df

def export_to_databricks(df, filename):
    df.to_csv(filename, index=False)  # Save the DataFrame as a CSV file

    # Upload the file to Databricks
    # You can use the appropriate Databricks API or CLI command to upload the file to your workspace

    print(f"Exported data to Databricks: {filename}")



def main():
    city_Raw = pd.read_csv(r"C:\Users\Gabriel Marchi\Downloads\cities.csv")
    product_Raw = pd.read_csv(r"C:\Users\Gabriel Marchi\Downloads\product.csv")
    sales_Raw = pd.read_csv(r"C:\Users\Gabriel Marchi\Downloads\sales.csv")

    # Clean city code
    city_Raw['city_code'] = city_Raw['city_code'].apply(clean_city_code)

    # Fill missing values
    sales_Raw = fill_missing_values(sales_Raw)

    # Merge data
    merged_data = pd.merge(sales_Raw, city_Raw, on="store_id")
    merged_data = pd.merge(merged_data, product_Raw, on="product_id")

    # Drop columns
    columns_to_drop = ['city_id_old', 'Unnamed: 0', 'country_id', 'promo_type_1', 'promo_bin_1', 'promo_type_2',
                       'promo_bin_2', 'promo_discount_2', 'promo_discount_type_2', 'hierarchy1_id', 'hierarchy2_id',
                       'hierarchy3_id', 'hierarchy4_id', 'hierarchy5_id']
    merged_data = drop_columns(merged_data, columns_to_drop)

    # Drop rows with null sales
    merged_data = merged_data.dropna(subset=['sales'])

    # Calculate continuous week number
    merged_data = calculate_continuous_week_number(merged_data)

    return merged_data


if __name__ == '__main__':
    merged_data = main()
    export_filename = r"C:\Users\Gabriel Marchi\OneDrive - Tantek Digital Solutions Limited\Desktop\Documents\Export.csv"  # Specify the desired filename for the exported file
    export_to_databricks(merged_data, export_filename)