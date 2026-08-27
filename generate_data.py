import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

os.makedirs("data/reduced", exist_ok=True)

# Articles - H&M schema maadiri
print("Creating articles_reduced.csv...")
articles = pd.DataFrame({
    'article_id': range(100000, 101000),
    'product_code': np.random.randint(100, 900, 1000),
    'prod_name': ['T-shirt']*200 + ['Jeans']*200 + ['Dress']*200 + ['Jacket']*200 + ['Skirt']*200,
    'product_type_name': np.random.choice(['T-shirt','Trousers','Dress','Jacket'], 1000),
    'product_group_name': ['Garment Upper wear']*500 + ['Garment Lower wear']*500,
    'graphical_appearance_name': np.random.choice(['Solid','Stripe','Print'], 1000),
    'colour_group_name': np.random.choice(['Black','White','Blue','Red'], 1000),
    'department_name': np.random.choice(['Jersey Basic','Trousers','Dresses'], 1000),
    'index_name': np.random.choice(['Ladieswear','Menswear','Baby'], 1000)
})
articles.to_csv("data/reduced/articles_reduced.csv", index=False)

# Customers - H&M schema maadiri
print("Creating customers_reduced.csv...")
customers = pd.DataFrame({
    'customer_id': [f"{i:010d}" for i in range(1000000000, 1000001000)],
    'FN': np.random.choice([1.0, np.nan, 0.0], 1000, p=[0.3, 0.1, 0.6]),
    'Active': np.random.choice([1.0, np.nan], 1000, p=[0.9, 0.1]),
    'club_member_status': np.random.choice(['ACTIVE','PRE-CREATE','LEFT CLUB'], 1000),
    'fashion_news_frequency': np.random.choice(['Regularly','Monthly', None], 1000),
    'age': np.random.randint(16, 80, 1000),
    'postal_code': [f"{np.random.randint(10000, 99999)}" for _ in range(1000)]
})
customers.to_csv("data/reduced/customers_reduced.csv", index=False)

# Transactions - H&M schema maadiri
print("Creating transactions_reduced.csv...")
dates = [datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 800)) for _ in range(5000)]
transactions = pd.DataFrame({
    't_dat': [d.strftime('%Y-%m-%d') for d in dates],
    'customer_id': np.random.choice(customers['customer_id'], 5000),
    'article_id': np.random.choice(articles['article_id'], 5000),
    'price': np.round(np.random.uniform(5.99, 199.99, 5000), 2),
    'sales_channel_id': np.random.choice([1, 2], 5000)
})
transactions.to_csv("data/reduced/transactions_reduced.csv", index=False)

print("Done! Real data created.")