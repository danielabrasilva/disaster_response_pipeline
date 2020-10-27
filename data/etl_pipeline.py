# import libraries
import pandas as pd
import re
from sqlalchemy import create_engine


# load messages dataset
messages = pd.read_csv('disaster_messages.csv')
messages.head()

# load categories dataset
categories = pd.read_csv('disaster_categories.csv')
categories.head()

# merge datasets
df = pd.merge(messages, categories, on='id') 
df.head()

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';',expand=True)
categories.head()

# select the first row of the categories dataframe
row_cat = categories.iloc[0,:].apply(lambda x: x[:-2])
# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing

category_colnames = row_cat.tolist()
print(category_colnames)

##### rename the columns of `categories`
categories.columns = category_colnames
categories.head()

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str.replace(r'\D','')
    
    # convert column from string to numeric
    categories[column] = categories[column].astype('int')
categories.head()


# drop the original categories column from `df`
df = df.drop(columns='categories')

df.head()

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df,categories], axis=1)
df.head()

# check number of duplicates
print("There are {} duplicated rows".format(df.duplicated().sum()))

# drop duplicates
df = df.drop_duplicates()

# check number of duplicates
print("There are {} duplicated rows".format(df.duplicated().sum()))

# verifiyng another duplicates and drop them
print(df[['id', 'message', 'original', 'genre']].duplicated().sum())

df = df.drop_duplicates(subset=['id', 'message', 'original', 'genre'])

# Replace related values 2 for 0
df['related'] = df['related'].apply(lambda x: 0 if x==2 else x)

# Test
df.groupby('related')[category_colnames[1:]].sum().sum(axis=1)


# Create a DataBase and Store the clean data 
engine = create_engine('sqlite:///UdacityProject.db')
df.to_sql('project_data', engine, index=False)