#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 01:56:14 2020

@author: facas
"""

import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data_url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"
movies_url = "http://files.grouplens.org/datasets/movielens/ml-100k/u.item"

data_cols = ['user_id','item_id','rating','titmestamp']
movie_cols = ['item_id', 'title']

data = pd.read_csv(data_url, sep='\t', names=data_cols)
movies = pd.read_csv(movies_url, sep='|', names=movie_cols, usecols=range(2), encoding='latin-1')

print(data.head())
print(movies.head())

df = pd.merge(data, movies)
print(df.head())
print(df.describe())

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())

ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
print(ratings.head())

############## Histograms #################
#ratings['rating'].hist(bins=50)
#ratings['number_of_ratings'].hist(bins=60)

sns.jointplot(x='rating', y='number_of_ratings', data=ratings)

movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
print(movie_matrix.head())

print(ratings.sort_values('number_of_ratings', ascending=False).head(10))

AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']

print(AFO_user_rating.head())
print(contact_user_rating.head())

similar_to_air_force_one=movie_matrix.corrwith(AFO_user_rating)
similar_to_contact = movie_matrix.corrwith(contact_user_rating)

print(similar_to_air_force_one.head())
print(similar_to_contact.head())

corr_contact = pd.DataFrame(similar_to_contact, columns=['Correlation'])
corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['Correlation'])
corr_contact.dropna(inplace=True)
corr_AFO.dropna(inplace=True)

print(corr_AFO.head())
print(corr_contact.head())

corr_AFO = corr_AFO.join(ratings['number_of_ratings'])
corr_contact = corr_contact.join(ratings['number_of_ratings'])

print(corr_AFO .head())
print(corr_contact.head())

print(corr_AFO[corr_AFO['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10))
print(corr_contact[corr_contact['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10))


