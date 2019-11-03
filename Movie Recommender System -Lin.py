#!/usr/bin/env python
# coding: utf-8

# # Hollywood Movie Recomender System

# In[37]:


Image(filename='images.jpg',width=600, height=600)


# In[79]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[80]:


# load the datasets
movie_titles_df = pd.read_csv('Movie_Id_Titles')


# In[81]:


# checking data
movie_titles_df.head(10)


# In[82]:


# Let's load the second one!
movies_rating_df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])


# In[83]:


#checking 2nd data
movies_rating_df.tail()


# In[84]:


# Let's drop the timestamp 
movies_rating_df.drop(['timestamp'], axis = 1, inplace = True)


# In[85]:


movies_rating_df.head()


# In[86]:


# Let's merge both dataframes together so we can have ID with the movie name
movies_rating_df= pd.merge(movies_rating_df,movie_titles_df,on='item_id')


# In[88]:


movies_rating_df.head()


# In[89]:


movies_rating_df.tail()


# In[90]:


movies_rating_df.shape


# In[91]:


movies_rating_df.describe()


# In[92]:


#Some more statistics
movies_rating_df.groupby('title')['rating'].describe().head()


# In[93]:


ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean']


# In[94]:


ratings_df_mean


# In[106]:


ratings_df_count=movies_rating_df.groupby('title')['rating'].describe()['count']


# In[107]:


ratings_df_count


# In[108]:


ratings_mean_count_df = pd.concat([ratings_df_count,ratings_df_mean],axis=1)


# In[109]:


ratings_mean_count_df.reset_index()


# In[111]:


#Top 10 rating films
ratings_mean_count_df.sort_values(by='count',ascending=False).head(10)


# # Data Visualization for Mean & Count

# In[112]:


import matplotlib.pyplot as plt


# In[113]:


ratings_mean_count_df['mean'].plot(bins=50,kind='hist',color='g',edgecolor='black')


# In[130]:


ratings_mean_count_df['count'].plot.hist(bins=30,edgecolor='black')


# In[145]:


# Let's see the highest rated movies!
# Apparently these movies does not have many reviews (i.e.: small number of ratings)
ratings_mean_count_df[ratings_mean_count_df['mean'] ==5]


# In[147]:


# List all the movies that are most rated
# Please note that they are not necessarily have the highest rating (mean)
ratings_mean_count_df.sort_values('count',ascending=False).head(50)


# # PERFORM ITEM-BASED COLLABORATIVE FILTERING ON ONE MOVIE SAMPLE

# In[149]:


movies_rating_df.head()


# In[150]:


# To create rating matrix table
userid_movietitle_matrix = movies_rating_df.pivot_table(index='user_id',columns='title',values='rating')


# In[151]:


userid_movietitle_matrix


# In[169]:


Image(filename='titanic.jpg',width=600,height=600)


# In[155]:


# Finding titanic from the rating matrix table
titanic = userid_movietitle_matrix['Titanic (1997)']


# In[157]:


titanic.head()


# In[160]:


# Let's calculate the correlations & join rating table
titanic_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(titanic), columns=['Correlation'])
titanic_correlations =titanic_correlations.join(ratings_mean_count_df['count'])


# In[163]:


titanic_correlations


# In[164]:


#Drop NA 
titanic_correlations.dropna(inplace=True)


# In[167]:


#Sort correlation number closest to Titanic, as we can we Newton Boys is closest to Titanic but very  a few reviews
titanic_correlations.sort_values('Correlation',ascending=False)


# In[170]:


#Let's set a metric for at least 200 reviews for the movie it means more people had watched and reviewed for it. 
titanic_correlations[titanic_correlations['count']>200].sort_values('Correlation',ascending=False).head(10)


# In[171]:


# So people watch Titanic 1997 would also like to watch True Lies 1994!
Image(filename='Truelies.jpg',width=600,height=600)


# In[ ]:


#Let's do another movie 


# In[199]:


Image(filename="jurassic.jpg",width=600,height=600)


# In[172]:


jurassic = userid_movietitle_matrix['Jurassic Park (1993)']


# In[198]:


jurassic.head()


# In[200]:


jurassic_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(jurassic),columns=['Correlation'])


# In[201]:


jurassic_correlations = jurassic_correlations.join(ratings_mean_count_df['count'])


# In[202]:


jurassic_correlations.head()


# In[203]:


jurassic_correlations.dropna(inplace=True)


# In[205]:


jurassic_correlations.sort_values('Correlation',ascending=False).head()


# In[206]:


jurassic_correlations[jurassic_correlations['count']>200].sort_values('Correlation',ascending=False).head()


# In[ ]:


# The Game 1997 should be recommended for the audiences like Juarassic Park.


# In[232]:


Image(filename='game.jpg',width=600,height=600)


# In[217]:


movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods = 80)


# In[227]:


# Let's create our own dataframe with our own ratings!
myRatings = pd.read_csv('My_Ratings.csv')


# In[228]:


myRatings


# In[229]:


len(myRatings)


# In[230]:


similar_movies_list = pd.Series()
for i in range(0,2):
    similar_movie= movie_correlations[myRatings['Movie Name'][i]].dropna()# Get same movies with same ratings
    similar_movie= similar_movie.map(lambda x:x*myRatings['Ratings'][i])
    similar_movies_list= similar_movies_list.append(similar_movie)


# In[231]:


similar_movies_list.sort_values(ascending=False)


# In[ ]:


#Done!

