#!/usr/bin/env python
# coding: utf-8

# ## **DATA SCIENCE AND BUISNESS ANALYTICS INTERN AT THE SPARK FOUNDATION** 

# ### **Prediction Using Supervised ML** 

# ### **Author : Gopal Dewoolkar** 

# ### **Objective: Predict the percentage of an student based on the no. of study hours.** 

# ## **Linear Regression with Python Scikit Learn**
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# ### **Simple Linear Regression**
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Reading data 
data = pd.read_csv("C:/Users/HP/Desktop/s_data.csv")


# In[4]:


data.head(10)


# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# In[6]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# ## **Preparing the data**
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[19]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 
print (X)
print (y)


# ### **Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:**

# In[9]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ### **Training the Algorithm**
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[11]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### **Making Predictions**
# Now that we have trained our algorithm, it's time to make some predictions.

# In[12]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[13]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[20]:


# You can also test with your own data
hours = (9.25)
print("No of Hours = " , hours)
own = regressor.predict([[hours],])
print("Predicted Score = " , own[0])


# In[21]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




