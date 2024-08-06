#!/usr/bin/env python
# coding: utf-8

# # Lab 1: ML Life Cycle: Business Understanding and Problem Formulation

# In[124]:


import pandas as pd
import numpy as np


# In this lab, you will practice the first step of the machine learning life cycle: formulating a machine learning problem. But first, you will get more practice working with some of the Python machine learning packages that you will use throughout the machine learning life cycle to develop your models.

#  ## Part 1. Practice Working with ML Python Tools

# In this part of the lab you will:
# 
# 1. Work with NumPy arrays and NumPy functions
# 2. Create Pandas DataFrames from data
# 3. Use NumPy and Pandas to analyze the data
# 4. Visualize the data with Matplotlib

# <b>Note</b>: In Jupyter Notebooks, you can output a variable in two different ways: 
# 
# 1. By writing the name of the variable 
# 2. By using the python `print()` function
# 
# The code cells below demonstrate this. Run each cell and inspect the results.

# In[125]:


x = 5
x


# In[126]:


x = 5
print(x)


# If you want to output multiple items, you must use a `print()` statement. See the code cell below as an example.

# In[127]:


y = 4
z = 3

print(y)
print(z)


# ## Practice Operating on NumPy Arrays

# ### a. Define a Python list

# The code cell below defines a new list in Python.

# In[128]:


python_list = [0,2,4,6,8,10,12,14]
python_list


# ### b. Define a Python range
# 
# The code cell below defines a Python range in that contains the same values as those in the list above.

# In[129]:


python_range = range(0, 15, 2)
python_range


# The above returns an object of type `range`. The code cell below coverts this object to a Python list using the Python `list()` function.
#  

# In[130]:


list(python_range)


# ### c. Define a NumPy range

# <b>Task:</b> In the code cell below, use NumPy's `np.arange()` method to create a NumPy range that has the same output as the Python range above. Save the output to the variable `numpy_range`.

# In[131]:


numpy_range = np.arange(0,15,2)


# The code above returns an object of type `ndarray` (i.e. an array). 
# 
# <b>Task:</b> In the code cell below, convert the NumPy array `numpy_range` to a Python list.
# 

# In[133]:


list(numpy_range)


# ### d. List comprehension

# Consider the task of replacing each value in a list with its square. The traditional way of performing a transformation on every element of a list is via a `for` loop. 
# 
# 
# <b>Task:</b> In the code cell below, use a `for` loop to replace every value in the list `my_list` with its square (e.g. 2 will become 4). In your loop, use a range.
# 
# 

# In[134]:


my_list = [1,2,3,4]

for i in range(len(my_list)):
    my_list[i] = my_list[i]**2
    
print(my_list)


# There is a different, more 'Pythonic' way to accomplish the same task.<br>
# 
# *List comprehension* is one of the most elegant functionalities native to Python. It offers a concise way of applying a particular transformation to every element in a list. <br>
# 
# By using list comprehension syntax, we can write a single, easily interpretable line of code that does the same transformation without using ranges or an iterating index variable `i`:

# In[135]:


my_list = [1,2,3,4]

my_list = [x**2 for x in my_list]

print(my_list)


# ### e. Create a NumPy array

# <b>Task:</b> In the code cell below, create a `numpy` array that contains the integer values 1 through 4. Save the result to variable `arr`.

# In[136]:


arr = np.arange(1,5)
arr


# ### e.  Obtain the dimensions of the NumPy array

# The NumPy function `np.shape()` returns the dimensions of a `numpy` array. Because `numpy` arrays can be two dimensional (i.e. a matrix), `np.shape()` returns a tuple that contains the lengths of the array's dimensions. You can consider the result to be the number of rows and the number of columns in the NumPy array.
# 
# <b>Task:</b> In the code cell below, use `np.shape()` to find the 'shape' of `numpy` array `arr`.

# In[137]:


np.shape(arr)


# Notice that there appears to be an empty 'slot' for another number in the tuple.
# Since `arr` is a one-dimensional array, we only care about the first element in the tuple that `np.shape()` returns.
# 
# <b>Task:</b> In the code cell below, obtain the length of `arr` by extracting the first element that is returned by `np.shape()`. Save the result to the variable `arr_length`.
# 

# In[138]:


arr_length = np.shape(arr)[0]

print(arr_length)


# ### f. Create a uniform (same value in every position) array

# We will now use ```np.ones()``` to create an array of a specified length that contains the value '1' in each position:

# In[139]:


np.ones(55, dtype=int)


# We can use this method to create an array of any identical value. Let's create an array of length 13, filled with the value '7' in every position:

# In[140]:


7 * np.ones(13, dtype=int)


# ### g. Create a two-dimensional NumPy array

# Let us explore the possibilities of the ```np.array()``` function further. NumPy arrays can be of more than one dimension. The code cell below creates a two-dimensional `numpy` array (a matrix).

# In[141]:


matrix = np.array([[1,2,3], [4,5,6]])
matrix


# <b>Task:</b> In the code cell below, use `np.shape()` to find the dimensions of `matrix`.

# In[142]:


np.shape(matrix)


# ### h. Create an identity matrix

# `np.eye()` is a NumPy function that creates an identity matrix of a specified size. An identity matrix is a matrix in which all of the values of the main diagonal are one, and all other values in the matrix are zero.

# In[143]:


np.eye(5)


# Check your intuition: What do you think will be the output after running this cell? Run the cell to see if you are correct.

# In[144]:


A = np.eye(3)
B = 4 * np.eye(3)
A+B


# ### i. A small challenge:  matrix transformation and random matrix generation
# 
# The `np.triu()` function obtains the upper right triangle of a two-dimensional NumPy array (matrix). Inspect the documentation by running the command ```np.triu?``` in the cell below. 

# In[145]:


get_ipython().run_line_magic('pinfo', 'np.triu')


# <b>Task:</b> Inspect the code in the cell below, then run the code and note the resulting matrix `M`.

# In[146]:


M = np.round(np.random.rand(5,5),2)
print("M=\n", M)

#np.random.rand(5,5) = 5 rows, 5 columns. by default, number is between 0 and 1
#np.round(each index, 2 decimal points)


# <b>Task:</b> Use `np.triu()` to create a matrix called ```new_M``` which is identical to the matrix```M```, except that in the lower triangle (i.e., all the cells below the diagonal), all values will be zero.

# In[147]:


new_M = np.triu(M, k = 0)

print("new_M=\n", new_M)


# <b>Task:</b> Using the code provided above for generating the matrix ```M```, try creating a matrix with 13 rows and 3 columns containing random numbers. Save the resulting matrix to the variable `random_M`.

# In[148]:


random_M = np.round(np.random.rand(13,3),2)

print("random_M= \n", random_M)


# ### j. Indexing and slicing two-dimensional NumPy arrays

# The code cell below extracts an element of a two-dimensional NumPy array by indexing into the array by specifying its location. Just like Python lists, NumPy arrays use 0-based indexing.

# In[149]:


random_M[3][2]


# You can also use the following syntax to achieve the same result.

# In[150]:


random_M[3,2]


# You learned how to slice a Pandas DataFrames. You can use the same techniques to slice a NumPy array. 
# 
# 
# <b>Task:</b> In the code cell below, use slicing to obtain the rows with the index 3 through 5 in `random_M`.

# In[151]:


random_M[3:6]


# <b>Task:</b> In the code cell below, use slicing to obtain all of the rows in the second column (column has the index of 1) of `random_M`.

# In[152]:


random_M[:,1]


# <b>Task:</b> Use the code cell below to perform slicing on `random_M` to obtain a portion of the array of your choosing.

# In[153]:


random_M[3:7,1:2]


# ### k. Evaluating a Boolean condition

# In real-life data tasks, you will often have to compute the boolean ```(True/False)``` value of some statement for all entries in a given NumPy array. You will formulate a condition &mdash; think of it as a *test* &mdash; and run a computation that returns `True` or `False` depending on whether the test passed or failed by a particular value in the array.
# 
# The condition may be something like "the value is greater than 0.5". You would like to know if this is true or false for every value in  the array. 
# 
# The code cells below demonstrates how to perform such a task on NumPy arrays.
# 
# First, we will create the array:

# In[154]:


our_array = np.random.rand(1, 20)
print(our_array)


# Next, we will apply a condition to the array:

# In[155]:


is_greater = our_array > 0.5
print(is_greater)


# Let's apply this technique to our matrix `random_M`. Let's inspect the matrix again as a refresher.

# In[156]:


print(random_M)


# <b>Task:</b> In the code cell below, determine whether the value of every element in the second column of `random_M` is greater than 0.5. Save the result to the variable `is_greater`.

# In[157]:


is_greater = random_M[:,1] > 0.5

print(is_greater)


# We can use the function `np.any()` to determine if there is any element in a NumPy array that is True. Let us apply this to the array `is_greater` above. Using this function we can easily determine that indeed there are values greater than 0.5 in the second row of `random_M`.

# In[158]:


np.any(is_greater)


# Let's apply `np.any()` to another condition. 
# 
# <b>Task:</b> Use `np.any()` along with a conditional statement to determine if any value in the third row of `random_M` is less than .1.

# In[159]:


np.any(random_M[2,:] < 1)


# ## Practice Working With Pandas DataFrames

# ### a. Creating a DataFrame: two (of the many) ways

# The code cells below demonstrate how we can create Pandas DataFrames in two ways: 
# 
# 1. From a *list of lists*
# 2. From a *dictionary*
# 
# First, the cell below creates a DataFrame from a list containing phone numbers and their country codes. The DataFrame is named `df`. Run the cell below to inspect the DataFrame `df` that was created.

# In[160]:


my_list = [['+1', '(929)-000-0000'], ['+34', '(917)-000-0000'], ['+7', '(470)-000-0000']]

df = pd.DataFrame(my_list, columns = ['country_code', 'phone'])
df


# Second, the cell below creates a DataFrame from a dictionary that contains the same information as the list above. The dictionary contains phone numbers and their country codes. Run the cell below to inspect the DataFrame `df_from_dict` that was created from the dictionary. Notice that both DataFrames `df` and `df_from_dict` contain the same values.

# In[161]:


my_dict = {'country_code': ['+1', '+34', '+7'], 'phone':['(929)-000-0000', '(917)-000-0000', '(470)-000-0000']}

df_from_dict = pd.DataFrame(my_dict)
df_from_dict


# ### b. Adding a column to a DataFrame object

# We are going to continue working with the DataFrame `df` that was created above. The code cell below adds a new column of values to `df`. Run the cell and inspect the DataFrame to see the new column that was added.

# In[162]:


df['grade']= ['A','B','A']
df


# <b>Task:</b> In the cell below, create a new column in DataFrame `df` that contains the names of individuals.
# 
# * First, create a list containing three names of your choosing. 
# * Next, create a new column in `df` called `names` by using the list you created.

# In[163]:


names = ['Natalie', 'Nour', 'Keerthana']
df['Names'] = names

df


# ### c. Sorting the DataFrame by values in a specific column

# The `df.sort_values()` method sorts a DataFrame by the specified column. The code cell below will use `df.sort_values()` to sort DataFrame`df` by the values contained in column `grade`. The original DataFrame `df` will not be changed, so we will assign the resulting DataFrame to variable `df` to update the values in the DataFrame.

# In[164]:


df = df.sort_values(['grade'])
df


# ### d. Combining multiple DataFrames  and renaming  columns with `df.rename()`

# In real life settings, you will often need to combine separate sets of related data. Two functions used for this purpose are `pd.concat()` and `pd.merge()`.
# 
# 
# To illustrate, let's create a new DataFrame. The code cell below creates a new DataFrame `df2` that also contains phone numbers, their country codes and a grade. Run the cell and inspect the new DataFrame that was created.

# In[165]:


my_dict2 = {'country': ['+32', '+81', '+11'], 'grade':['B', 'B+', 'A'], 'phone':['(874)-444-0000', '(313)-003-1000', '(990)-006-0660']}

df2 = pd.DataFrame(my_dict2)
df2


# The code cell below uses the Pandas ```pd.concat()``` function to append `df2` to `df`. The `pd.concat()` function will not change the values in the original DataFrames, so we will save the newly formed DataFrame to variable `df_concat`. 

# In[166]:


# experimenting during pair programming: 
# my_dict3 = {'animal': ['dog', 'cat', 'fish']}
# df3 = pd.DataFrame(my_dict3)
# df_concat = pd.concat([df,df2, df3])

df_concat = pd.concat([df,df2])
df_concat


# Notice that the new DataFrame `df_concat` contains two columns containing country codes. This is because the two original DataFrames contained different spellings for the columns. 
# 
# 
# We can easily fix this by changing the name of the column in DataFrame `df2` to be consistent with the name of the column in DataFrame `df`.

# In[167]:


df2 = df2.rename(columns={'country':'country_code'})
df2


# <b>Task</b>: In the cell below, run the `pd.concat()` function again to concatenate DataFrames `df` and `df2` and save the resulting DataFrame to variable `df_concat2`. Run the cell and inspect the results.

# In[168]:


df_concat2 = pd.concat([df,df2])

print(df_concat2)


# One other problem is that the index has repeated values. This defeats the purpose of an index, and ought to be fixed. Let's try the concatenation again, this time adding `reset_index()` method to produce correct results:

# In[169]:


df_concat2 = pd.concat([df,df2]).reset_index()
df_concat2


# Now we have one column for `country_code`. Notice that we have missing values for the names of individuals, since names  were contained in `df` but not in `df2`. In a future unit, you will learn how to deal with missing values.

# What if our task were to merge ```df2``` with yet another dataset &mdash; one that contains additional unique columns? Let's look at DataFrame `df2` again:

# In[170]:


df2


# The code cell below creates a new DataFrame `df3`.

# In[171]:


my_dict3 = {'country_code': ['+32', '+44', '+11'], 'phone':['(874)-444-0000', '(575)-755-1000', '(990)-006-0660'], 'grade':['B', 'B+', 'A'], 'n_credits': [12, 3, 9]}

df3 = pd.DataFrame(my_dict3)
df3


# The following code cell merges both DataFrames based on the values contained in the `phone` column. If one column in both DataFrames contains the same value, the rows in which the value appears are merged. Otherwise, the row will not be included in the updated DataFrame. Run the code cell below and inspect the results. Note that the values in DataFrame `df2` will be automatically changed by `merge()`. 

# In[172]:


df2.merge(df3, on = 'phone')
df2


# I believe there is a logical error here. 
# When running df2, the dataframe shows all three rows 
# despite the phone #(313)-003-1000 being different for df3 and df2.
# Solution A: print(df2.merge(df3, on = 'phone'))
# Solution B: df4 = df2.merge(df3, on = 'phone') 
#             df4
# :D 
# This makes me believe df2 will not be *automaticaly changed* by merge().


# ## Practice Working With a Dataset

# We are now well equipped to deal with a real dataset! Our dataset will contain information about New York City listings on the Airbnb platform.
# 
# ### a. Load the dataset: `pd.read_csv()`
# 
# The code cell below loads a dataset from a CSV file and saves it to a Pandas DataFrame. 
# 
# First, we will import the `OS` module. This module enables you to interact with the operating system, allowing you access to file names, etc.
# 
# 

# In[173]:


import os 


# Next, we will use the `os.path.join()` method to obtain a path to our data file. This method concatenates different path components (i.e. directories and a file name, into one file system path). We will save the results of this method to the variable name `filename`.
# 
# Now that we have a path to our CSV file, we will use the `pd.read_csv()` method to load the CSV file into a Pandas DataFrame named `dataFrame`.
# 
# Examine the code in the cell below and run the cell.
# 
# <b>Note</b>: the cell below may generate a warning. Ignore the warning. 

# In[174]:


filename = os.path.join(os.getcwd(), "data", "airbnbData.csv") 
dataFrame = pd.read_csv(filename)


# In[175]:


dataFrame.shape


# First, get a peek at the data:

# In[176]:


dataFrame.head()


# When using the `head()` method, you can specify the number of rows you would like to see by calling `head()` with an integer parameter (e.g. `head(2)`).

# ### b. Get column names: `df.columns`
# 
# Let us retrieve just the list of column names.

# In[177]:


list(dataFrame.columns)


# What do the column names mean? Some of them are less intuitively interpretable than others. <br>
# Careful data documentation is indispensable for business analytics. You can consult the documentation that accompanies this open source dataset for a detailed description of the key variable names, what they represent, and how they were generated.

# ### c. Summary statistics of the DataFrame: `df.describe()`

# Let's print some general statistics for each one of the `data` columns:

# In[178]:


dataFrame.describe(include='all')


# ### d. Filtering the data: `df[ < condition > ]`

# Consider the following business question: What is the average availability (out of 365 days in a year) for the listings in Brooklyn? <br>
# 
# The answer can be obtained by the use of **filters** on the dataset. We need to filter the entries that are in Brooklyn. To do this, we need to know the exact way that Manhattan listings are spelled and entered in the data. Let's print all of the unique values of the `neighbourhood` column:

# In[179]:


dataFrame['neighbourhood'].unique()


# You may have noticed that there is a lot of heterogeneity in the way `neighbourhood` values are specified. The values are not standardized. There are overlaps, redundancies, and inconsistencies (e.g., some entries specify ```'Greenpoint, Brooklyn, New York, United States'```, some other ones list `'BROOKLYN, New York, United States',`, yet other ones say `'Williamsburg, Brooklyn, New York, United States'`, etc. In real life, you would have to clean this data and replace these values with standard, identically formated, consistent values. <br>
# 
# For this dataset, we are lucky to already have a 'cleansed' version of the neighborhood information based on the latitude and the longitude of every listing location. 
# 
# We will list the unique values of the columns titled `neighbourhood_cleansed` and `neighbourhood_group_cleansed`:

# In[180]:


dataFrame['neighbourhood_cleansed'].unique()


# In[181]:


dataFrame['neighbourhood_group_cleansed'].unique()


# Let's filter out all data entries that pertain to Brooklyn listings:

# In[182]:


bk = dataFrame[dataFrame['neighbourhood_group_cleansed'] == 'Brooklyn']
bk.shape


# <b>Tip</b>: to better understand what happened above, in the code cell below, you are encouraged to copy *just the condition* of the filter that we used on the `data` object above: `dataFrame['neighbourhood_group_cleansed'] == 'Brooklyn'`. 
# 
# Run the cell and see what that condition alone evaluates to. You should see a Pandas series containing True/False values. When we use that series as a Boolean filter by writing `dataFrame[ < our Boolean series > ]`, i.e `dataFrame['neighbourhood_group_cleansed'] == 'Brooklyn']`, we are telling Pandas to keep the values in the DataFrame `dataFrame` only with those indices for which the condition evaluated to `True`. 

# In[183]:


dataFrame['neighbourhood_group_cleansed'] == 'Brooklyn'


# 
# ### e. Combining values in a column: `np.mean()`

# Now that we isolated only the relevant entries, it remains to average the value of a particular column that we care about:

# In[184]:


np.mean(bk['availability_365'])


# ### f. Group data by (categorical) column values: `df.groupby()`

# The next question of interest could be:<br>
# 
# What are the top 5 most reviewed neighborhoods in New York? (By sheer number of reviews, regardless of their quality). <br>
# 
# We will use the Pandas ```df.groupby()``` method to determine this:

# In[185]:


nbhd_reviews = dataFrame.groupby('neighbourhood_cleansed')['number_of_reviews'].sum()
nbhd_reviews.head()


# Perform a (descending order) sorting on this series:

# In[186]:


nbhd_reviews = nbhd_reviews.sort_values(ascending = False)
nbhd_reviews.head(5)


# What are the least reviewed neighborhoods?

# In[187]:


nbhd_reviews.tail(5)


# This result makes it apparent that our dataset is somewhat messy!

# Notice we could have chained the transformations above into a single command, as in:

# In[188]:


dataFrame.groupby('neighbourhood_cleansed')['number_of_reviews'].sum().sort_values(ascending = False).head(5)


# This way we don't store objects that we won't need.

# ### Bonus: Histogram plotting with Matplotlib: `plt.hist()`

# As a final touch, run the cell below to visualize the density of average values of review numbers across all neighborhoods. <b>Note:</b> The cell may take a few seconds to run.

# In[189]:


# %matplotlib inline command is a magic function in IPython and Jupyter notebooks. 
# It is used to configure the Matplotlib library to display plots directly within the notebook, 
# as opposed to in a separate window
# inline means it runs right below
get_ipython().run_line_magic('matplotlib', 'inline')
nbhd_reviews.hist()


# This plot suggests that the vast majority of neighborhoods have only very few reviews, with just a handful of outliers (those ranked at the top in our previous computed cell) having the number of reviews upward of 40000. 

# ## Part 2. ML Life Cycle: Business Understanding and Problem Formulation

# In this part of the lab, you will practice the first step of the machine learning life cycle: business understanding and problem formulation.
# 
# Recall that the first step of the machine learning life cycle involves understanding and formulating your ML business problem, and the second step involves data understanding and preparation. In this lab however, we will first provide you with data and have you formulate a machine learning business problem based on that data.
# 
# We have provided you with four datasets that you will use to formulate a machine learning problem.
# 
# 1. <b>HousingPrices.csv</b>: dataset that contains information about a house's characteristics (number of bedrooms, etc.) and its purchase price.
# 
# 2. <b>Top100Restaurants2020.csv</b>: dataset that contains information about 100 top rated restaurants in 2020.
# 
# 3. <b>ZooData.csv</b>: dataset that contains information about a variety of animals and their characteristics.
# 
# 4. <b>FlightInformation.csv</b>: dataset that contains flight information.
# 
# The code cells below use the specified paths and names of the files to load the data into four different DataFrames.
# 
# <b>Task \#1</b>: After you run a code cell below to load the data, use some of the techniques you have practiced to inspect the data. Do the following: 
# 
# 1. Inspect the first 10 rows of each DataFrame.
# 2. Inspect all of the column names in each DataFrame.
# 3. Obtain the shape of each DataFrame.
# 
# (Note: You can add more cells below to accomplish this task by going to the `Insert` Menu and clicking on `Insert Cell Below`. By default, the new code cell will be of type `Code`.)
# 
# <b>Task \#2</b>: Once you have an idea of what is contained in a dataset, you will formulate a machine learning problem for that dataset. This will be a predictive problem. For example, the Airbnb dataset you worked with above can be used to train a machine learning model that can predict the price of a new Airbnb. 
# 
# Come up with at least one machine learning problem per dataset. Specify what you would like to use the data to predict in the future. Since these will be supervised learning problems, specify whether it is a classification (binary or multiclass) or a regression problem. List the label and feature columns. 
# 
# Note: Make sure you successfully ran the cell above that loads the `OS` module prior to running the cells below.

# <b>Housing Prices Dataset</b>:

# In[190]:


filename1 = os.path.join(os.getcwd(), "data", "HousingPrices.csv") 

dataFrame1 = pd.read_csv(filename1)


# Inspect the data:

# In[191]:


#Inspect the first 10 rows of each DataFrame.
dataFrame1.head(10)


# In[192]:


#Inspect all of the column names in each DataFrame.
list(dataFrame1.columns)


# In[193]:


#Obtain the shape of each DataFrame.
dataFrame1.shape


# Formulate ML Business Problem:

# * **Problem/ Predicting**: I would use this data to predict the price of houses based on their feautures. This can be useful for many different reasons. For instance - if one is trying to buy a home, they can determine whether they are being outrageously scammed or not based on the feautures showcased.  
# * **Supervised Learning Problem**: Regression
# * **Label**: price
# * **Features**: area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus. Based on my limited understanding of house prices, all information seems useful in predicting the most accurate price.

# <b>Restaurants Dataset</b>:

# In[194]:


filename2 = os.path.join(os.getcwd(), "data", "Top100Restaurants2020.csv") 

dataFrame2 = pd.read_csv(filename2)


# Inspect the data:

# In[195]:


#Inspect the first 10 rows of each DataFrame.
dataFrame2.head(10)


# In[196]:


#Inspect all of the column names in each DataFrame.
list(dataFrame2.columns)


# In[197]:


#Obtain the shape of each DataFrame.
dataFrame2.shape


# Formulate ML Business Problem:

# * **Problem/ Predicting**: I would use this data to predict the number of sales a top 100 restaurant makes based on characteristics provided by the data. Tackling this problem can be very helpful for individuals in the restaurant industry trying to create their our restaurant. By considering the restaurant's characteristics/attributes (features listed below), new business owner's can estimate their potential sales. 
# * **Supervised Learning Problem**: Regression
# * **Label**: Sales
# * **Features**: Average Check, City, State, Meals Served, Category

# <b>Zoo Dataset</b>:

# In[198]:


filename3 = os.path.join(os.getcwd(), "data", "ZooData.csv") 

dataFrame3 = pd.read_csv(filename3)


# Inspect the data:

# In[199]:


#Inspect the first 10 rows of each DataFrame.
dataFrame3.head(10)


# In[200]:


#Inspect all of the column names in each DataFrame.
list(dataFrame3.columns)


# In[201]:


#Obtain the shape of each DataFrame.
dataFrame3.shape


# Formulate ML Business Problem:

# * **Problem/ Predicting**: I would use this data to predict the animal's name based on the given columns in the data frame. This can be useful for individuals who encounter an unfamiliar animal and want to identify it. Knowing the animal's name is important for understanding how to interact with it or to satisfy curiosity. By recognizing the animal's characteristics, people can accurately determine its name. 
# * **Supervised Learning Problem**: Classification
# * **Label**: animal_name
# * **Features**: All the other columns: hair	feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, catsize, class_type. It's important to note that depending on the specific business purpose, some columns may not be useful. For example, many people might not know if an animal is a mammal or a predator. This makes me excited to learn how we confront null values when using supervised learning to make predictions! 

# <b>Flight Dataset</b>:

# In[202]:


filename4 = os.path.join(os.getcwd(), "data", "FlightInformation.csv") 

dataFrame4 = pd.read_csv(filename4)


# Inspect the data:

# In[203]:


#Inspect the first 10 rows of each DataFrame.
dataFrame4.head(10)


# In[204]:


#Inspect all of the column names in each DataFrame.
list(dataFrame4.columns)


# In[205]:


#Obtain the shape of each DataFrame.
dataFrame4.shape


# Formulate ML Business Problem:

# * **Problem/ Predicting**: I would use this data to predict whether a flight will be on time or delayed based on the given columns in the data frame. This can be useful for people who are traveling via flights and are curious to predict if their flight is likely to be delayed or not.
# * **Supervised Learning Problem**: Binary, delayed (1) or not delayed (0).
# * **Label**: Delay (0: not delayed or 1: delayed)
# * **Features**: Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length

# <b>Next Steps</b>: The second step of the machine learning life cycle is data understanding and data preparation. You practiced some aspects of data understanding when using NumPy and Pandas to inspect the Airbnb dataset. You will learn more about this second step of the machine learning life cycle in the next unit.

# In[206]:


print("Fun lab :D")

