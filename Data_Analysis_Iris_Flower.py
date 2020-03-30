#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")


# In[3]:


print(iris.shape) #data points and features


# In[9]:


print(iris.columns) #column names


# In[8]:


iris["species"].value_counts()#Datapoints/flowers for each class/specie


# In[10]:


iris.plot(kind= "scatter", x="sepal_length", y="sepal_width")
plt.show()


# In[13]:


sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="species", height=4)     .map(plt.scatter, "sepal_length", "sepal_width")     .add_legend();
plt.show()


# In[14]:


#Blue point can be easily separated from red and green.
#But red and green data points can not be easily separated
#Using sepal_length and sepal_width features, we can distinguish Setosa flowers from others
#Separating Versicolor from Virginica is harde beacuse they overlap
#So, try using pair-plot


# In[15]:


sns.set_style("whitegrid");
sns.pairplot(iris, hue="species", height= 3);
plt.show()


# In[16]:


#petal_length and petal_wdth  are the most useful features to identify various flower types
#pairplots useful til 5/6 features
#What about using 1D scatter plot using just one feature?


# In[32]:


iris_setosa = iris.loc[iris["species"] == "setosa"];
iris_virginica = iris.loc[iris["species"] == "virginica"];
iris_versicolor = iris.loc[iris["species"] == "versicolor"];

plt.plot(iris_setosa["petal_length"], np.zeros_like(iris_setosa["petal_length"]), 'o')
plt.plot(iris_versicolor["petal_length"], np.zeros_like(iris_versicolor["petal_length"]), 'o')
plt.plot(iris_virginica["petal_length"], np.zeros_like(iris_virginica["petal_length"]), 'o')

plt.show()


# In[18]:


#points are overlaping and really hard to read
#What about an histogram?


# In[24]:


sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="species", height=5)     .map(sns.distplot, "petal_length")     .add_legend()
plt.show()


# In[ ]:


#Histogram of petal_length of each flower
#y axis represents how often we find this petal_length in each flower
#Smooth lines = PDF(Probability density function)
#petal_lenght = pl;
#Model 1 => if pl < 2 then setosa
#Model 2 => if pl >2 && pl<= 5.5 then versicolor // DO NOT WORK BECAUSE OVERLAPING
#Model 3 => if pl>5.5 then virginica // DO NOT WORK BECAUSE OVERLAPING

#So, my final model>
#if pl <= 2 then setosa
#else
#if pl < 4.7 then versicolor
#else
#virginica // BUT STILL NOT PERFECT

#Univariate analysis:


# In[28]:


sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="species", height=5)     .map(sns.distplot, "petal_width")     .add_legend()
plt.show()

#The further the distribution are, the better


# In[27]:


sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="species", height=5)     .map(sns.distplot, "sepal_length")     .add_legend()
plt.show()

#Worst overlaping


# In[26]:


sns.set_style("whitegrid");
sns.FacetGrid(iris, hue="species", height=5)     .map(sns.distplot, "sepal_width")     .add_legend()
plt.show()

#Worst overlaping


# In[29]:


#From previous plot: PL>PW>>>SL>>SW


# In[30]:


#Cumulative Distribution FUnction CDF
#Plot CDF of petal_length


# In[33]:


counts, bin_edges = np.histogram(iris_setosa['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# virginica
counts, bin_edges = np.histogram(iris_virginica['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#versicolor
counts, bin_edges = np.histogram(iris_versicolor['petal_length'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


plt.show();


# In[34]:


#Objetive: classication
# Plots of CDF of petal_length for various types of flowers.
#95% of versica flower: pl>2 && pl<= 5 then versicolor
#So, i will be correct 95% of the times


# In[35]:


#Mean
print("Means:")
print(np.mean(iris_setosa["petal_length"]))
print(np.mean(iris_virginica["petal_length"]))
print(np.mean(iris_versicolor["petal_length"]))


# In[40]:


#Setosa has a smaller petal lenght in average
#Setosa mean with an outlier:

print(np.mean(np.append(iris_setosa["petal_length"],50)));


# In[38]:


#PROBLEM with mean

print("\nStd-dev:");
print(np.std(iris_setosa["petal_length"]))
print(np.std(iris_virginica["petal_length"]))
print(np.std(iris_versicolor["petal_length"]))


# In[39]:


#Std -> spread

#Median = similar to mean but outliers do not affect them
print("\nMedians:")
print(np.median(iris_setosa["petal_length"]))

print(np.median(iris_virginica["petal_length"]))
print(np.median(iris_versicolor["petal_length"]))


# In[41]:


#Setosa Median with an outlier
print(np.median(np.append(iris_setosa["petal_length"],50)));


# In[45]:


#No problem with variance

print("\nQuantiles:")
print(np.percentile(iris_setosa["petal_length"],np.arange(0, 100, 25)))
print(np.percentile(iris_virginica["petal_length"],np.arange(0, 100, 25)))
print(np.percentile(iris_versicolor["petal_length"], np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(iris_setosa["petal_length"],90))
print(np.percentile(iris_virginica["petal_length"],90))
print(np.percentile(iris_versicolor["petal_length"], 90))


# In[ ]:




