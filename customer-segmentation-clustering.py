#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # CUSTOMER SEGMENTATION
# IMPORTING LIBRARIES
# In[2]:


pip install yellowbrick


# In[3]:


#Importing the Libraries
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)

#Loading Data
# In[4]:


#Loading the dataset
data = pd.read_csv(r'customer_segmentation.csv', sep=",")
print("Number of datapoints:", len(data))
data.head()


# <img src="https://github.com/KarnikaKapoor/Files/blob/main/Colorful%20Handwritten%20About%20Me%20Blank%20Education%20Presentation.png?raw=true">
# 
# For more information on the attributes visit [here](https://www.kaggle.com/imakash3011/customer-personality-analysis).
# 
# <a id="3"></a>
# # <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">DATA CLEANING</p>
# 
# 
# **In this section** 
# * Data Cleaning
# * Feature Engineering 
# 
# In order to, get a full grasp of what steps should I be taking to clean the dataset. 
# Let us have a look at the information in data. 
# 

# In[5]:


#Information on features 
data.info()


# **From the above output, we can conclude and note that:**
# 
# * There are missing values in income
# * Dt_Customer that indicates the date a customer joined the database is not parsed as DateTime
# * There are some categorical features in our data frame; as there are some features in dtype: object). So we will need to encode them into numeric forms later. 
# 
# First of all, for the missing values, I am simply going to drop the rows that have missing income values. 

# In[6]:


#To remove the NA values
data = data.dropna()
print("The total number of data-points after removing the rows with missing values are:", len(data))


# In the next step, I am going to create a feature out of **"Dt_Customer"** that indicates the number of days a customer is registered in the firm's database. However, in order to keep it simple, I am taking this value relative to the most recent customer in the record. 
# 
# Thus to get the values I must check the newest and oldest recorded dates. 

# In[7]:


data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
dates = []
for i in data["Dt_Customer"]:
    i = i.date()
    dates.append(i)  
#Dates of the newest and oldest recorded customer
print("The newest customer's enrolment date in therecords:",max(dates))
print("The oldest customer's enrolment date in the records:",min(dates))


# Creating a feature **("Customer_For")** of the number of days the customers started to shop in the store relative to the last recorded date

# In[8]:


#Created a feature "Customer_For"
days = []
d1 = max(dates) #taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
data["Customer_For"] = days
data["Customer_For"] = pd.to_numeric(data["Customer_For"], errors="coerce")


# Now we will be exploring the unique values in the categorical features to get a clear idea of the data.  

# In[9]:


print("Total categories in the feature Marital_Status:\n", data["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", data["Education"].value_counts())


# **In the next bit, I will be performing the following steps to engineer some new features:**
# 
# * Extract the **"Age"** of a customer by the **"Year_Birth"** indicating the birth year of the respective person.
# * Create another feature **"Spent"** indicating the total amount spent by the customer in various categories over the span of two years.
# * Create another feature **"Living_With"** out of **"Marital_Status"** to extract the living situation of couples.
# * Create a feature **"Children"** to indicate total children in a household that is, kids and teenagers.
# * To get further clarity of household, Creating feature indicating **"Family_Size"**
# * Create a feature **"Is_Parent"** to indicate parenthood status
# * Lastly, I will create three categories in the **"Education"** by simplifying its value counts.
# * Dropping some of the redundant features

# In[10]:


#Feature Engineering
#Age of customer today 
data["Age"] = 2021-data["Year_Birth"]

#Total spendings on various items
data["Spent"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]

#Deriving living situation by marital status"Alone"
data["Living_With"]=data["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})

#Feature indicating total children living in the household
data["Children"]=data["Kidhome"]+data["Teenhome"]

#Feature for total members in the householde
data["Family_Size"] = data["Living_With"].replace({"Alone": 1, "Partner":2})+ data["Children"]

#Feature pertaining parenthood
data["Is_Parent"] = np.where(data.Children> 0, 1, 0)

#Segmenting education levels in three groups
data["Education"]=data["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
data=data.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})

#Dropping some of the redundant features
to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
data = data.drop(to_drop, axis=1)


# Now that we have some new features let's have a look at the data's stats. 

# In[11]:


data.describe()


# The above stats show some discrepancies in mean Income and Age and max Income and age.
# 
# Do note that  max-age is 128 years, As I calculated the age that would be today (i.e. 2021) and the data is old.
# 
# I must take a look at the broader view of the data. 
# I will plot some of the selected features.

# In[12]:


#To plot some selected features 
#Setting up colors prefrences
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
#Plotting following features
To_Plot = [ "Income", "Recency", "Customer_For", "Age", "Spent", "Is_Parent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(data[To_Plot], hue= "Is_Parent",palette= (["#682F2F","#F3AB60"]))
#Taking hue 
plt.show()


# Clearly, there are a few outliers in the Income and Age features. 
# I will be deleting the outliers in the data. 

# In[13]:


#Dropping the outliers by setting a cap on Age and income. 
data = data[(data["Age"]<90)]
data = data[(data["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(data))


# Next, let us look at the correlation amongst the features. 
# (Excluding the categorical attributes at this point)

# In[14]:


#correlation matrix
corrmat= data.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)


# The data is quite clean and the new features have been included. I will proceed to the next step. That is, preprocessing the data. 
# 
# <a id="4"></a>
# # <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">DATA PREPROCESSING</p>
# 
# In this section, I will be preprocessing the data to perform clustering operations.
# 
# **The following steps are applied to preprocess the data:**
# 
# * Label encoding the categorical features
# * Scaling the features using the standard scaler 
# * Creating a subset dataframe for dimensionality reduction

# In[15]:


#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)


# In[16]:


#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")


# In[17]:


#Creating a copy of data
ds = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
print("All features are now scaled")


# In[18]:


#Scaled data to be used for reducing the dimensionality
print("Dataframe to be used for further modelling:")
scaled_ds.head()


# <a id="5"></a>
# # <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">DIMENSIONALITY REDUCTION</p>
# In this problem, there are many factors on the basis of which the final classification will be done. These factors are basically attributes or features. The higher the number of features, the harder it is to work with it. Many of these features are correlated, and hence redundant. This is why I will be performing dimensionality reduction on the selected features before putting them through a classifier.  
# *Dimensionality reduction is the process of reducing the number of random variables under consideration, by obtaining a set of principal variables.* 
# 
# **Principal component analysis (PCA)** is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimizing information loss.
# 
# **Steps in this section:**
# * Dimensionality reduction with PCA
# * Plotting the reduced dataframe
# 
# **Dimensionality reduction with PCA**
# 
# For this project, I will be reducing the dimensions to 3.

# In[19]:


#Initiating PCA to reduce dimentions aka features to 3
pca = PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T


# In[20]:


#A 3D Projection Of Data In The Reduced Dimension
x =PCA_ds["col1"]
y =PCA_ds["col2"]
z =PCA_ds["col3"]
#To plot
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x,y,z, c="maroon", marker="o" )
ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
plt.show()


# <a id="6"></a>
# # <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">CLUSTERING</p>
# 
# Now that I have reduced the attributes to three dimensions, I will be performing clustering via Agglomerative clustering. Agglomerative clustering is a hierarchical clustering method.  It involves merging examples until the desired number of clusters is achieved.
# 
# **Steps involved in the Clustering**
# * Elbow Method to determine the number of clusters to be formed
# * Clustering via Agglomerative Clustering
# * Examining the clusters formed via scatter plot

# In[21]:


# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters to be formed:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()


# The above cell indicates that four will be an optimal number of clusters for this data. 
# Next, we will be fitting the Agglomerative Clustering Model to get the final clusters. 

# In[22]:


#Initiating the Agglomerative Clustering model 
AC = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat_AC = AC.fit_predict(PCA_ds)
PCA_ds["Clusters"] = yhat_AC
#Adding the Clusters feature to the orignal dataframe.
data["Clusters"]= yhat_AC


# To examine the clusters formed let's have a look at the 3-D distribution of the clusters. 

# In[23]:


#Plotting the clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap )
ax.set_title("The Plot Of The Clusters")
plt.show()


# <a id="7"></a>
# # <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">EVALUATING MODELS</p>
# 
# Since this is an unsupervised clustering. We do not have a tagged feature to evaluate or score our model. The purpose of this section is to study the patterns in the clusters formed and determine the nature of the clusters' patterns. 
# 
# For that, we will be having a look at the data in light of clusters via exploratory data analysis and drawing conclusions. 
# 
# **Firstly, let us have a look at the group distribution of clustring**

# In[24]:


#Plotting countplot of clusters
pal = ["#682F2F","#B9C0C9", "#9F8A78","#F3AB60"]
pl = sns.countplot(x=data["Clusters"], palette= pal)
pl.set_title("Distribution Of The Clusters")
plt.show()


# 
# 

# The clusters seem to be fairly distributed.

# In[25]:


pl = sns.scatterplot(data = data,x=data["Spent"], y=data["Income"],hue=data["Clusters"], palette= pal)
pl.set_title("Cluster's Profile Based On Income And Spending")
plt.legend()
plt.show()


# **Income vs  spending plot shows the clusters pattern**
# * group 0: high spending & average income
# * group 1: high spending & high income
# * group 2: low spending & low income 
# * group 3: high spending & low income  
# 
# Next, I will be looking at the detailed distribution of clusters as per the various products in the data. Namely: Wines, Fruits, Meat, Fish, Sweets and Gold

# In[26]:


plt.figure()
pl=sns.swarmplot(x=data["Clusters"], y=data["Spent"], color= "#CBEDDD", alpha=0.5 )
pl=sns.boxenplot(x=data["Clusters"], y=data["Spent"], palette=pal)
plt.show()


# 
# From the above plot, it can be clearly seen that cluster 1 is our biggest set of customers closely followed by cluster 0.
# We can explore what each cluster is spending on for the targeted marketing strategies.
# 

# Let us next explore how did our campaigns do in the past.

# In[27]:


#Creating a feature to get a sum of accepted promotions 
data["Total_Promos"] = data["AcceptedCmp1"]+ data["AcceptedCmp2"]+ data["AcceptedCmp3"]+ data["AcceptedCmp4"]+ data["AcceptedCmp5"]
#Plotting count of total campaign accepted.
plt.figure()
pl = sns.countplot(x=data["Total_Promos"],hue=data["Clusters"], palette= pal)
pl.set_title("Count Of Promotion Accepted")
pl.set_xlabel("Number Of Total Accepted Promotions")
plt.show()


# There has not been an overwhelming response to the campaigns so far. Very few participants overall. Moreover, no one part take in all 5 of them. Perhaps better-targeted and well-planned campaigns are required to boost sales. 
# 

# In[28]:


#Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=data["NumDealsPurchases"],x=data["Clusters"], palette= pal)
pl.set_title("Number of Deals Purchased")
plt.show()


# Unlike campaigns, the deals offered did well. It has best outcome with cluster 0 and cluster 3. 
# However, our star customers cluster 1 are not much into the deals. 
# Nothing seems to attract cluster 2 overwhelmingly 
# 

# In[29]:


#for more details on the purchasing style 
Places =["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",  "NumWebVisitsMonth"] 

for i in Places:
    plt.figure()
    sns.jointplot(x=data[i],y = data["Spent"],hue=data["Clusters"], palette= pal)
    plt.show()


# Now that we have formed the clusters and looked at their purchasing habits. 
# Let us see who all are there in these clusters. For that, we will be profiling the clusters formed and come to a conclusion about who is our star customer and who needs more attention from the retail store's marketing team.
# 
# To decide that I will be plotting some of the features that are indicative of the customer's personal traits in light of the cluster they are in. 
# On the basis of the outcomes, I will be arriving at the conclusions. 

# In[30]:


Personal = [ "Kidhome","Teenhome","Customer_For", "Age", "Children", "Family_Size", "Is_Parent", "Education","Living_With"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data[i], y=data["Spent"], hue =data["Clusters"], kind="kde", palette=pal)
    plt.show()


# <a id="9"></a>
# # <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">CONCLUSION</p>
# 
# In this project, I performed unsupervised clustering. 
# I did use dimensionality reduction followed by agglomerative clustering. 
# I came up with 4 clusters and further used them in profiling customers in clusters according to their family structures and income/spending. 
# This can be used in planning better marketing strategies. 
# 
# 
# <a id="10"></a>
# # <p style="background-color:#682F2F;font-family:newtimeroman;color:#FFF9ED;font-size:150%;text-align:center;border-radius:10px 10px;">END</p>
