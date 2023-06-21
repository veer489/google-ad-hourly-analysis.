#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as dp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#importing dataset


# In[5]:


file=dp.read_excel("~/desktop/Google ads hourly analysis 19th june.xlsx")


# In[6]:


file


# In[ ]:


# the data set gives us how many people who clicked and which was cold lead and hot lead and warm lead of advertisment with ctr and cpc .


# In[7]:


file. head()


# In[9]:


file.tail()


# In[10]:


#data preprocessing 


# In[11]:


#check number of unique value from all data set 


# In[12]:


file.select_dtypes(include='object').nunique()


# In[14]:


file.shape


# In[18]:


data=file.drop("Sr no",axis=1)


# In[19]:


data


# In[20]:


data.isnull().sum()


# In[21]:


file.isnull().sum()


# In[41]:


rt=data.dropna()


# In[42]:


rt


# In[43]:


rt.isnull().sum()


# In[44]:


don=rt.fillna(rt.mean())


# In[45]:


don


# In[46]:


# descriptive analysis.


# In[47]:


don.info()


# In[48]:


data.describe()


# In[49]:


#We can also see the number of unique Impresssion and Clicks  in the dataset.


# In[50]:


don.nunique()


# In[51]:


don.sum()


# In[52]:


don.mean()


# In[53]:


don.isnull().sum()


# In[71]:


x=don.iloc[:,:-1].values
y=don.iloc[:,-1].values


# In[72]:


x


# In[73]:


y


# In[74]:


#linear regreesion 


# In[75]:


#exploratory data analysis.


# In[76]:


from sklearn.model_selection import train_test_split 
x_train, x_test,  Y_train, Y_test = train_test_split(x,y, test_size = 0.30, random_state = 0) 


# In[83]:


print(x_train)


# In[85]:


print(Y_train)


# In[86]:


print(x_test)


# In[87]:


print(Y_test)


# In[77]:


from sklearn.linear_model import LinearRegression 


# In[78]:


lr= LinearRegression()


# In[79]:


lr


# In[80]:


lr.fit(x_train,Y_train)


# In[81]:


y_predict=lr.predict(x_test)


# In[82]:


y_predict


# In[94]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[109]:


x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[110]:


print(x_train)


# In[111]:


print(x_test)


# In[112]:


from sklearn.tree import DecisionTreeClassifier


# In[115]:


vt= DecisionTreeClassifier(criterion="gini",random_state=0)



# In[117]:


vt.fit(x_train,Y_train)


# In[123]:


print(vt.predict(sc.transform([[2500,121,300,0.0121,2.232,1,0.1]])))


# In[124]:


print(vt.predict(sc.transform([[2600,232,300,0.0121,2.232,1,0.1]]))) 


# In[125]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[126]:


Y_prediction =vt.predict(x_test)


# In[127]:


Y_prediction


# In[128]:


gm= confusion_matrix(Y_test,Y_prediction)


# In[129]:


accuracy_score(Y_test,Y_prediction)


# In[ ]:


# we obtain 73 percentage of accuracy of dataset 


# In[ ]:


#main alogorithms 


# In[ ]:


#Correlations


# In[ ]:


# For whole dataset


# In[153]:


don.corr()


# In[ ]:


# For some selected coulmns or attributes


# In[156]:


rt=don[['Clicks','CTR']].corr()


# In[157]:


rt


# In[162]:


rr=don[['CTR','CPC']].corr()


# In[163]:


rr


# In[175]:


sns.lineplot(x='CTR',y='CPC',data=data)


# In[170]:


sns.lineplot(x='Clicks',y='CTR',data=data)


# In[176]:


corr = don.corr()

plt.figure(figsize=(8,4))
sns.heatmap(corr,cmap="Greens",annot=True)


# In[177]:


corr = don.corrwith(don['CTR']).sort_values(ascending = False ).to_frame()
corr.columns =['CTR']
plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot= True,cmap = 'Greens',linewidths=2,linecolor='black');
plt.title('CTR Correlation')


# In[179]:


corr = don.corrwith(don['CPC']).sort_values(ascending = False ).to_frame()
corr.columns =['CPC']
plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot= True,cmap = 'Greens',linewidths=2,linecolor='black');
plt.title('CPC Correlation')


# In[180]:


corr = don.corrwith(don['Hot Leads']).sort_values(ascending = False ).to_frame()
corr.columns =['Hot Leads']
plt.subplots(figsize=(5,5))
sns.heatmap(corr,annot= True,cmap = 'Greens',linewidths=2,linecolor='black');
plt.title('HOT Leads Correlation')


# In[185]:


#conclusion of analysis.

#various phases of data analysis including data collection, cleaning and analysis are discussed briefly.

#Explorative data analysis is mainly studied here. 

#For the implementation, Python programming language is used.

#For detailed research, jupyter notebook is used. Different Python libraries and packages are introduced.

# We can see that the Impression ,Clicks and Sales units there are interrelation between them 

#  we can see that when clicks increase Sales also increase .

# The  clicks had the best sales.

# we can see that the DecisionTreeClassifier is used for accuaracy define in dataset 

# we can see that hot leads correlation 

# we can see that warm lead correlation.

# we can see that how the leads are relation between this cliks and impression 

# We can see that the linerregression to increase sales with increase the clicks and impression 

# we can see that the average std deviation of impression very spread due to marketing canablize.

# We can see that the std deviation of clicks is some low than Impression because there is low awarnees of this ads.

# it is all about To analyses how many people who clicked on the advertisement enrolled in our course.

#in that data set we learn about data is there ctc and ctr data analysis after clicking ad .


# In[187]:


#insights


# In[188]:


# In all about analaysis dataset to inform that general marketing and how the people was aware about advertisment 
# this advertisment was 6th june. 
# It main think that there was tuesday is a working day .
# people mindset was to do  workholic or motivated 
# that day they search or aware about cources
# some people was went house from office that time is about 12.am 
# some people go to saw this particular ads but not click .
# to all dataset analysis there was impression was slightly peak but not click this ads .
# some people to aware this ads more information was find to click them this ads then this ads useful for this.
# those people want to sale this course.
# by the analysis is found that there is intereltaion between CTR, and CPC of paticualr advertisement 
# and to analyais of what is hot leads and cold and warm leads genertion of particualar advertisement 
#for the analysis there is 0.733 accuracy of data is obtained it menas that there is 73% customer refer or see the this advertsisment for course .
# i suggest to marketing head to increses ad. and disply reptabley for marketing and awaerness purpose . 
# on the time which choose the have you leads like warm , cold and and hot lead which will be consider.


# In[ ]:




