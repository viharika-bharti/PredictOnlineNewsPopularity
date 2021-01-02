#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[24]:


#The dataset 

#Reading the dataset
data= pd.read_csv(r"C:\Users\vbhar\Desktop\project_575\OnlineNewsPopularity_Dataset\OnlineNewsPopularity.csv")

#dimensions and other details
#data.shape (39644, 61)
#data.info()
#data.describe()

#Removing space character from the variable names
data.columns=data.columns.str.replace(" ","")


# In[25]:


#1)Understanding target variable distribution 

#Checking median since dsitribution is highly skewed and median to covert to binary classification
#data.describe()
data['shares'].median() #1400

plt.subplots(3,1,figsize=(12,10))
plt.subplot(3,1,1)
sns.distplot(data['shares'], hist=True, kde=False)
plt.subplot(3,1,2)
sns.violinplot(data['shares'])
plt.subplot(3,1,3)
sns.scatterplot(data=data, x='timedelta', y='shares')

#Removing highlighy skewed rows
Q1 = data['shares'].quantile(0.25)
Q3 = data['shares'].quantile(0.75)
IQR = Q3 - Q1
LTV= Q1 - (1.5 * IQR)
UTV= Q3 + (1.5 * IQR)
data = data.drop(data[data['shares'] > UTV].index)
data.shape #35103

#After removal distribution
plt.subplots(3,1,figsize=(12,10))
plt.subplot(3,1,1)
sns.distplot(data['shares'], hist=True, kde=False)
plt.subplot(3,1,2)
sns.violinplot(data['shares'])
plt.subplot(3,1,3)
sns.scatterplot(data=data, x='timedelta', y='shares')

#Coverting to a binary classfication problem
#creating new target variable based on median
data['popularity'] = data['shares'].apply(lambda x: 0 if x<1400 else 1)

#distribution of popular and not popular articles
popularity_count=data.groupby("popularity").count()
popularity_count['url'].plot(kind='bar')
plt.xlabel("Popularity", fontsize=12)
plt.ylabel("Count", fontsize=12)
#plt.grid(True)

#popular and unpopular rows
Unpopular=data[data['popularity']==0]
Popular=data[data['popularity']==1]


# In[26]:


#Final number of rows
data.shape #35103


# In[27]:


#2) Data exploration

#a)distribution of articles over days of week
days_of_week = data.columns.values[31:38]
Unpopular_days = Unpopular[days_of_week].sum()
Unpopular_days=np.sort(Unpopular_days, False)
Popular_days = Popular[days_of_week].sum().values
Popular_days=np.sort(Popular_days, False)

fig = plt.figure(figsize = (12,5))
plt.title("Count of popular/unpopular news over different day of week", fontsize = 10)

plt.bar(np.arange(len(days_of_week)),Popular_days,width=0.25,align='center',color='#57334c',label='Popular')
plt.bar(np.arange(len(days_of_week))-0.25,Unpopular_days,width=0.25,align='center',color='#bb8bac',label='Unpopular')

plt.xticks(np.arange(len(days_of_week)),days_of_week)
plt.ylabel('Count',fontsize=12)
plt.xlabel('Days of the Week',fontsize=12)

plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

#b)distribution of articles over category
content = data.columns.values[13:19]
Unpopular_content = Unpopular[content].sum().values
Unpopular_content=np.sort(Unpopular_content, False)
Popular_content = Popular[content].sum().values
Popular_content = np.sort(Popular_content, False)

fig = plt.figure(figsize = (12,5))
plt.title("Count of popular/unpopular news over article category", fontsize = 10)

plt.bar(np.arange(len(content)),Popular_content,width=0.25,align='center',color='#57334c',label='Popular')
plt.bar(np.arange(len(content))-0.25,Unpopular_content,width=0.25,align='center',color='#bb8bac',label='Unpopular')

plt.xticks(np.arange(len(content)),content)
plt.ylabel('Count',fontsize=12)
plt.xlabel('Article category',fontsize=12)

plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

#c)Distibution of popularity based on internal content of article
Internal_content = data[['n_tokens_title','num_hrefs','num_self_hrefs','num_imgs','num_videos']].columns

Unpopular_internal_content = Unpopular[Internal_content].sum().values
Unpopular_internal_content =np.sort(Unpopular_internal_content, False)

Popular_internal_content = Popular[Internal_content].sum().values
Popular_content = np.sort(Popular_internal_content, False)

fig = plt.figure(figsize = (12,5))
plt.title("Count of popular/unpopular news over internal content", fontsize = 10)

plt.bar(np.arange(len(Internal_content)),Popular_internal_content,width=0.25,align='center',color='#57334c',label='Popular')
plt.bar(np.arange(len(Internal_content))-0.25,Unpopular_internal_content,width=0.25,align='center',color='#bb8bac',label='Unpopular')

plt.xticks(np.arange(len(Internal_content)),Internal_content)
plt.ylabel('Count',fontsize=12)
plt.xlabel('Internal Content',fontsize=12)

plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()

#d)Checking correlations between LDA and article categories.
Corr_Colums = data[['data_channel_is_lifestyle','data_channel_is_entertainment','data_channel_is_bus','data_channel_is_socmed','data_channel_is_tech','data_channel_is_world','LDA_00','LDA_01','LDA_02','LDA_03','LDA_04']]
corr_col=Corr_Colums.corr()
plt.figure(figsize=(3,3))
sns.heatmap(corr_col, annot=False, cmap='Blues')


# In[28]:


#Data Preparation
Original_data=data

#1)Drop non predictive columns since they dont contribute to prediciton
data=data.drop(data[['url','timedelta']], axis=1)

#2)Removing rows with zero tokens in content-means we remove articles with no proper wordings-1181 rows removed.
data=data[data['n_tokens_content']!= 0]

#3)Checkinf for null and na values
data.isna().values.any()
data.isnull().values.any()

#data.shape #34103 rows


# In[29]:


#4)Veiwing hihgly correlated varibales with a value greater than 0.9
cor=data.corr()
pd.set_option('max_columns', None)
x = cor[cor>0.8]
plt.figure(figsize=(15,15))
sns.heatmap(data.corr()>0.8,annot=False,cmap="RdYlGn")

data1= data[['n_unique_tokens','n_non_stop_unique_tokens','data_channel_is_bus','data_channel_is_tech','kw_max_min','kw_avg_min','weekday_is_sunday','is_weekend','kw_max_avg','kw_avg_avg','LDA_00','LDA_04','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess','global_sentiment_polarity','global_rate_negative_words',
            'rate_positive_words','rate_negative_words','avg_negative_polarity','min_negative_polarity','title_subjectivity','abs_title_sentiment_polarity']]

plt.figure(figsize=(7,5))
sns.heatmap(data1.corr()>0.7,annot=False,cmap="RdYlGn")

#Removing highly correlted variables with greater than correlation greater than 80%
data= data.drop(data[["n_non_stop_unique_tokens","n_non_stop_words","kw_avg_min",'data_channel_is_world','kw_max_avg','self_reference_min_shares','self_reference_max_shares']],axis=1)

data.shape #now we have 52 variables

#Finally dropping shares from  data.- since we dont need it
data=data.drop(["shares"],axis=1)


# In[30]:


#seperating numerical and categorical columns

#Data numerical has only the numerical columns- target popularity have been removed.
data_numerical=data.drop(["weekday_is_monday","weekday_is_tuesday","weekday_is_wednesday","weekday_is_thursday",
                  "weekday_is_friday","weekday_is_saturday","weekday_is_sunday",                  
                  "data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus",
                  "data_channel_is_socmed","data_channel_is_tech","popularity"],axis=1)

# data_cat dataframe contains the catagoricl features. 
data_cat=data[["weekday_is_monday","weekday_is_tuesday","weekday_is_wednesday","weekday_is_thursday",
             "weekday_is_friday","weekday_is_saturday","weekday_is_sunday",            
             "data_channel_is_lifestyle","data_channel_is_entertainment","data_channel_is_bus",
                  "data_channel_is_socmed","data_channel_is_tech","popularity"]]

#Checking distribution of numerical attributes to decide the method of scaling
data_numerical.describe()
data_numerical.shape #39 columns
data_cat.shape #13 colums

#Checking distributions of each numerical vairbles
#data_numerical.hist(figsize=(25,25))
#we see that the the numerical attributes have different distributions- some with negative values and some with very large ranges.
#data_numerical.boxplot(figsize=(25,25))


# In[31]:


#Outlier treatment on highly skewed numerical variables 

#data_large_range.dist(figsize=(25,25))
#data_large_range.hist()
#sns.distplot(data_numerical['num_videos'], hist=True, kde=False)
#data_numerical['num_videos'].describe()
#data_numerical[data_numerical['num_videos']>20].count()
##data_numerical['n_non_stop_words']=data_numerical['n_non_stop_words'].round(decimals=5)

data_numerical['kw_max_min'] = np.where(data_numerical.kw_max_min > 1000, 1000, data_numerical.kw_max_min)
data_numerical['kw_min_max'] = np.where(data_numerical.kw_min_max > 7300, 7300, data_numerical.kw_min_max)
data_numerical= data_numerical.drop(['kw_max_max'], axis=1)
data_numerical['kw_avg_max'] = np.where(data_numerical.kw_avg_max > 324200, 324200, data_numerical.kw_avg_max)
data_numerical['kw_min_avg'] = np.where(data_numerical.kw_min_avg > 1974.827586, 1974.827586, data_numerical.kw_min_avg)
data_numerical['kw_avg_avg'] = np.where(data_numerical.kw_avg_avg > 3444.746660, 3444.746660, data_numerical.kw_avg_avg)
#data_numerical['self_reference_min_shares'] = np.where(data_numerical.self_reference_min_shares >2500.000,2500.000, data_numerical.self_reference_min_shares)
#data_numerical['self_reference_max_shares'] = np.where(data_numerical.self_reference_max_shares >7600.000,7600.0000, data_numerical.self_reference_max_shares)
data_numerical['min_positive_polarity'] = np.where(data_numerical.min_positive_polarity >0.3, 0.3, data_numerical.min_positive_polarity)
#data_numerical= data_numerical.drop(['n_non_stop_words'], axis=1)
data_numerical['n_tokens_content'] = np.where(data_numerical.n_tokens_content >1400, 1400, data_numerical.n_tokens_content)
data_numerical['num_hrefs'] = np.where(data_numerical.num_hrefs >50, 50, data_numerical.num_hrefs)
data_numerical['num_imgs'] = np.where(data_numerical.num_imgs >40, 40, data_numerical.num_imgs)
data_numerical['num_self_hrefs'] = np.where(data_numerical.num_self_hrefs >30, 30, data_numerical.num_self_hrefs)
data_numerical['num_videos'] = np.where(data_numerical.num_videos >20, 20, data_numerical.num_videos)

#checking distribution after outlier treatment
#data_numerical.hist(figsize=(25,25))

#data_numerical.shape 38 columns


# In[11]:


#Keeping a reference
data_numerical_copy=data_numerical

#Based on the wide distributions we choose first remove outliers using RobustScaler
from sklearn import preprocessing
scaler = preprocessing.RobustScaler()
scaler.fit(data_numerical)
data_num_scaled_1=scaler.transform(data_numerical)

#Then we put all the variables within a range of 0 to 1 using MinMaxScaler
from sklearn import preprocessing
scaler_2 = preprocessing.MinMaxScaler()
scaler_2.fit(data_num_scaled_1)
data_num_scaled=scaler_2.transform(data_num_scaled_1)

#Coverting the scaled values to a dataframe
data_num_scaled_df=(pd.DataFrame(data_num_scaled,columns=data_numerical.columns))

#Checking the distribution of the scaled numerical variables:
#data_num_scaled_df.hist(figsize=(25,25))
#data_num_scaled_df.boxplot(figsize=(25,25))

#data_num_scaled_df.shape 38 colums


# In[32]:


#PCA

#1)##PCA  for LDA set of avriables
data_LDA_scaled= data_num_scaled_df[['LDA_00','LDA_01','LDA_02','LDA_03','LDA_04']]

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(data_LDA_scaled)
x_data_LDA = pca.transform(data_LDA_scaled)
x_data_LDA.shape
print(pca.explained_variance_ratio_.cumsum())

#We see that 4 out of 5 compenents retain information, hence PCA is not useful for LDA set of variables.


# In[13]:


data_num_scaled_df.shape


# In[33]:


#PCA for keyword related vairbales

data_key_scaled= data_num_scaled_df[['kw_min_min','kw_max_min','kw_min_max','kw_avg_max','kw_min_avg','kw_avg_avg']]

from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(data_key_scaled)
x_data_key = pca.transform(data_key_scaled)
x_data_key.shape
print(pca.explained_variance_ratio_.cumsum())


#All PCA columns
PCA_key_columns = pd.DataFrame(data = x_data_key, columns = ['PC_K1', 'PC_K2','PC_K3','PC_K4','PC_K5','PC_K6'])
#PCA_key_columns

#retaining first 5(99.46% information)
PCA_key_columns_5= PCA_key_columns.drop(['PC_K6'], axis=1)
PCA_key_columns_5


# In[34]:


#PCA for keyword related vairbales

data_NLP_scaled= data_num_scaled_df[['global_subjectivity','global_sentiment_polarity','global_rate_positive_words','global_rate_negative_words',
                                    'rate_positive_words','rate_negative_words','avg_positive_polarity','min_positive_polarity','max_positive_polarity',
                                    'avg_negative_polarity','min_negative_polarity','max_negative_polarity','title_subjectivity','title_sentiment_polarity',
                                    'abs_title_subjectivity','abs_title_sentiment_polarity']]

pca = PCA(n_components=16)
pca.fit(data_NLP_scaled)
x_data_NLP= pca.transform(data_NLP_scaled)
x_data_NLP.shape
print(pca.explained_variance_ratio_.cumsum())

PCA_NLP_columns = pd.DataFrame(data = x_data_NLP, columns = ['PC_NLP_1', 'PC_NLP_2','PC_NLP_3','PC_NLP_4','PC_NLP_5','PC_NLP_6',
                                                             'PC_NLP_7','PC_NLP_8','PC_NLP_9','PC_NLP_10','PC_NLP_11','PC_NLP_12',
                                                            'PC_NLP_13','PC_NLP_14','PC_NLP_15','PC_NLP_16'])


#retaining 11 colums that contain 99% information
PCA_NLP_columns = PCA_NLP_columns.drop(['PC_NLP_12','PC_NLP_13','PC_NLP_14','PC_NLP_15','PC_NLP_16'],axis=1)
PCA_NLP_columns


# In[35]:


#using the new PCA columns in our dataset

#Removing NLP and keyword related variable sets
data_num_scaled_PCA=data_num_scaled_df.drop(['kw_min_min','kw_max_min','kw_min_max','kw_avg_max','kw_min_avg','kw_avg_avg'], axis=1)
data_num_scaled_PCA=data_num_scaled_PCA.drop(['global_subjectivity','global_sentiment_polarity','global_rate_positive_words','global_rate_negative_words',
                                    'rate_positive_words','rate_negative_words','avg_positive_polarity','min_positive_polarity','max_positive_polarity',
                                    'avg_negative_polarity','min_negative_polarity','max_negative_polarity','title_subjectivity','title_sentiment_polarity',
                                    'abs_title_subjectivity','abs_title_sentiment_polarity'], axis=1)

#Concatening the 5 PCA from keyword related variables and 11 PCA from NLP related variables
Scaled_PCA_1= pd.concat([data_num_scaled_PCA,PCA_key_columns_5], axis=1)
df_final_PCA= pd.concat([Scaled_PCA_1,PCA_NLP_columns], axis=1)

#attaching the categorical variables columns.
l1=df_final_PCA.values.tolist()
l2=data_cat.values.tolist()
for i in range(len(l1)):
    l1[i].extend(l2[i])

df_final=pd.DataFrame(l1,columns=df_final_PCA.columns.tolist()+data_cat.columns.tolist())


# In[36]:


#final dataset that will be used for modelling
#df_final.hist(figsize=(30,30))
df_final.shape  #45 variables


# In[ ]:


#KNN Model- baseline

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

df_final_KNN=df_final

X = df_final_KNN.drop(['popularity'],axis=1)
#drop target variable
y= df_final_KNN['popularity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Chaciking dimension of train and test sets
#print('Training Features Shape:', X_train.shape)
#print('Training Labels Shape:', y_train.shape)
#print('Testing Features Shape:', X_test.shape)
#print('Testing Labels Shape:', y_test.shape)

#KNN for k=1 to 20
k_range= range(1,20)
scores={}
scores_list=[]
error_rate=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    predict_test=knn.predict(X_test)
    scores[k]=metrics.accuracy_score(y_test,predict_test)
    scores_list.append(metrics.accuracy_score(y_test,predict_test))
    error_rate.append(np.mean(predict_test != y_test))

scores_list
error_rate

#Elbow plot to decide best k
plt.figure(figsize=(7,4))
plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[44]:


#Running Knn at K=3 

knn=KNeighborsClassifier(n_neighbors=3)

#fit model to train data
knn.fit(X_train,y_train)

#opredict on training
predict_train = knn.predict(X_train)
#print('Target on train data',predict_train) 

#accuracy on training set
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ',round(accuracy_train,2))

# predict the target on the test dataset
predict_test = knn.predict(X_test)
#print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_test,predict_test)
print('accuracy_score on test dataset : ',round(accuracy_test,2))

print('confusion matrix on test data')
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))


# In[43]:


#Running- Naive Bayaes

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df_final_NB=df_final

#drop target variable
X = df_final_NB.drop(['popularity'],axis=1)
y= df_final_NB['popularity']

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Chaciking dimension of train and test sets
#print('Training Features Shape:', X_train.shape)
#print('Training Labels Shape:', y_train.shape)
##print('Testing Features Shape:', X_test.shape)
#print('Testing Labels Shape:', y_test.shape)

#Create a model instance
NB_model = GaussianNB()

#fit model to train data
NB_model.fit(X_train,y_train)

#opredict on training
predict_train = NB_model.predict(X_train)
#print('Target on train data',predict_train) 

#accuracy on training set
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', round(accuracy_train,2))

# predict the target on the test dataset
predict_test = NB_model.predict(X_test)
#print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_test,predict_test)
print('accuracy_score on test dataset : ', round(accuracy_test,2))

print('confusion matrix on test data')
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))


# In[ ]:




