
# coding: utf-8

# ## <center>Data Structure and Algorithms - Spring'19
# ## <u><center>Spam Email Detection using Naive Bayes Algorithm

# In[68]:


import pandas as pd
import numpy as np
import time
start_time = time.time()


# In[69]:


msg = [line.rstrip() for line in open("C:/Ayush/SMSSpamCollection")]             # Importing Data 
print(len(msg))


# In[70]:


for msg_no,msg in enumerate(msg[:5]):                           # Display top 5 input messages
   print(msg_no,msg)
   print('\n')


# In[71]:


msg=pd.read_csv("C:/Ayush/SMSSpamCollection",sep='\t',names=["labels","message"])        # Display top 5 text messages with labels from the data file.
msg.head()             


# ### Data Analysis

# In[72]:


msg.groupby('labels').describe()               


# In[73]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Text Preprocessing

# In[74]:


import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[75]:


stopwords.words('english')[0:20]          # Commonly used words.


# In[76]:


def txt_pre_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]      


# In[77]:


msg.head(8)                         # Original Dataframe of first 8 rows 


# In[78]:


msg['message'].head(5).apply(txt_pre_process)                  # Tokenization


# In[79]:


from sklearn.feature_extraction.text import CountVectorizer


# In[80]:


bag_of_words_trans = CountVectorizer(analyzer=txt_pre_process).fit(msg['message'])     # Taking one text message and getting its bag-of-words counts as a vector.
print(len(bag_of_words_trans.vocabulary_))


# In[81]:


msg4=msg['message'][2]
print(msg4)


# In[82]:


bow=bag_of_words_trans.transform([msg4])
print(bow)
print(bow.shape)


# #### The above output shows that there are 16 unique words in message 4. All occurs just once.

# In[83]:


msg_bow = bag_of_words_trans.transform(msg['message'])       # checking the bag-of-words counts for the entire SMS corpus


# In[84]:


print('Shape of the Sparse Matrix: ',msg_bow.shape)
print('Amount of non-zero occurences:',msg_bow.nnz)


# In[85]:


spars =(100.0 * msg_bow.nnz/(msg_bow.shape[0]*msg_bow.shape[1]))
print('sparsity:{}'.format(round(spars)))


# In[86]:


from sklearn.feature_extraction.text import TfidfTransformer           
tfidf_transformer=TfidfTransformer().fit(msg_bow)
tfidf = tfidf_transformer.transform(bow)
print(tfidf)


# In[87]:


print(tfidf_transformer.idf_[bag_of_words_trans.vocabulary_['e']])            # Calculating Inverse Document frequency of the word ""
print(tfidf_transformer.idf_[bag_of_words_trans.vocabulary_['Free']])


# In[88]:


msg_tfidf=tfidf_transformer.transform(msg_bow)
print(msg_tfidf.shape)


# In[89]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(msg_tfidf,msg['labels'])


# In[90]:


print('Predicted:',spam_detect_model.predict(tfidf)[0])
print('Expected:',msg.labels[2])


# In[91]:


model_predict = spam_detect_model.predict(msg_tfidf)               # Evaluating the Model wrt the entire dataset
print(model_predict)                                            


# #### Training the model

# In[92]:


from sklearn.model_selection import train_test_split                        # Split dataset into training and testing           
msg_train,msg_test,label_train,label_test = train_test_split(msg['message'],msg['labels'],test_size=0.2)


# In[93]:


print(len(msg_train),len(msg_test),len(label_train),len(label_test))


# In[94]:


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
   ( 'bow',CountVectorizer(analyzer=txt_pre_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB()),
])


# In[95]:


pipeline.fit(msg_train,label_train)


# In[96]:


predictions = pipeline.predict(msg_test)


# In[97]:


print(np.mean(predictions==label_test))


# #### The above output shows that the model has acheived accuracy of about 97%

# In[98]:


print("--- %s seconds ---" % (time.time() - start_time))          # Total Runtime


# 
# 

# ##### Reference for code:
# https://www.kaggle.com/dilip990/spam-ham-detection-using-naive-bayes-classifier
