
# coding: utf-8

# In[57]:


import types
import pandas as pd
import numpy as np
from botocore.client import Config
import ibm_boto3


# In[58]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_da37547e19ed476197be667e2b9b1337 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='E4GRZ7cNKRDzY_TsM-GVSzd3aN0V3B7SF5QuR2ZLZbeC',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_da37547e19ed476197be667e2b9b1337.get_object(Bucket='projectsmartbees-donotdelete-pr-zbxi6qebkjsvh8',Key='smartbees-loan.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()



# In[59]:


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,10].values


# In[60]:


x


# In[61]:


y


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[64]:


from sklearn.linear_model import LogisticRegression


# In[65]:


lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[66]:


y_pred=lr.predict(x_test)


# In[67]:


u_pred=lr.predict(np.array([[34, 9, 180, 1, 8.9, 3, 0, 0, 0, 0]]))


# In[68]:


u_pred


# In[69]:


u_pred=lr.predict(np.array([[48, 23, 114, 2, 3.8, 3, 0, 1, 0, 0]]))


# In[70]:


u_pred


# In[71]:


get_ipython().system(u'pip install watson-machine-learning-client --upgrade')


# In[72]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[73]:


wml_credentials={
    "url":"https://eu-gb.ml.cloud.ibm.com",
    "access_key":"SGOOWDd3b66syOydPhPOM0weS20RrJGHn6XIEKE4Ylxq",
    "username":"fd1a9f16-b246-49e8-99c0-0876fdf39dec",
    "password":"a52798e0-f0aa-462b-ad22-71605e950a61",
    "instance_id":"6a3094bb-7e7a-47be-9adf-c04ddc8cc500"
}


# In[74]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[75]:


model_props={client.repository.ModelMetaNames.AUTHOR_NAME:"SmartBees",
             client.repository.ModelMetaNames.AUTHOR_EMAIL:"tejaswini.ira@gmail.com",
             client.repository.ModelMetaNames.NAME:"Personal Loan Predictor"
}


# In[76]:


model_artifact=client.repository.store_model(lr,meta_props=model_props)


# In[77]:


published_model_uid=client.repository.get_model_uid(model_artifact)


# In[78]:


published_model_uid


# In[79]:


deployment=client.deployments.create(published_model_uid,name="Personal Loan Prediction")


# In[80]:


scoring_endpoint=client.deployments.get_scoring_url(deployment)


# In[81]:


scoring_endpoint

