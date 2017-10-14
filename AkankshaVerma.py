
# coding: utf-8

# # CREDIT FRAUD ANALYSIS THROUGH PYTHON

# # Packages Used

# In[1]:

# data analysis and wrangling
import arff
import pandas as pd

# visualization
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc, roc_auc_score
from graphviz import Source
from IPython.display import SVG
import pylab as pl


# Dataset is available in an arff file which needs to be converted into a dataframe for our analysis. To see the arff file format refer appendix.

# # Data Pre-processing

# ## Creating a dataframe from arff file

# In[2]:

#load the data in arff file as list
list_data = list(arff.load('credit_fruad.arff'))
#convert the list into dataframe
df = pd.DataFrame(list_data)
#have a look at the first 3 rows of the dataframe
df.head(3)


# In[3]:

#know the dimensions of the dataframe created
df.shape


# In[4]:

#create a list with the names of columns to be given to the dataframe
names = ["over_draft", "credit_usage", "credit_history", "purpose", "current_balance", "Average_Credit_Balance", "employment", "location", "personal_status", "other_parties", "residence_since", "property_magnitude", "cc_age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker", "class"]


# In[5]:

#assign the list of column names to the dataframe
df.columns = names
#have a look at the changes made to the dataframe
df.head(1)


# The dataframe with few rows is attached in appendix to have a look.

# In[6]:

#know the distribution of class variable as to how many loan applicants are good and how many are bad
df['class'].value_counts()


# In[7]:

#know the details of the dataframe
df.info()


# This tells us there are no null values in our dataset. If there were missing values, we would have to assign dummy values in order to have a complete dataframe. Otherwise the tests cannot be applied for analysis when there are NaN values.

# ## Checking for categories existing in numerical variables

# In[8]:

df['credit_usage'].nunique()
#this is a continuous variable


# In[9]:

df['current_balance'].nunique()
# This variable is not very useful because average_credit_balance variable already exists


# In[10]:

#deleting the 'current_balance' column
df.drop('current_balance',axis=1,inplace=True)


# In[11]:

print(df['location'].nunique())
df['location'].value_counts()
# There exists a category here with values [1,2,3,4]


# In[12]:

print(df['residence_since'].nunique())
df['residence_since'].value_counts()
# There exists a category here with values [1,2,3,4] 


# In[13]:

df['cc_age'].nunique()
#this is a continuous variable


# In[14]:

print(df['existing_credits'].nunique())
df['existing_credits'].value_counts()
# There exists a category here with values [1,2,3,4]


# In[15]:

print(df['num_dependents'].nunique())
df['num_dependents'].value_counts()
# There exists a category here with values [1,2]


# ## Changing class variable from text to numerical categories

# We change the class variable as 0 = good and 1 = bad using if statement inside function and then apply command.

# In[16]:

df["class"].unique()


# In[17]:

def if_func_class (row):
        c = row['class']
        if c == 'good':
            return 0
        else:
            return 1


# In[18]:

df["class"] = df.apply(if_func_class, axis = "columns")


# ## Converting categorical variables to numerical values

# In[19]:

#making a copy of the dataframe to convert it into numeric values for categories
df_num = df.copy()


# ### over_draft

# In[20]:

df_num["over_draft"].unique()


# In[21]:

def if_func_od (row):
        od = row['over_draft']
        if od == "'no checking'":
            return 1
        elif od == "'<0'":
            return 2
        elif od == "'0<=X<200'":
            return 3
        else:
            return 4


# In[22]:

df_num["over_draft"] = df_num.apply(if_func_od, axis = "columns")


# ### credit_history

# In[23]:

df_num["credit_history"].unique()


# In[24]:

def if_func_ch (row):
        ch = row['credit_history']
        if ch == "'no credits/all paid'":
            return 1
        elif ch == "'all paid'":
            return 2
        elif ch == "'existing paid'":
            return 3
        elif ch == "'delayed previously'":
            return 4
        else:
            return 5


# In[25]:

df_num["credit_history"] = df_num.apply(if_func_ch, axis = "columns")


# ### purpose

# In[26]:

df_num["purpose"].unique()


# In[27]:

def if_func_purpose (row):
        pr = row['purpose']
        if pr == "radio/tv":
            return 1
        elif pr == "education":
            return 2
        elif pr == "furniture/equipment":
            return 3
        elif pr == "'new car'":
            return 4
        elif pr == "'used car'":
            return 5
        elif pr == "business":
            return 6
        elif pr == "'domestic appliance'":
            return 7
        elif pr == "repairs":
            return 8
        elif pr == "retraining":
            return 9
        else:
            return 10


# In[28]:

df_num["purpose"] = df_num.apply(if_func_purpose, axis = "columns")


# ### Average_Credit_Balance

# In[29]:

df_num["Average_Credit_Balance"].unique()


# In[30]:

def if_func_acb (row):
        acb = row['Average_Credit_Balance']
        if acb == "'<100'":
            return 1
        elif acb == "'100<=X<500'":
            return 2
        elif acb == "'500<=X<1000'":
            return 3
        elif acb == "'>=1000'":
            return 4
        else:
            return 5


# In[31]:

df_num["Average_Credit_Balance"] = df_num.apply(if_func_acb, axis = "columns")


# ### employment

# In[32]:

df_num["employment"].unique()


# In[33]:

def if_func_employment (row):
        ep = row['employment']
        if ep == "'<1'":
            return 1
        elif ep == "'1<=X<4'":
            return 2
        elif ep == "'4<=X<7'":
            return 3
        elif ep == "'>=7'":
            return 4
        else:
            return 5


# In[34]:

df_num["employment"] = df_num.apply(if_func_employment, axis = "columns")


# ### personal_status

# In[35]:

df_num["personal_status"].unique()


# In[36]:

def if_func_ps (row):
        ps = row['personal_status']
        if ps == "'female div/dep/mar'":
            return 1
        elif ps == "'male single'":
            return 2
        elif ps == "'male mar/wid'":
            return 3
        else:
            return 4


# In[37]:

df_num["personal_status"] = df_num.apply(if_func_ps, axis = "columns")


# ### other_parties

# In[38]:

df_num["other_parties"].unique()


# In[39]:

def if_func_op (row):
        op = row['other_parties']
        if op == 'guarantor':
            return 1
        elif op == "'co applicant'":
            return 2
        else:
            return 3


# In[40]:

df_num["other_parties"] = df_num.apply(if_func_op, axis = "columns")


# ### property_magnitude

# In[41]:

df_num["property_magnitude"].unique()


# In[42]:

def if_func_pm (row):
        pm = row['property_magnitude']
        if pm == 'car':
            return 1
        elif pm == "'life insurance'":
            return 2
        elif pm == "'real estate'":
            return 3
        else:
            return 4


# In[43]:

df_num["property_magnitude"] = df_num.apply(if_func_pm, axis = "columns")


# ### other_payment_plans

# In[44]:

df_num["other_payment_plans"].unique()


# In[45]:

def if_func_opp (row):
        opp = row['other_payment_plans']
        if opp == 'bank':
            return 1
        elif opp == 'stores':
            return 2
        else:
            return 3


# In[46]:

df_num["other_payment_plans"] = df_num.apply(if_func_opp, axis = "columns")


# ### housing

# In[47]:

df_num["housing"].unique()


# In[48]:

def if_func_housing (row):
        h = row['housing']
        if h == 'own':
            return 1
        elif h == 'rent':
            return 2
        else:
            return 3


# In[49]:

df_num["housing"] = df_num.apply(if_func_housing, axis = "columns")


# ### job

# In[50]:

df_num["job"].unique()


# In[51]:

def if_func_job (row):
        j = row['job']
        if j == "'unemp/unskilled non res'":
            return 1
        elif j == "'unskilled resident'":
            return 2
        elif j == 'skilled':
            return 3
        else:
            return 4


# In[52]:

df_num["job"] = df_num.apply(if_func_job, axis = "columns")


# ### own_telephone

# In[53]:

df_num["own_telephone"].unique()


# In[54]:

def if_func_ot (row):
        ot = row['own_telephone']
        if ot == 'yes':
            return 0
        else:
            return 1


# In[55]:

df_num["own_telephone"] = df_num.apply(if_func_ot, axis = "columns")


# ### foreign_worker

# In[56]:

df_num["foreign_worker"].unique()


# In[57]:

def if_func_fw (row):
        fw = row['foreign_worker']
        if fw == 'no':
            return 0
        else:
            return 1


# In[58]:

df_num["foreign_worker"] = df_num.apply(if_func_fw, axis = "columns")


# ## Changing specific columns to category data type

# In[59]:

df_num.columns


# In[60]:

for col in ['over_draft', 'credit_history', 'purpose',
       'Average_Credit_Balance', 'employment', 'location',
       'personal_status', 'other_parties', 'residence_since',
       'property_magnitude', 'other_payment_plans', 'housing',
       'existing_credits', 'job', 'num_dependents', 'own_telephone',
       'foreign_worker', 'class']:
    df_num[col] = df_num[col].astype('category')


# In[61]:

df_num.info()


# ## Splitting test and train dataframe (df_num)

# train_df, test_df = train_test_split(df_num, test_size=0.2)

# Creating files in the local system as train.csv and test.csv

# train_df.to_csv('train.csv')

# test_df.to_csv('test.csv')

# # Data Analysis

# ## Checking the distribution of categorical features along with continuous

# In[62]:

a = df.describe(include='all')
a


# In[63]:

a.loc['%',:] = a.loc['freq',:]/a.loc['count',:]*100
a


# ## Analyzing by pivoting features

# Now we analyze all the categorical variables, one by one, to know the proportion of bad creditors within each category. We do this by using the groupby command and we take out the mean. This is possible because we have given the value for bad creditors as 1 and good creditors as 0. The reason for keeping bad creditors as 1 and not good creditors as 1 is because we are intrested in analysing and predicting the defaulters from our data. So through this command, the sum of bad creditors within a particular category are divided with the total count of good and bad creditors in that category.

# In[64]:

#over_draft
df[['over_draft', 'class']].groupby(['over_draft'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[65]:

#credit_history
df[['credit_history', 'class']].groupby(['credit_history'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[66]:

#purpose
df[['purpose', 'class']].groupby(['purpose'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[67]:

#Average_Credit_Balance
df[['Average_Credit_Balance', 'class']].groupby(['Average_Credit_Balance'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[68]:

#employment
df[['employment', 'class']].groupby(['employment'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[69]:

#location
df[['location', 'class']].groupby(['location'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[70]:

#personal_status
df[['personal_status', 'class']].groupby(['personal_status'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[71]:

#other_parties
df[['other_parties', 'class']].groupby(['other_parties'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[72]:

#residence_since
df[['residence_since', 'class']].groupby(['residence_since'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[73]:

#property_magnitude
df[['property_magnitude', 'class']].groupby(['property_magnitude'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[74]:

#other_payment_plans
df[['other_payment_plans', 'class']].groupby(['other_payment_plans'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[75]:

#housing
df[['housing', 'class']].groupby(['housing'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[76]:

#existing_credits
df[['existing_credits', 'class']].groupby(['existing_credits'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[77]:

#job
df[['job', 'class']].groupby(['job'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[78]:

#num_dependents
df[['num_dependents', 'class']].groupby(['num_dependents'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[79]:

#own_telephone
df[['own_telephone', 'class']].groupby(['own_telephone'], as_index=False).mean().sort_values(by='class', ascending=False)


# In[80]:

#foreign_worker
df[['foreign_worker', 'class']].groupby(['foreign_worker'], as_index=False).mean().sort_values(by='class', ascending=False)


# Few important obervations :
# Variables like credit_history, employment, other_parties, property_magnitude and foriegn_worker seem to have more impact on the class for few categories within the variable than others. So prediction could be made according to these variables.

# ## Analysis taking all the variables

# In[81]:

train = pd.read_csv('train.csv')


# In[82]:

test = pd.read_csv('test.csv')


# In[83]:

train.drop('Unnamed: 0',axis=1,inplace=True)


# In[84]:

test.drop('Unnamed: 0',axis=1,inplace=True)


# In[85]:

test.shape


# In[86]:

X_train = train.drop("class", axis=1)
Y_train = train["class"]
X_test  = test.drop("class", axis=1)
Y_test = test["class"]
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[87]:

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
print(logreg.coef_)
print(logreg.intercept_)
Y_pred_logi = logreg.predict(X_test)


# In[88]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred_logi)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test, Y_pred_logi))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Logistic Regression Model")
plt.show()


# In[89]:

cm_logi = pd.crosstab(Y_test, Y_pred_logi, rownames=['True'], colnames=['Predicted'], margins=True)
cm_logi


# In[90]:

cost_logi = cm_logi.iloc[1,0]*5 + cm_logi.iloc[0,1]*1
cost_logi


# In[91]:

# Decision Tree

decision_tree = DecisionTreeClassifier(random_state=100)
decision_tree.fit(X_train, Y_train)
Y_pred_deci = decision_tree.predict(X_test)


# Plotting the Decision Tree

# graph = Source(export_graphviz(decision_tree, out_file=None, feature_names=X_train.columns))

# SVG(graph.pipe(format='svg')) 

# Output available in appendix.

# In[92]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred_deci)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test, Y_pred_deci))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Decision Tree Model")
plt.show()


# In[93]:

cm_deci = pd.crosstab(Y_test, Y_pred_deci, rownames=['True'], colnames=['Predicted'], margins=True)
cm_deci


# In[94]:

cost_deci = cm_deci.iloc[1,0]*5 + cm_deci.iloc[0,1]*1
cost_deci


# In[95]:

#importance plot for decision tree
importances_dt = decision_tree.feature_importances_
df_dt_imp = pd.DataFrame(X_train.columns)
df_dt_imp['importance'] = importances_dt
df_dt_imp.columns = ['variables','importance']
df_dt_imp.sort_values(by='importance',ascending=False,inplace=True)
print(df_dt_imp)
df_dt_imp.plot(kind="bar",x=df_dt_imp['variables'],title="Importance Plot for Decision Tree")


# In[96]:

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100,random_state=100)
random_forest.fit(X_train, Y_train)
Y_pred_rand = random_forest.predict(X_test)


# In[97]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred_rand)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test, Y_pred_rand))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Random Forest Model")
plt.show()


# In[98]:

cm_rand = pd.crosstab(Y_test, Y_pred_rand, rownames=['True'], colnames=['Predicted'], margins=True)
cm_rand


# In[99]:

cost_rand = cm_rand.iloc[1,0]*5 + cm_rand.iloc[0,1]*1
cost_rand


# In[100]:

#importance plot for random forest
importances_rf = random_forest.feature_importances_
df_rf_imp = pd.DataFrame(X_train.columns)
df_rf_imp['importance'] = importances_rf
df_rf_imp.columns = ['variables','importance']
df_rf_imp.sort_values(by='importance',ascending=False,inplace=True)
print(df_rf_imp)
df_rf_imp.plot(kind="bar",x=df_rf_imp['variables'],title="Importance Plot for Random Forest")


# ## Analysis taking only important variables observed after analysing through pivoting

# In[101]:

train5 = train[['credit_history','employment','other_parties','property_magnitude','foreign_worker','class']]


# In[102]:

test5 = test[['credit_history','employment','other_parties','property_magnitude','foreign_worker','class']]


# In[103]:

X_train5 = train5.drop("class", axis=1)
Y_train5 = train5["class"]
X_test5  = test5.drop("class", axis=1)
Y_test5 = test5["class"]
X_train5.shape, Y_train5.shape, X_test5.shape, Y_test5.shape


# In[104]:

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train5, Y_train5)
print(logreg.coef_)
print(logreg.intercept_)
Y_pred_log = logreg.predict(X_test5)


# In[105]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test5, Y_pred_log)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test5, Y_pred_log))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Logistic Regression Model")
plt.show()


# In[106]:

cm_log = pd.crosstab(Y_test5, Y_pred_log, rownames=['True'], colnames=['Predicted'], margins=True)
cm_log


# In[107]:

cost_log = cm_log.iloc[1,0]*5 + cm_log.iloc[0,1]*1
cost_log


# In[108]:

# Decision Tree

decision_tree = DecisionTreeClassifier(random_state=100)
decision_tree.fit(X_train5, Y_train5)
Y_pred_dec = decision_tree.predict(X_test5)


# Plotting the Decision Tree

# graph = Source(export_graphviz(decision_tree, out_file=None, feature_names=X_train5.columns))

# SVG(graph.pipe(format='svg'))

# Output available in appendix.

# In[109]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test5, Y_pred_dec)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test5, Y_pred_dec))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Decision Tree Model")
plt.show()


# In[110]:

cm_dec = pd.crosstab(Y_test5, Y_pred_dec, rownames=['True'], colnames=['Predicted'], margins=True)
cm_dec


# In[111]:

cost_dec = cm_dec.iloc[1,0]*5 + cm_dec.iloc[0,1]*1
cost_dec


# In[112]:

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100,random_state=100)
random_forest.fit(X_train5, Y_train5)
Y_pred_ran = random_forest.predict(X_test5)


# In[113]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test5, Y_pred_ran)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test5, Y_pred_ran))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Random Forest Model")
plt.show()


# In[114]:

cm_ran = pd.crosstab(Y_test5, Y_pred_ran, rownames=['True'], colnames=['Predicted'], margins=True)
cm_ran


# In[115]:

cost_ran = cm_ran.iloc[1,0]*5 + cm_ran.iloc[0,1]*1
cost_ran


# ## Taking only the important variables by random forest importance

# In[116]:

train6 = train[['cc_age','credit_usage','over_draft','credit_history','purpose','employment','class']]


# In[117]:

test6 = test[['cc_age','credit_usage','over_draft','credit_history','purpose','employment','class']]


# In[118]:

X_train6 = train6.drop("class", axis=1)
Y_train6 = train6["class"]
X_test6  = test6.drop("class", axis=1)
Y_test6 = test6["class"]
X_train6.shape, Y_train6.shape, X_test6.shape, Y_test6.shape


# In[119]:

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train6, Y_train6)
print(logreg.coef_)
print(logreg.intercept_)
Y_pred_lr = logreg.predict(X_test6)


# In[120]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test6, Y_pred_lr)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test6, Y_pred_lr))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Logistic Regression Model")
plt.show()


# In[121]:

cm_lr = pd.crosstab(Y_test6, Y_pred_lr, rownames=['True'], colnames=['Predicted'], margins=True)
cm_lr


# In[122]:

cost_lr = cm_lr.iloc[1,0]*5 + cm_lr.iloc[0,1]*1
cost_lr


# In[123]:

# Decision Tree

decision_tree = DecisionTreeClassifier(random_state=100)
decision_tree.fit(X_train6, Y_train6)
Y_pred_dt = decision_tree.predict(X_test6)


# Plotting the Decision Tree

# graph = Source(export_graphviz(decision_tree, out_file=None, feature_names=X_train6.columns))

# SVG(graph.pipe(format='svg'))

# Output available in appendix

# In[124]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test6, Y_pred_dt)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test6, Y_pred_dt))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Decision Tree Model")
plt.show()


# In[125]:

cm_dt = pd.crosstab(Y_test6, Y_pred_dt, rownames=['True'], colnames=['Predicted'], margins=True)
cm_dt


# In[126]:

cost_dt = cm_dt.iloc[1,0]*5 + cm_dt.iloc[0,1]*1
cost_dt


# In[127]:

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100,random_state=100)
random_forest.fit(X_train6, Y_train6)
Y_pred_rf = random_forest.predict(X_test6)


# In[128]:

#roc & auc score
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test6, Y_pred_rf)
print (auc(false_positive_rate, true_positive_rate))
print (roc_auc_score(Y_test6, Y_pred_rf))
x = false_positive_rate
y = true_positive_rate
# This is the ROC curve
plt.plot(x,y)
plt.title("ROC curve for Random Forest Model")
plt.show()


# In[129]:

cm_rf = pd.crosstab(Y_test6, Y_pred_rf, rownames=['True'], colnames=['Predicted'], margins=True)
cm_rf


# In[130]:

cost_rf = cm_rf.iloc[1,0]*5 + cm_rf.iloc[0,1]*1
cost_rf


# ## Automating by creating a function

# Now that the data is analysed and various statistical models are applied, a function can be created which can be applied every time such analysis has to be done. The function will also help in analysing the same dataset with different set of explanatory variables to know the least cost for prediction. And also few other models could be added later on in the function according to the need of the analyst. This will save time and will help in better prediction while comparing the cost from different models used for prediction.

# About the function: The parameters used within the function are train_df- this is to provide the train dataset to the function. test_df- this is to provide the test dataset to the function. response_variable- this is to give the name of the column which has to be predicted (with inverted comma) and which is dependent on the values of the other columns. cost_of_1_predicted_0=1- this is to give the cost of error where 1 value of response variable is predicted as 0. cost_of_0_predicted_1=1- this is to give the cost of error where 0 value of response variable is predicted as 1. By default the cost parameters are set as 1 in order to give equal weightage to both the errors in prediction.

# The output of the function: Logistic Regression Model- The output of the function gives the coeficients and intercept of the logistic regression line. Then it shows the AUC score which is converted to percentage and it also plots the ROC curve. Then is shows the confusion matrix and the cost for predicted error. Decision Tree Model- The output shows the AUC score which is converted to percentage and it also plots the ROC curve. Then is shows the confusion matrix and the cost for predicted error. Random Forest Model- The output shows the AUC score which is converted to percentage and it also plots the ROC curve. Then is shows the confusion matrix and the cost for predicted error. After that it shows the table for camparison of all the models on the basis of cost and AUC score. In the end the output plots the importance of variables according to decision tree model and random forest model seperately.

# Few assumptions for using the function created: The dataset has be already split into train and test dataframe. The response variable has to have binary values. All the variables in the dataset have to have numeric values. If there are categories, those should be first converted into numeric values before applying the function.

# ## Function for statistical analysis

# In[131]:

def func (train_df,test_df,response_variable,cost_of_1_predicted_0=1,cost_of_0_predicted_1=1):
    X_train_df = train_df.drop(response_variable, axis=1)
    Y_train_df = train_df[response_variable]
    X_test_df  = test_df.drop(response_variable, axis=1)
    Y_test_df = test_df[response_variable]
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train_df, Y_train_df)
    print("The coeficients of logistic regression line are:")
    print(logreg.coef_)
    print(" ")    
    print("The intercept for logistic regression line is:")
    print(logreg.intercept_)
    print(" ")    
    Y_pred_log = logreg.predict(X_test_df)
    #roc & auc score
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test_df, Y_pred_log)
    auc_log = roc_auc_score(Y_test_df, Y_pred_log) * 100
    print ("The AUC score for Logistic Regression Model is %2f"%(auc_log))
    x = false_positive_rate
    y = true_positive_rate
    # This is the ROC curve
    plt.plot(x,y)
    plt.title("ROC curve for Logistic Regression Model")
    plt.show()
    cm_lr = pd.crosstab(Y_test_df, Y_pred_log, rownames=['True'], colnames=['Predicted'], margins=True)
    print("Cost Matrix for Logistic Regression:")
    print(cm_lr)
    print(" ")
    cost_lr = cm_lr.iloc[1,0]*cost_of_1_predicted_0 + cm_lr.iloc[0,1]*cost_of_0_predicted_1
    print("The Cost for Lostic Regression Model is %2f"%cost_lr)
    print(" ")    
    # Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=100)
    decision_tree.fit(X_train_df, Y_train_df)
    Y_pred_dec = decision_tree.predict(X_test_df)
    #roc & auc score
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test_df, Y_pred_dec)
    auc_dec = roc_auc_score(Y_test_df, Y_pred_dec) * 100
    print ("The AUC score for Decision Tree Model is %2f"%(auc_dec))
    x = false_positive_rate
    y = true_positive_rate
    # This is the ROC curve
    plt.plot(x,y)
    plt.title("ROC curve for Decision Tree Model")
    plt.show()
    cm_dt = pd.crosstab(Y_test_df, Y_pred_dec, rownames=['True'], colnames=['Predicted'], margins=True)
    print("Cost Matrix for Decision Tree:")
    print(cm_dt)
    print(" ")    
    cost_dt = cm_dt.iloc[1,0]*cost_of_1_predicted_0 + cm_dt.iloc[0,1]*cost_of_0_predicted_1
    print("The Cost for Decision Tree Model is %2f"%cost_dt)
    print(" ")       
    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100,random_state=100)
    random_forest.fit(X_train_df, Y_train_df)
    Y_pred_ran = random_forest.predict(X_test_df)
    #roc & auc score
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test_df, Y_pred_ran)
    auc_ran = roc_auc_score(Y_test_df, Y_pred_ran) * 100
    print ("The AUC score for Random Forest Model is %2f"%(auc_ran))
    x = false_positive_rate
    y = true_positive_rate
    # This is the ROC curve
    plt.plot(x,y)
    plt.title("ROC curve for Random Forest Model")
    plt.show()
    print(" ")
    cm_rf = pd.crosstab(Y_test_df, Y_pred_ran, rownames=['True'], colnames=['Predicted'], margins=True)
    print("Cost Matrix for Random Forest:")
    print(cm_rf)
    print(" ")
    cost_rf = cm_rf.iloc[1,0]*cost_of_1_predicted_0 + cm_rf.iloc[0,1]*cost_of_0_predicted_1
    print("The Cost for Random Forest Model is %2f"%cost_rf)
    print(" ")
    models = pd.DataFrame({'Models':['Logistic Regression', 'Decision Tree', 
              'Random Forest']})
    models['Cost'] = [cost_lr, cost_dt, cost_rf]
    models['auc_score'] = [auc_log, auc_dec, auc_ran]
    print("The cost for various models is as follows:")
    print(models)
    print(" ")
    importances_dt = decision_tree.feature_importances_
    df_dt_imp = pd.DataFrame(X_train_df.columns)
    df_dt_imp['importance'] = importances_dt
    df_dt_imp.columns = ['variables','importance']
    df_dt_imp.sort_values(by='importance',ascending=False,inplace=True)
    print(df_dt_imp.plot(kind="bar",x=df_dt_imp['variables'],title="Importance Plot for Decision Tree"))
    print(" ")
    importances_rf = random_forest.feature_importances_
    df_rf_imp = pd.DataFrame(X_train_df.columns)
    df_rf_imp['importance'] = importances_rf
    df_rf_imp.columns = ['variables','importance']
    df_rf_imp.sort_values(by='importance',ascending=False,inplace=True)
    print(df_rf_imp.plot(kind="bar",x=df_rf_imp['variables'],title="Importance Plot for Random Forest"))
    


# In[132]:

func(train,test,"class",5,1)


# In[133]:

func(train5,test5,"class",5,1)


# In[134]:

func(train6,test6,"class",5,1)


# ## Taking only the important variables by decision tree importance

# Function demo on set of variables other than the ones used before creating the function. This shows how easily the models can be re-applied on different set of variables from the dataset in just three line of code.

# In[135]:

train_dt = train[['cc_age','over_draft','credit_usage','residence_since','Average_Credit_Balance','employment','class']]
test_dt = test[['cc_age','over_draft','credit_usage','residence_since','Average_Credit_Balance','employment','class']]
func(train_dt,test_dt,"class",5,1)


# ## Testing the function on another dataset (bankloan.csv)

# In[136]:

df_bnk = pd.read_csv('bankloan.csv')


# In[137]:

df_bnk.head(1)


# In[138]:

train_df_bnk, test_df_bnk = train_test_split(df_bnk, test_size=0.2)


# In[139]:

func(train_df_bnk,test_df_bnk,"default")


# # Interpretation

# For interpretation, the cost is considered as an important factor than accuracy score because the cost matrix for our dataset has a huge difference between the two types of errors which can occur during prediction. The cost multiplied to the error is giving relatively high and low importance to both the errors accordingly, which is not the case with the accuracy score. If a dataset has equal weightage for both the errors, then accuracy score will be considered equally important as cost.

# So our analysis shows that the cost of predicted error is lowest for Decision tree model, taking the set of variables, considered important by random forest importance. Namely, 'cc_age','credit_usage','over_draft','credit_history','purpose','employment'.
# Therefore it is recommended that the bank manager should decide which new loan applicant to accept or to reject by taking into consideration the predictions based on decision tree model with the above mentioned attributes.
