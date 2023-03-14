
import pandas as pd
s = pd.Series([10,77,12,4,5])
s
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head()
s.tail()

df = pd.read_csv(r'C:\Users\USER\Desktop\advertising.csv')
print(df)
df.head()

#quick look at the data
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info
df.columns
df.index
df.describe().T #transpose of
df.isnull().values
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()
df[0:13]
df.drop(0,axis=0).head()
delete_indexes = [1,3,5,7]
df.drop(delete_indexes,axis=0).head(10)
#two way to make this general
# df = df.drop(delete_indexes,axis=0).head(10)
# df.drop(delete_indexes,axis=0 , inplace = True)

df["age"].head()
df.age.head()

df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age", axis=1 , inplace=True)
df.index
# changing index to variable
#way1
df["age"] = df.index
df.head()
#way 2
df.drop("age", axis=1 , inplace=True)
df.head()

df.reset_index().head()
df = df.reset_index().head()
df.head()

#operations on variable
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic")
df.head()

"age" in df
df["age"].head()
df.age.head()
type(df["age"].head())
df[["age"]].head()
type(df[["age"]].head())

df[["age","alive"]]
col_names = ["age","adult_male","alive"]
df[col_names]
df["age2"] = df["age"]**2
df
df["age3"] = df["age"]/df["age2"]
df

df.drop("age3",axis=1).head()
df.drop(col_names,axis=1).head()

df.loc[:,df.columns.str.contains("age")].head()
df.loc[:,~df.columns.str.contains("age")].head() # ~~ others

# iloc integer based selection
#  loc label based selection
df.iloc[0:3]
df.iloc[0,0]
df.loc[0:3]

# from 0 to 3 by rows
df.loc[0:3,"age"]
col_names = ["age","embarked","alive"]
df.loc[0:3,col_names]

# age > 50

df[df["age"]>50].head()
df[df["age"]>50]["age"].count()
df.loc[df["age"]>50 , "class"].head()
df.loc[df["age"]>50 , ["age","class"]].head()

df.loc[(df["age"]>50) & (df["sex"]=="male"), ["age","class"]].head()

df.loc[(df["age"]>50)
       &(df["sex"]=="male")
       & (df["embark_town"]=="Cherbourg") ,
       ["age","class","embark_town"]].head()

df["embark_town"].value_counts()

df.new = df.loc[(df["age"]>50) & (df["sex"]=="male")
       & ((df["embark_town"]== "Cherbourg") | (df["embark_town"]=="Southampton")) ,
       ["age","class","embark_town"]]

df.new["embark_town"].value_counts()

#aggregation and grouping
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age":"mean"})
df.groupby("sex").agg({"age":["sum","mean"]})
df.groupby("sex").agg({"age":["sum","mean"],
                       "embark_town":"count"})
df.groupby("sex").agg({"age":["sum","mean"],
                       "survived":"mean"})
df.groupby(["sex","embark_town"]).agg({"age":["mean"],
                       "survived":"mean"})
df.groupby(["sex","embark_town","class"]).agg({"age":["mean"],
                       "survived":"mean"})
df.groupby(["sex","embark_town","class"]).agg({
    "age":["mean"],
    "survived":"mean",
    "sex":"count"})

#3pivot table
#df.pivot_table(intended variable,row variable,column variable)
df.pivot_table("survived","sex","embarked")
df.pivot_table("survived","sex","embarked", aggfunc="std")
df.pivot_table("survived","sex",["embarked","class" ])

#.cut(data, from what to what you want to group by)
df["new_age"] = pd.cut(df["age"],[0,10,18,25,40,90])
df.pivot_table("survived","sex","new_age")
df.pivot_table("survived","sex",["new_age","class"])
pd.set_option("display.width",500)

## apply in a row or column makes a func work
# lambda like def define a code for a single time
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns',None)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10
df.head()

##
df[["age","age2","age3"]].head()
df[["age","age2","age3"]].apply(lambda x : x**2)
df[["age","age2","age3"]].apply(lambda x : x**2).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x : x/10).head()
#from all rows
# from all columns that contains "age"

# normalization formula average/standard deviation
df.loc[:, df.columns.str.contains("age")].apply(lambda x : (x-x.mean())/ x.std()).head()

def standartet(col_name):
    return (col_name - col_name.mean())/ col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standartet).head()

df.loc[:,["age","age2","age3"]]= df.loc[:, df.columns.str.contains("age")].apply(standartet).head()

df.loc[:,df.columns.str.contains("age")]= df.loc[:, df.columns.str.contains("age")].apply(standartet).head()

df.head()

# join
import pandas as pd
import numpy as np
m = np.random.randint(1,30,size=(5,3))
df1 = pd.DataFrame(m,columns=["var1","var2","var3"])
df2 = df1 + 99

pd.concat([df1,df2])
pd.concat([df1,df2], ignore_index=True)

# merge
df2 = pd.DataFrame({"employees":["mark","john","dennis","maria"],
                   "start_date":[2010,2009,2014,2019]})
df1 = pd.DataFrame({"employees":["mark","john","dennis","maria"],
                   "group":["accounting","engineering","engineering","hr"]})

pd.merge(df1,df2)
pd.merge(df1,df2, on="employees")

#manager information
df3 = pd.merge(df1,df2)
df4= pd.DataFrame({"group":["accounting","engineering","hr"],
                   "manager":["caner","mustafa","berkcan"]})

pd.merge(df3,df4)

