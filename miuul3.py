import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")
df.head()
df.tail
df.shape
df.columns
df.info
df.index
df.describe()
df.isnull().values.any()
df.isnull().sum

def check_df(dataframe, head=5):
    print("#shape")
    print(dataframe.shape)
    print("#types")
    print(dataframe.dtypes)
    print("#head")
    print(dataframe.head)
    print("#tail")
    print(dataframe.tail)
    print("# na")
    print(dataframe.isnull().sum())
    print("#quantiles")
    print(dataframe.describe([0,0.05,0.50,0.95,0.99,1]).T)

check_df(df)

df = sns.load_dataset("tips")
check_df(df)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")

df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()

df["sex"].dtypes
str(df["sex"].dtypes)
str(df["sex"].dtypes) in ["object"]
str(df["alone"].dtypes) in ["bool"]

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category","bool","object"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64","float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

[col for col in df.columns if col not in cat_cols]

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100* dataframe[col_name].value_counts() / len(dataframe)}))

cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df,col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100* dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df,"sex",plot=True)
cat_summary(df,"who",plot=True)
cat_summary(df,"pclass",plot=True)

for col in cat_cols:
    cat_summary(df,col, plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("boolean!!!!!!!!!!!!!!")
    else:
        cat_summary(df,col, plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df["adult_male"].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df,col, plot=True)


##hard

def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col] = dataframe[col].astype(int)
        print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100* dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df,"adult_male",plot=True)

###########################3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")
df.head()
df[["age","fare"]].describe().T

[col for col in df.columns if df[col].dtypes in ["int64","float64"]]

for col in df.columns:
    print(df[col].dtypes)

num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10,0.20,0.30,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
    print(df[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df,"age", plot=True)

for col in num_cols:
    num_summary(df,col,plot=True)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df = sns.load_dataset("titanic")
df.head()
df.info

#general useful function

# docstring

def grab_col_names(dataframe, categoric_threshold = 10, cardinal_threshold=30):
    """
    categorical and
     numerical or categorical but cardinal variables from the data set
    Parameters
    ----------
    dataframe : dataframe

    categoric_th:int,float
        numerical but cardinal variables threshold

    cardinal_th: int , float
        categorical but cardinal variables threshold
    Returns
    -------
    cat_cols: list
        list of categorical variables
    num_cols : list
        list of numerical variables
    cat_but_car: list
        list of categoric lookking cardinal variables
    Notes
    -------
    cat_cols + num_cols + cat_but_car = summation of all variables
    cat_col spans num_but_cat

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "bool", "object"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"observations : {dataframe.shape[0]}")
    print(f"variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")

    return cat_cols,num_cols,cat_but_car

grab_col_names(df)
help(grab_col_names)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100* dataframe[col_name].value_counts() / len(dataframe)}))

for col in cat_cols:
    cat_summary(df,col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10,0.20,0.30,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
    print(df[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col,plot=True)

#bonus
df= sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols,num_cols,cat_but_car

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                        "Ratio":100* dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col, plot=True)

for col in num_cols:
    num_summary(df,col, plot=True)

#analysis of target variable
##survived

df.head()
df["survived"].value_counts()
cat_summary(df,"survived")

#categorical variable
df.groupby("sex")["survived"].mean()
df.groupby("pclass")["survived"].mean()

def target_summary_with_cat(dataframe,target,categorical_col):
    print(pd.DataFrame({"target_mean" : dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df,"survived", "sex")

for col in cat_cols:
    target_summary_with_cat(df,"survived",col)

#numerical

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n")

target_summary_with_num(df,"survived",col)

for col in num_cols:
    target_summary_with_num(df,"survived",col)






















