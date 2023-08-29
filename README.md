# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
## Step 1:
Importing the libraries
## Step 2:
Importing the dataset
## Step 3:
Taking care of missing data
## Step 4:
Encoding categorical data
## Step 5:
Normalizing the data
## Step 5:
Splitting the data into test and train

## PROGRAM:
## DEVELOPED BY: R.BRINDHA
## REG NO: 212222230023
```
import pandas as pd

df=pd.read_csv("/content/Churn_Modelling.csv")

df.head()

df.isnull().sum()

df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)

print(df)

x=df.iloc[:,:-1].values

y=df.iloc[:,-1].values

print(x)

print(y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df1 = pd.DataFrame(scaler.fit_transform(df))

print(df1)

from sklearn.model_selection import train_test_split

xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

print(xtrain)

print(len(xtrain))

print(xtest)

print(len(xtest))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df1 = sc.fit_transform(df)

print(df1)
```

## OUTPUT:
## df.head():
![a1](https://github.com/Brindha77/Ex.No.1---Data-Preprocessing/assets/118889143/09ba811d-7e72-462c-beb3-a3e8af67e42f)
## df.isnull().sum():
![a2](https://github.com/Brindha77/Ex.No.1---Data-Preprocessing/assets/118889143/a557de62-d8de-4f32-9dbb-722d711e3aee)
## df value:
![a3](https://github.com/Brindha77/Ex.No.1---Data-Preprocessing/assets/118889143/0185a5fa-44f2-4f67-9dd3-a1460a49847b)
## VALUES OF INPUT AND OUTPUT DATA ON VAR X AND Y:
![a4](https://github.com/Brindha77/Ex.No.1---Data-Preprocessing/assets/118889143/2c6370aa-209c-4861-8ac6-f6bf5cb41d71)
## NORMALIZING DATA:
![a5](https://github.com/Brindha77/Ex.No.1---Data-Preprocessing/assets/118889143/85a347e2-3be1-48f1-a8a2-7ae038a8ae03)
## X_TRAIN AND Y_TRAIN VALUES:
![a6](https://github.com/Brindha77/Ex.No.1---Data-Preprocessing/assets/118889143/495ddfdc-1818-4c6a-8acc-5699464b24d8)
## X AND Y VALUES :
![a7](https://github.com/Brindha77/Ex.No.1---Data-Preprocessing/assets/118889143/8712142d-e7e6-4b17-a132-6fcb27dc72cb)
## X_TEST AND Y_TEST VALUES :
![a8](https://github.com/Brindha77/Ex.No.1---Data-Preprocessing/assets/118889143/1c0d5cc5-91c4-4c81-a510-28af2d23aeb0)

## RESULT
Thus,the program to perform Data preprocessing in a data set downloaded from Kaggle is implemented successfully   
