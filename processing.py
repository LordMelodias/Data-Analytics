import pandas as pd
import numpy as np
import matplotlib.pyplot as mpt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ds = pd.read_csv('purchase.csv')
# Extract independent variable
x = ds.iloc[:,:-1].values
# Extract Dependent variable
y = ds.iloc[:,3].values

# handling missing value
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fitting imputer object to the independent variable x
imputer.fit(x[:,1:3]) 

# Replacing missing data with the calculated mean value
x[:, 1:3]= imputer.transform(x[:,1:3])
print(x)

''' our dataset would have a categorical variable, then it may create trouble while building the model. 
    So it is necessary to encode these categorical variables into numbers.'''
# Encode Categorical data
label_encoder_x = LabelEncoder()
x[:,0] = label_encoder_x.fit_transform(x[:,0])
# the country categorical has changed in number

''' our case, there are three country variables, and 
as we can see in the above output,these variables are encoded into 0, 1, and 2. 
By these values, the machine learning model may assume that there is some correlation 
between these variables which will produce the wrong output. So to remove this issue, 
we will use dummy encoding '''
# Encoding Dummy variables 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Encoding Dummy variable for y
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# Splitting dataset into training and testset
''' X_train: features for the training data
    X_test: features for testing data
    y_train: Dependent variables for training data
    y_test: Independent variable for testing data'''
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# In feature scaling, we put our variables in the same range and in the same scale so that no any variable dominate the other variable.
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

'''
Here, we have not scaled the dependent variable because there are only two values 0 and 1. 
But if these variables will have more range of values, then we will also need to scale those variables.
'''

