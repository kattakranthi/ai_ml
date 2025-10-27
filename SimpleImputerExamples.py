What is SimpleImputer?
SimpleImputer is used to fill missing values (NaN) in your dataset with a default strategy like:
mean
median
mode
constan

Fill Missing Values with Mean
import numpy as np
from sklearn.impute import SimpleImputer

data = [
[1,2,np.nan],
[3,np.nan,6],
[7,8,9]
]

print("Original Data:" , np.array(data))

output:
[
[1.,2.,np.nan],
[3.,np.nan,6.],
[7.,8.,9.]
]

Apply SimpleImputer:

imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(data)

print("Imputed data", imputed_data)

Imputed Data:
 [[1.  2.  7.5]
  [3.  5.  6. ]
  [7.  8.  9. ]]


#imputer = SimpleImputer(strategy='mean')
#Fill Missing values with Median

#imputer = SimpleImputer(strategy='most_frequent')
#Fill Missing values with most frequent value 

#imputer = SimpleImputer(strategy='constant', fill_value=0)
#Fill Missing Values with a constant
