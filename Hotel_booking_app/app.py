#core pkgsstreamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import pandas_profiling as pp
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
# Evaluating Result
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score



# function to  Iterate through columns of Pandas DataFrame.Where NaNs exist replace with median
def impute_with_median(df):
    """Iterate through columns of Pandas DataFrame.
    Where NaNs exist replace with median"""

    # Get list of DataFrame column names
    cols = list(df)
    # Loop through columns
    for column in cols:
        # Transfer column to independent series
        col_data = df[column]
        # Look to see if there is any missing numerical data
        missing_data = sum(col_data.isna())
        if missing_data > 0:
            # Get median and replace missing numerical data with median
            col_median = col_data.median()
            col_data.fillna(col_median, inplace=True)
            df[column] = col_data
    return df


from sklearn.preprocessing import LabelEncoder


# from sklearn.preprocessing import OneHotEncoder

# Auto encodes any dataframe column of type category or object.
def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    le = LabelEncoder()
    # enc = OneHotEncoder(handle_unknown='ignore')
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
            # df[feature]=enc.fit_transform(df[feature])

        except:
            print('Error encoding ' + feature)
    return df





def main():
   """"Hotel booked """

   st.title("Hotel Booked ")
   st.subheader("Predict whether or not a customer will cancel a room")

   menu=['EDA','Machine Learning' ]
   choices=st.sidebar.selectbox("Select",menu)

   if choices=='EDA':
       st.subheader("EDA")
       #data = st.file_uploader("upload datset", type=['csv', 'txt'])
       data = pd.read_csv("hotel_bookings.csv")
       if data is not None:
           df = data
           st.dataframe(df.head())

           if st.checkbox("show shape"):
               st.write(df.shape)

           if st.checkbox("show colums"):
               all_columns = df.columns.to_list()
               st.write(df.columns)

           if st.checkbox("show Summary"):
               st.write(df.describe())








   if choices == 'Machine Learning':
       st.subheader('Machine Learning')
       # Remove two columns name is 'Country' and 'reservation',company
       data = pd.read_csv("hotel_bookings.csv")
       df = data.drop(['company', 'country', 'reservation_status_date'], axis=1)
       st.subheader('check if null value ')
       df_imputed = impute_with_median(df)
       st.write(df_imputed.isnull().sum())
       dataset_encoded = dummyEncode(df_imputed)
       st.write(dataset_encoded.info())
       X = dataset_encoded.drop(columns='is_canceled')
       y = dataset_encoded['is_canceled']

       # generate the datasets for training. Test 20%
       X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                           test_size=0.2,
                                                           random_state=0)

       from sklearn.preprocessing import StandardScaler
       # generate an standarScalar
       sc_X = StandardScaler()

       # StandardScaler return a Numpy Array so we need to convert to a Dataframe
       X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
       X_test2 = pd.DataFrame(sc_X.fit_transform(X_test))

       # copy columns name to the new traning and testing dataset
       X_train2.columns = X_train.columns.values
       X_test2.columns = X_test.columns.values

       # copy index to the new traning and testing dataset
       X_train2.index = X_train.index.values
       X_test2.index = X_test.index.values

       # reasigned copy dataset to original
       X_train = X_train2
       X_test = X_test2

       st.subheader('Xtrain')
       st.dataframe(X_train.head())
       st.subheader('Xtest')
       st.dataframe(X_test.head())
       st.subheader('Y_test')
       st.dataframe(y_test.head())

       classifier = LogisticRegression(random_state=0, solver='lbfgs')
       classifier.fit(X_train, y_train)

       st.subheader('Logistic regression model accuracy ')
       st.write(classifier.score(X_test, y_test))

       #clf = SVC(gamma='auto')
       #clf.fit(X_train, y_train)
       #st.subheader('SVM accuracy ')
       #st.write(clf.score(X_test, y_test))

       # Evaluating Test set
       y_pred = classifier.predict(X_test)

       st.subheader('Precision_score ')
       pre=precision_score(y_test, y_pred)
       rec=recall_score(y_test,y_pred)

       st.write(precision_score(y_test, y_pred))

       st.subheader('Recall')
       st.write(recall_score(y_test, y_pred))

       st.subheader('F1 Score ')
       st.write(f1_score(y_test, y_pred))

       cm = confusion_matrix(y_test, y_pred)
       st.write('Confusion matrix: ', cm)



       st.subheader('ROC_AUC CURVE ')
       st.write( roc_auc_score(y_test, y_pred))







if __name__ == '__main__':
    main()
