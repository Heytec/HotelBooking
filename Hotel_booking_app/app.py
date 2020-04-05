import streamlit as st
import pandas as pd
import numpy as np

def impute_with_median(df):
    columns = list(df.columns)

    for column in columns:
        col_data = df[column]
        missing_data = sum(col_data.isna())

        if missing_data > 0:
            col_median = col_data.median()
            col_data.fillna(col_median, inplace = True)
            df[column] = col_data

    return df

from sklearn.preprocessing import LabelEncoder

def dummyEncode(df):
    columnsToEncode = list(df.select_dtypes(include = ['category', 'object']))
    le = LabelEncoder()

    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])

        except:
            print(f'Error encoding: {feature}')

    return df

def main():

    st.title("Hotel Booked")
    st.subheader("Predict whether or not a customer will cancel a room")

    menu = ['Exploritory Data Anlysis', 'Machine Learning']
    choices = st.sidebar.selectbox("Select", menu)

    data = pd.read_csv("hotel_bookings.csv")

    if choices == 'Exploritory Data Anlysis':
        st.subheader("Exploritory Data Anlysis")

        if data is not None:
            df = data.copy()
            st.dataframe(df.head())

            if st.checkbox("Show Shape:"):
                st.write(df.shape)

            if st.checkbox("Show Columns:"):
                all_columns = df.columns.to_list()
                st.write(df.columns)

            if st.checkbox("Show Summary:"):
                st.write(df.describe())

            if st.checkbox("Show Report:"):
                st.write(data.profile_report())

    elif choices == 'Machine Learning':
        st.subheader('Machine Learning')
        df = data.drop(['company', 'country', 'reservation_status_date'], axis = 1)

        st.subheader('check if null value ')
        df_imputed = impute_with_median(df)
        st.write(df_imputed.isnull().sum())

        dataset_encoded = dummyEncode(df_imputed)
        st.write(dataset_encoded.info())

        X = dataset_encoded.drop(columns = 'is_canceled')
        y = dataset_encoded['is_canceled']

        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        st.subheader('X_train')
        st.dataframe(pd.DataFrame(X_train).head())

        st.subheader('y_train')
        st.dataframe(pd.DataFrame(y_train).head())

        clf = LogisticRegression(random_state = 0, solver = 'lbfgs')
        clf.fit(X_train, y_train)

        st.subheader('Logistic regression model accuracy:')
        st.write(round(clf.score(X_test, y_test), 3))

        y_pred = clf.predict(X_test)

        st.subheader('Precision_score:')
        st.write(round(precision_score(y_test, y_pred), 3))

        st.subheader('Recall:')
        st.write(round(recall_score(y_test, y_pred), 3))

        st.subheader('F1 Score:')
        st.write(round(f1_score(y_test, y_pred), 3))

        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion matrix:', cm)

        st.subheader('Receiver Operating Characteristic Accuracy Curve:')
        st.write(round(roc_auc_score(y_test, y_pred), 3))

if __name__ == '__main__':
    main()
