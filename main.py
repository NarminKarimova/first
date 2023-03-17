import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import numpy as np
from sklearn.impute import SimpleImputer
import os


df=pd.read_csv('loan_pred.csv')
df1=pd.read_csv('water_potability (1).csv')

icon=Image.open('DS.jfif')
logo=Image.open('download.jfif')
photo=Image.open('photo.jpg')

st.set_page_config(layout="wide",page_title="Python",page_icon=icon)
st.title ("Python Week 8 - Deployment")
st.text("Case Study - Machine Learning Web Application with Streamlit")

#sidebar
st.sidebar. image (image=logo)
menu=st. sidebar.selectbox ("", ["Homepage", "EDA", "Modeling"])

if menu== 'Homepage':
    st.header( 'HOMEPAGE')
    st.image(photo,use_column_width="always")

    dataset=st.selectbox ('Secilen',["Loan Prediction", "Water Potability"])
    st.markdown("Secilen {0}". format (dataset) )

    if dataset=="Loan Prediction":
        st.warning('Load Prediction problems you can see there')
    else:
        st.warning('Water potability problems you can see there')

elif menu== 'EDA':
    def outlier_treatment(datacolumn):
        sorted(datacolumn)
        Q1,Q3 = np.percentile(datacolumn , [25,75])
        IQR=Q1 -Q3
        lower_range = Q1 - (1.5 * IQR)
        upper_range = Q3 + (1.5 * IQR) 
        return lower_range, upper_range

    def describeStat(df):
        st.dataframe(df)
        st.subheader ("Statistical Values")
        df.describe().T

        st.subheader("Balance of Data")
        st.bar_chart (df. iloc[:,-1].value_counts())

        null_df=df.isnull().sum() .to_frame().reset_index()
        null_df.colums= ['Columns', "Counts" ]
                         


        c_eda1,c_eda2,c_eda3=st.beta_columns([2.5,1.5,2.5])

        c_eda1.subheader('Null Variables')
        c_eda1.dataframe(null_df)

        c_eda2.subheader("Imputation")
        cat_methods=c_eda2.radio ( "Categorical ", [ 'Mode', 'Backfill', 'FFi11'])
        num_method=c_eda2.radio( "Numerical", [ "Mode", "Median", ])

        #feature enginering
        c_eda2.subheader('Feature Engineering')
        balance_problem = c_eda2.checkbox( "Under Sampling")
        outlier_problem = c_eda2.checkbox( "Clean Outlier")

        if c_eda2.button('Prep'):
            cat_array=df.iloc[:,:-1].select_dtypes(include="object").columns
            num_array=df.iloc[:,:-1].select_dtypes(include='number').columns
            
            
            if cat_array.size>0:
                if cat_methods=='Mode':
                    imp_cat= SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                    df[cat_array]=imp_cat.fit_transform(df[cat_array])
                elif cat_methods =='Backfill':
                    df[cat_array].fillna(methods="Backfill" ,inplace=True)
                else:
                    df[cat_array].fillna(methods="ffill" ,inplace=True)

            if num_array.size>0:
                if num_method=='Mode':
                    imp_num= SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                else:
                     imp_num= SimpleImputer(missing_values=np.nan, strategy='median')
                df[num_array]=imp_num.fit_transform(df[num_array])
            
            df.dropna(axis=0,inplace=True)



            if balance_problem:
                from imblearn.under_sampling import RandomUnderSampler
                rus=RandomUnderSampler
                X= df.iloc[:,:-1]
                Y= df.iloc[:,[1]]
                X, Y= rus.fit_resample(X,Y)
                df=pd.concat([X,Y],axis=1)

            
            if outlier_treatment():
                for col in num_array:
                    lowerbound,upperbound = outlier_treatment(df[col])
                    df[col] = np.clip(df[col],a_min=lowerbound,a_max=upperbound)
            

            null_df=df. isnuli().sum().to_frame().reset_index()
            null_df.columns= ["Columns",'Counts']
            c_eda3.subheader("Null Variables")
            c_eda3.dataframe(null_df)
            st.subheader("Balance of Data") 
            st.bar_chart(df.loc[:,-1].value_counts())


            heatmap = px. imshow(df.corr())
            st.plotly_chart(heatmap) 
            st.dataframe (df)

            if os.path.exists('formodel.csv'):
                os.remove('formodel.csv')
            df.to_csv('formodel.csv')

            
    st.header( "Exploratory Data Analysis")
    dataset=st.selectbox ('Select dataset',["Loan Prediction", "Water Potability"])

    if dataset=="Loan Prediction":
        df=pd.read_csv('loan_pred.csv')
        describeStat(df)
    else:
        df=pd.read_csv('water_potability (1).csv')
        describeStat(df)
else:
    st.header('Modellin')
    if os.path.exists('formodel.csv'):
        st.header('Please Run Preprossing Part')
    else:
        df=pd.read_csv('formodel.csv')
        st.dataframe(df)
    
    c_model1,c_model2=st.beta_columns(2)


    
    c_model2. subheader( "Encoders" )
    encoder_method=c_model2.radio('', ['Label',"One-Hot"])

    st.header("Train and Test Splitting")
    c_model_1,c_model_2=st.beta_columns(2)
    random_state=c_model_1.text_input("Randon state") 
    test_size=c_model_2.text_input ("Percentage")


    model=st.selectbox ('Secilen',["XGBoost", "CatBoost"])
    st.markdown("Secilen {0}". format (model) )


    if st.button("Run"):
        cat_array=df.iloc[:,:-1].select_dtypes(include="object").columns
        num_array=df.iloc[:,:-1].select_dtypes(include='number').columns
        Y=df.iloc[:,[-1]]


        if cat_array.size>0:
            if encoder_method=='Label':
                from sklearn.preprocessing import LabelEncoder
                lb=LabelEncoder()
                for col in cat_array:
                    df[col]=lb.fit_transform(df[col])
               
            else:
                df.drop(df.iloc[:,[-1]],axis=1,inplace=True)
                dms_df=df[cat_array]
                dms_df=pd.get_dummies(dms_df,drop_first=True)
                df_ =df.drop(cat_array,inplace=True)
                df=pd.concat([dms_,df_,df,Y],axis=1)
        st.dataframe(df)




        X= df.iloc[:,:-1]
        Y= df.iloc[:,[1]]

        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


        st.markdown('X_train size = {0}'.format(X_train.shape))
        st.markdown('X_test size = {0}'.format(X_test.shape))
        st.markdown('y_train size = {0}'.format(y_train.shape)) 
        st.markdown('y_test size = {0}'.format(y_test.shape))


        st.title('Your Model is working')

        if model=='XGBoost':
            import xgboost as xgb
            model=xgb .XGBClassifier().fit(X_train,Y_train)
        if model=='CatBoost':
            from catboost import CatBoostClassifier
            model=CatBoostClassifier().fit(x_train,y_train)

        y_pred=model.predict(X_test)

 



    







        


    












