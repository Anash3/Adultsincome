#importing modules
from operator import length_hint
import pandas as pd
import numpy as np
import streamlit as st

#reading dataset
df=pd.read_csv("adult.csv")

#droping column
df.drop('fnlwgt',axis=1,inplace=True)

#dealing with country column
mask=df.country.map(df.country.value_counts()) < 100
df.country =  df.country.mask(mask, 'other')

#encodin the category variable
df['salary']=df['salary'].map({' <=50K':0, ' >50K':1})
df['sex']=df['sex'].map({' Male':0,' Female':1})

#one hot encoding the category variable
data=pd.get_dummies(df,columns=['workclass','marital-status','occupation','relationship','race','country'],drop_first=True)
data=pd.get_dummies(data,columns=['education'],drop_first=True)

#saperating dependent and independent variables
x=data.drop('salary',axis=1)
y=data['salary']

#spliting dataset for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Scaling the dataset
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

#train the model
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
model=reg.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


st.title("Adult Census Income Prediction")
with st.form("my_form"):
   age= st.number_input("Age")
   workclass=st.selectbox('workclass',[' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov',
       ' Local-gov', ' Self-emp-inc', ' Without-pay',
       ' Never-worked'])
   education=st.selectbox('education',[' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
       ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
       ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th',
       ' Preschool', ' 12th'])
   educationnumber= st.number_input("Education number")
   marital_status=st.selectbox('Marital Status',[' Never-married', ' Married-civ-spouse', ' Divorced',
       ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
       ' Widowed'])
   occupation=st.selectbox('Occupation',[' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',
       ' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair',
       ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',
       ' Tech-support', ' ?', ' Protective-serv', ' Armed-Forces',
       ' Priv-house-serv'])
   relationship=st.selectbox('Relationship',df['relationship'].unique())
   race=st.selectbox('Race',df['race'].unique())
   Gender=st.selectbox('Gender',['Male','Female'])
   capital_Gain=st.number_input("Capital Gain")
   capital_loss=st.number_input("Capital Loss")
   Hours_per_week=st.number_input("Hours_per_week")
   Country=st.selectbox('Country',df['country'].unique())
   

   
   submitted = st.form_submit_button("Submit")
    


if Gender==' Male':
    Gender=0
else:
    Gender=1


if workclass==' Federal-gov':
    Federalgov=1
    Localgov=0
    Neverworked=0
    Private=0
    Selfempinc=0
    Selfempnotinc=0
    Stategov=0
    Withoutpay=0
elif workclass==' Local-gov':
    Federalgov=0
    Localgov=1
    Neverworked=0
    Private=0
    Selfempinc=0
    Selfempnotinc=0
    Stategov=0
    Withoutpay=0
elif workclass==' Never-worked':
    Federalgov=0
    Localgov=0
    Neverworked=1
    Private=0
    Selfempinc=0
    Selfempnotinc=0
    Stategov=0
    Withoutpay=0
elif workclass==' Private':
    Federalgov=0
    Localgov=0
    Neverworked=0
    Private=1
    Selfempinc=0
    Selfempnotinc=0
    Stategov=0
    Withoutpay=0
elif workclass==' Self-emp-inc':
    Federalgov=0
    Localgov=0
    Neverworked=0
    Private=0
    Selfempinc=1
    Selfempnotinc=0
    Stategov=0 
    Withoutpay=0
elif workclass==' Self-emp-not-inc':
    Federalgov=0
    Localgov=0
    Neverworked=0
    Private=0
    Selfempinc=0
    Selfempnotinc=1
    Stategov=0
    Withoutpay=0
elif workclass==' State-gov':
    Federalgov=0
    Localgov=0
    Neverworked=0
    Private=0
    Selfempinc=0
    Selfempnotinc=0
    Stategov=1
    Withoutpay=0
elif workclass==' Without-pay':
    Federalgov=0
    Localgov=0
    Neverworked=0
    Private=0
    Selfempinc=0
    Selfempnotinc=0
    Stategov=0
    Withoutpay=1  
else:
    Federalgov=0
    Localgov=0
    Neverworked=0
    Private=0
    Selfempinc=0
    Selfempnotinc=0
    Stategov=0
    Withoutpay=0


if education==' 11th':
     eleven=1
     tweleve=0
     fourth=0
     sixth=0 
     eightth=0
     ningth=0
     Assocacdm=0 
     Assocvoc=0 
     Bachelors=0
     Doctorate=0 
     HSgrad=0 
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' 12th':
     eleven=0
     tweleve=1
     fourth=0
     sixth=0
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=0 
     Bachelors=0
     Doctorate=0  
     HSgrad=0 
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' 1st-4th':
     eleven=0
     tweleve=0 
     fourth=1
     sixth=0
     eightth=0
     ningth=0
     Assocacdm=0
     Assocvoc=0
     Bachelors=0
     Doctorate=0  
     HSgrad=0 
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' 5th-6th':
     eleven=0
     tweleve=0 
     fourth=0
     sixth=1
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=0
     Bachelors=0
     Doctorate=0  
     HSgrad=0 
     Masters=0
     Preschool=0
     Profschool=0
     Somecollege=0

elif education==' 7th-8th':
     eleven=0
     tweleve=0 
     fourth=0
     sixth=0 
     eightth=1
     ningth=0
     Assocacdm=0 
     Assocvoc=0 
     Bachelors=0
     Doctorate=0 
     HSgrad=0 
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' 9th':
     eleven=0
     tweleve=0 
     fourth=0
     sixth=0
     eightth=0
     ningth=1
     Assocacdm=0
     Assocvoc=0 
     Bachelors=0
     Doctorate=0  
     HSgrad=0
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' 1st-4th':
     eleven=0
     tweleve=0
     fourth=1
     sixth=0 
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=0 
     Bachelors=0
     Doctorate=0 
     HSgrad=0 
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' Assoc-acdm':
     eleven=0
     tweleve=0 
     fourth=0
     sixth=0
     eightth=0 
     ningth=0
     Assocacdm=1 
     Assocvoc=0 
     Bachelors=0
     Doctorate=0  
     HSgrad=0
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' Assoc-voc':
     eleven=0
     tweleve=0 
     fourth=0
     sixth=0 
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=1 
     Bachelors=0
     Doctorate=0  
     HSgrad=0 
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' Bachelors':
     eleven=0
     tweleve=0
     fourth=0
     sixth=0
     eightth=0
     ningth=0
     Assocacdm=0
     Assocvoc=0
     Bachelors=1
     Doctorate=0 
     HSgrad=0
     Masters=0
     Preschool=0 
     Profschool=0
     Somecollege=0

elif education==' Doctorate':
     eleven=0
     tweleve=0
     fourth=0
     sixth=0
     eightth=0
     ningth=0
     Assocacdm=0
     Assocvoc=0
     Bachelors=0
     Doctorate=1
     HSgrad=0
     Masters=0
     Preschool=0
     Profschool=0
     Somecollege=0

elif education==' HS-grad':
     eleven=0
     tweleve=0 
     fourth=1
     sixth=0
     eightth=0
     ningth=0
     Assocacdm=0
     Assocvoc=0
     Bachelors=0
     Doctorate=0  
     HSgrad=1
     Masters=0
     Preschool=0
     Profschool=0
     Somecollege=0

elif education==' Masters':
     eleven=0
     tweleve=0 
     fourth=0
     sixth=0 
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=0 
     Bachelors=0
     Doctorate=0  
     HSgrad=0 
     Masters=1
     Preschool=0
     Profschool=0
     Somecollege=0

elif education==' Preschool':
     eleven=0
     tweleve=0 
     fourth=1
     sixth=0
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=0 
     Bachelors=0
     Doctorate=0  
     HSgrad=0 
     Masters=0
     Preschool=1 
     Profschool=0
     Somecollege=0

elif education==' Prof-school':
     eleven=0
     tweleve=0 
     fourth=1
     sixth=0 
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=0 
     Bachelors=0
     Doctorate=0  
     HSgrad=0
     Masters=0
     Preschool=0
     Profschool=1
     Somecollege=0

elif education==' Prof-school':
     eleven=0
     tweleve=0
     fourth=1
     sixth=0
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=0
     Bachelors=0
     Doctorate=0 
     HSgrad=0
     Masters=0
     Preschool=0
     Profschool=0
     Somecollege=1

else:
     eleven=0
     tweleve=0
     fourth=1
     sixth=0
     eightth=0 
     ningth=0
     Assocacdm=0 
     Assocvoc=0
     Bachelors=0
     Doctorate=0 
     HSgrad=0
     Masters=0
     Preschool=0
     Profschool=0
     Somecollege=0


if occupation==' Adm-clerical':
       Admclerical=1
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Armed-Forces':
       Admclerical=0
       ArmedForces=1
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0 
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Craft-repair':
       Admclerical=0
       ArmedForces=0
       Craftrepair=1
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Exec-managerial':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=1
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Farming-fishing':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0 
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Handlers-cleaners':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=1
       Machineinspct=0
       Otherservice=0
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Machine-op-inspct':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=1
       Otherservice=0 
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Other-service':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Priv-house-serv':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0 
       Privhouseserv=1
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Prof-specialty':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0
       Privhouseserv=0
       Profspecialty=1
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Protective-serv':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0 
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=1
       Sales=0
       Techsupport=0
       Transportmoving=0

elif occupation==' Sales':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0 
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=1
       Techsupport=0
       Transportmoving=0

elif occupation==' Tech-support':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0 
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=1
       Transportmoving=0

elif occupation==' Transport-moving':
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=1
else:
       Admclerical=0
       ArmedForces=0
       Craftrepair=0
       Execmanagerial=0
       Farmingfishing=0
       Handlerscleaners=0
       Machineinspct=0
       Otherservice=0
       Privhouseserv=0
       Profspecialty=0
       Protectiveserv=0
       Sales=0
       Techsupport=0
       Transportmoving=0

if Country==' United-States':
    unitedstates=1
    Mexico  =0           
    Philippines=0        
    Germany     =0      
    Canada      =0       
    PuertoRico  =0      
    ElSalvador  =0     
    India       =0
    other       =0      


 
elif Country==' Mexico':
    unitedstates=0
    Mexico  =1           
    Philippines=0        
    Germany     =0       
    Canada      =0      
    PuertoRico  =0      
    ElSalvador  =0     
    India       =0
    other       =0     

elif Country==' India':
    unitedstates=0
    Mexico  =0         
    Philippines=0       
    Germany     =0       
    Canada      =0      
    PuertoRico  =0     
    ElSalvador  =0      
    India       =1
    other       =0   

elif Country==' Puerto-Rico':
    unitedstates=0
    Mexico  =0           
    Philippines=0       
    Germany     =0       
    Canada      =0       
    PuertoRico  =1      
    ElSalvador  =0     
    India       =0
    other       =0   

elif Country==' Canada':
    unitedstates=0
    Mexico  =1           
    Philippines=0       
    Germany     =0       
    Canada      =1       
    PuertoRico  =0      
    ElSalvador  =0      
    India       =0
    other       =0   

elif Country==' Germany':
    unitedstates=0
    Mexico  =0          
    Philippines=0        
    Germany     =1       
    Canada      =0       
    PuertoRico  =0     
    ElSalvador  =0     
    India       =0
    other       =0   

elif Country==' El-Salvador':
    unitedstates=0
    Mexico  =0         
    Philippines=0        
    Germany     =0       
    Canada      =0       
    PuertoRico  =0     
    ElSalvador  =1     
    India       =0
    other       =0   

elif Country==' Philippines':
    unitedstates=0
    Mexico  =0          
    Philippines=1        
    Germany     =0      
    Canada      =0      
    PuertoRico  =0     
    ElSalvador  =0      
    India       =0
    other       =0   

elif Country==' other':
    unitedstates=0
    Mexico  =0          
    Philippines=1        
    Germany     =0       
    Canada      =0      
    PuertoRico  =0     
    ElSalvador  =0     
    India       =0
    other       =1  

else:
    unitedstates=0
    Mexico  =0          
    Philippines=1        
    Germany     =0       
    Canada      =0      
    PuertoRico  =0     
    ElSalvador  =0     
    India       =0
    other       =0  
    


if relationship==' Not-in-family':
    Notinfamily=1
    Otherrelative=0
    Ownchild=0
    Unmarried=0
    Wife=0

elif relationship==' Other-relative':
    Notinfamily=0
    Otherrelative=1
    Ownchild=0
    Unmarried=0
    Wife=0    

elif relationship==' Unmarried':
    Notinfamily=0
    Otherrelative=0
    Ownchild=0
    Unmarried=1
    Wife=0    

elif relationship==' Own-child':
    Notinfamily=0
    Otherrelative=0
    Ownchild=1
    Unmarried=0
    Wife=0    

elif relationship==' Wife':
    Notinfamily=0
    Otherrelative=0
    Ownchild=1
    Unmarried=0
    Wife=1    

else:
    Notinfamily=0
    Otherrelative=0
    Ownchild=1
    Unmarried=0
    Wife=0   

if marital_status==' Married-AF-spouse':
      Marriedafspouse=1
      Marriedspouse=0
      Marriedspousebsent=0
      Nevermarried=0
      Separated=0
      Widowed=0

elif marital_status==' Married-civ-spouse':
      Marriedafspouse=0
      Marriedspouse=1
      Marriedspousebsent=0
      Nevermarried=0 
      Separated=0
      Widowed=0

elif marital_status==' Married-spouse-absent':
      Marriedafspouse=0
      Marriedspouse=0
      Marriedspousebsent=1
      Nevermarried=0
      Separated=0
      Widowed=0

elif marital_status==' Never-married':
      Marriedafspouse=0
      Marriedspouse=0
      Marriedspousebsent=0
      Nevermarried=1
      Separated=0
      Widowed=0

elif marital_status==' Separated':
      Marriedafspouse=0
      Marriedspouse=0
      Marriedspousebsent=0
      Nevermarried=0 
      Separated=1
      Widowed=0

elif marital_status==' Widowed':
      Marriedafspouse=0
      Marriedspouse=0
      Marriedspousebsent=0
      Nevermarried=0 
      Separated=0
      Widowed=1
else:
      Marriedafspouse=0
      Marriedspouse=0
      Marriedspousebsent=0
      Nevermarried=0 
      Separated=0
      Widowed=0

if race==' Asian-Pac-Islander':
     AsianPacIslander=1
     Black=0
     Other=0
     White=0

elif race==' Black':
     AsianPacIslander=0
     Black=1
     Other=0
     White=0

elif race==' Other':
     AsianPacIslander=0
     Black=0
     Other=1
     White=0

elif race==' White':
     AsianPacIslander=0
     Black=0
     Other=0
     White=1
else:
     AsianPacIslander=0
     Black=0
     Other=0
     White=0

       
result=[age,educationnumber,Gender,capital_Gain,capital_loss,Hours_per_week,
Federalgov,Localgov,Neverworked,Private,Selfempinc,Selfempnotinc,Stategov,Withoutpay,
Marriedafspouse,Marriedspouse,Marriedspousebsent,Nevermarried,Separated,Widowed,
Admclerical,ArmedForces,Craftrepair,Execmanagerial,Farmingfishing,Handlerscleaners,Machineinspct,Otherservice,Privhouseserv,Profspecialty,Protectiveserv,Sales,Techsupport,Transportmoving,
Notinfamily,Otherrelative,Ownchild,Unmarried,Wife,AsianPacIslander,White,Black,Other,unitedstates,Mexico,Philippines,Germany,Canada,PuertoRico,ElSalvador,India,other,
eleven,tweleve,fourth,sixth,eightth,ningth,Assocacdm,Assocvoc,Bachelors,Doctorate,HSgrad,Masters,Preschool,Profschool,Somecollege,]
final_input=scalar.transform(np.array(result).reshape(1,-1))
ans=model.predict(final_input)[0]


if submitted:
    if ans==0:
        st.write('Your income is less than or equal to 50k')
    else:
        st.write("Your income is greater than 50k")
