import numpy as np
import pandas as pd
import pickle
model = pickle.load(open("model.pkl","rb"))
island=' '
for n in range(1):
    a=input('''
 Enter the island name 
   1.  Torgersen 
   2.  Biscoe 
   3.  Dream
   :
''')
    if a=='1':
        island='Torgersen'
    elif a=='2':
        island='Biscoe'
    elif a=='3':
        island='Dream'
    else:
        island=''
bill_length_mm=abs(float(input("Enter the bill_length_mm: ")))
bill_depth_mm=abs(float(input("Enter the bill_depth_mm: ")))
flipper_length_mm=abs(float(input("Enter the flipper_length_mm: ")))
body_mass_g=abs(float(input("Enter the body_mass_g: ")))
sex=' '
for n in range(1):
    a=input('''
 Enter the sex 
   1. Male 
   2. Female
''')
    if a=='1':
        sex='Male'
    elif a=='2':
        sex='Female'
    else:
        sex=''

df=pd.DataFrame([[island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]],
                columns=['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex'])

y_pred=model.predict(df)

print("predicted species",y_pred[0])