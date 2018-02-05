
# coding: utf-8

# In[1005]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
glass_df = pd.read_csv(r"C:\Users\kartik singh\Desktop\glass classification\glass.csv")
glass_np = np.array(glass_df)
glass_list = list(glass_np)
glass_target=glass_df["Type"]
glass_data=glass_df.loc[:,"Ri":"Fe"]


# In[ ]:




# In[1006]:

import pylab as pl
for i in range(0, glass_np.shape[0]):
    c1 = pl.scatter(glass_np[i,0],glass_np[i,1],marker="o")
    
        

pl.show()


# In[ ]:




# In[1007]:

#from sklearn.decomposition import PCA
#pca = PCA(n_components = 3)
#pca.fit(glass_list)
#pca_2d = pca.transform(glass_list)


# In[ ]:




# In[1008]:

#import pylab as pl
#for i in range(0, pca_2d.shape[0]):
#    if glass_np[i][9] == 1:
#        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],marker='o')
#    elif glass_np[i][9] == 2:
#        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],marker='.')
#    elif glass_np[i][9] == 3:
#        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],marker='8')
#    
        

#pl.show()
#pca_2d.shape[0]


# In[1009]:

#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters=3, random_state=0).fit(glass_np)


# In[1010]:

#for i in range(0, pca_2d.shape[0]):
#    if kmeans.labels_[i] == 1:
#        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
#    elif kmeans.labels_[i] == 0:
#        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
#    elif kmeans.labels_[i] == 2:
#        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
#pl.show()


# In[1011]:

Type=[]
Ri = []
Na = []
Mg = []
Al = []
Si = []
K = []
Ca = []
Ba = []
Fe = []
Type_Ri=[]
Type_Na=[]
Type_Mg=[]
Type_Al=[]
Type_Si = []
Type_K = []
Type_Ca = []
Type_Ba = []
Type_Fe = []
for index,i in glass_df.iterrows():
    Ri.append(i["Ri"])
    Na.append(i["Na"])
    Mg.append(i["Mg"])
    Al.append(i["Al"])
    Si.append(i["Si"])
    K.append(i["K"])
    Ca.append(i["Ca"])
    Ba.append(i["Ba"])
    Fe.append(i["Fe"])
    Type_Ri.append(i["Type"])
    Type_Na.append(i["Type"])
    Type_Mg.append(i["Type"])
    Type_Al.append(i["Type"])
    Type_Si.append(i["Type"])
    Type_K.append(i["Type"])
    Type_Ca.append(i["Type"])
    Type_Ba.append(i["Type"])
    Type_Fe.append(i["Type"])
    Type.append(i["Type"])
    
len(Mg)


# In[ ]:




# In[1012]:

def Z_score(x,mean,std):
    Z_value=(x-mean)/std 
    return Z_value 
Z_score_Ri = []
for i in Ri:
    Z_score_Ri.append(Z_score(i,1.518365,0.003037))
Z_score_Na = []
for i in Na:
    Z_score_Na.append(Z_score(i,13.407850,0.816604))
Z_score_Al = []
for i in Al:
    Z_score_Al.append(Z_score(i,1.444907,0.499270))
Z_score_Si = []
for i in Si:
    Z_score_Si.append(Z_score(i,72.650935,0.774546))
Z_score_K = []
for i in K:
    Z_score_K.append(Z_score(i,0.497056,0.652192))
Z_score_Ca = []
for i in Ca:
    Z_score_Ca.append(Z_score(i,8.956963,1.423153))
Z_score_Ba = []
for i in Ba:
    Z_score_Ba.append(Z_score(i,0.175047,0.497219))
Z_score_Fe = []
for i in Fe:
    Z_score_Fe.append(Z_score(i,0.057009,0.097439))
Z_score_Mg = []
for i in Mg:
    Z_score_Mg.append(Z_score(i,2.684533,1.442408))
Z_score_Type = []
for i in Type:
    Z_score_Type.append(Z_score(i,2.780374,2.103739))


# In[1013]:

r= []
p=0
for i in range(0,len(Ri)):
    p=p+(Z_score_Ri[i]*Z_score_Type[i])
p=p/(len(Ri)-1) 
r.append(p)
p=0
for i in range(0,len(Na)):
    p=p+(Z_score_Na[i]*Z_score_Type[i])
p=p/(len(Ri)-1) 
r.append(p)
p=0
for i in range(0,len(Mg)):
    p=p+(Z_score_Mg[i]*Z_score_Type[i])
p=p/(len(Mg)-1) 
r.append(p)
p=0
for i in range(0,len(Al)):
    p=p+(Z_score_Al[i]*Z_score_Type[i])
p=p/(len(Al)-1) 
r.append(p)
p=0
for i in range(0,len(Si)):
    p=p+(Z_score_Si[i]*Z_score_Type[i])
p=p/(len(Si)-1) 
r.append(p)
p=0
for i in range(0,len(K)):
    p=p+(Z_score_K[i]*Z_score_Type[i])
p=p/(len(K)-1) 
r.append(p)
p=0
for i in range(0,len(Ca)):
    p=p+(Z_score_Ca[i]*Z_score_Type[i])
p=p/(len(Ca)-1) 
r.append(p)
p=0
for i in range(0,len(Ba)):
    p=p+(Z_score_Ba[i]*Z_score_Type[i])
p=p/(len(Ba)-1) 
r.append(p)
p=0
for i in range(0,len(Fe)):
    p=p+(Z_score_Fe[i]*Z_score_Type[i])
p=p/(len(Fe)-1) 
r.append(p)
r


# In[1014]:

plt.scatter(Mg,Type_Mg,color='Red', s=25, marker="o")
plt.xlabel('Mg')
plt.ylabel('Type_Mg')
plt.show()


# In[1015]:

from statistics import median,mean
Mg_1=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 1:
        Mg_1.append(i["Mg"])
median_Mg_1=median(Mg_1)
a=0
for i in Mg_1:
    if (i-median_Mg_1)<0:
        a=a-(i-median_Mg_1)
    else:
        a=a+(i-median_Mg_1)
mad_Mg_1=a/len(Mg_1) 
mean_Mg_1=mean(Mg_1)
print(median_Mg_1)
print(mad_Mg_1)
print(Mg_1)


# In[1016]:

for i in Mg_1:
    if i>(median_Mg_1 + 0.4487142857142858) or i<(median_Mg_1 - 0.4487142857142858):
        print(i)
Mg_ol = []
for i in Mg:
    Mg_ol.append(i)


# In[1017]:

Mg_ol.index(4.49)
Mg_ol[0]=3.565


# In[1018]:

print(Mg_ol.index(2.87))
print(Mg_ol.index(2.84))
print(Mg_ol.index(2.81))
print(Mg_ol.index(2.71))


# In[1019]:

Mg_ol[52]=3.565
Mg_ol[53]=3.565
Mg_ol[54]=3.565
Mg_ol[55]=3.565


# In[1020]:

from statistics import median,mean
Mg_2=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 2:
        Mg_2.append(i["Mg"])
median_Mg_2=median(Mg_2)
a=0
for i in Mg_2:
    if (i-median_Mg_2)<0:
        a=a-(i-median_Mg_2)
    else:
        a=a+(i-median_Mg_2)
mad_Mg_2=a/len(Mg_2) 
mean_Mg_2=mean(Mg_2)
print(median_Mg_2)
print(mad_Mg_2)
print(Mg_2)


# In[1021]:

for i in Mg_2:
    if i>(median_Mg_2 + 2.0328947368421061) or i<(median_Mg_2 - 2.0328947368421061):
        print(i)


# In[1022]:

Mg_ol.index(0)
Mg_ol[105]=3.52
Mg_ol[106]=3.52
Mg_ol[107]=3.52
Mg_ol[108]=3.52
Mg_ol[109]=3.52
Mg_ol[110]=3.52
Mg_ol[111]=3.52
Mg_ol[112]=3.52
Mg_ol[129]=3.52
Mg_ol[130]=3.52
Mg_ol[131]=3.52


# In[1023]:

from statistics import median,mean
Mg_5=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 5:
        Mg_5.append(i["Mg"])
median_Mg_5=median(Mg_5)
a=0
for i in Mg_5:
    if (i-median_Mg_5)<0:
        a=a-(i-median_Mg_5)
    else:
        a=a+(i-median_Mg_5)
mad_Mg_5=a/len(Mg_5) 
mean_Mg_5=mean(Mg_5)
print(median_Mg_5)
print(mad_Mg_5)
print(Mg_5)


# In[1024]:

for i in Mg_5:
    if i>(median_Mg_5 + 2.3215384615384617) or i<(median_Mg_5 - 2.3215384615384617):
        print(i)


# In[1025]:

Mg_ol.index(2.68)
Mg_ol[163]=0.0


# In[1026]:

from statistics import median,mean
Mg_7=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 7:
        Mg_7.append(i["Mg"])
median_Mg_7=median(Mg_7)
a=0
for i in Mg_7:
    if (i-median_Mg_7)<0:
        a=a-(i-median_Mg_7)
    else:
        a=a+(i-median_Mg_7)
mad_Mg_7=a/len(Mg_7) 
mean_Mg_7=mean(Mg_7)
print(median_Mg_7)
print(mad_Mg_7)
print(Mg_7)


# In[1027]:

for i in Mg_7:
    if i>(median_Mg_7 + 1.6148275862068965) or i<(median_Mg_7 - 1.6148275862068965):
        print(i)


# In[1028]:

print(Mg_ol.index(3.2))
print(Mg_ol.index(3.26))
print(Mg_ol.index(3.34))
print(Mg_ol.index(2.2))
print(Mg_ol.index(1.83))
print(Mg_ol.index(1.78))


# In[1029]:

Mg_ol[185]=0.0
Mg_ol[186]=0.0
Mg_ol[188]=0.0
Mg_ol[189]=0.0
Mg_ol[190]=0.0
Mg_ol[187]=0.0


# In[ ]:




# In[ ]:




# In[1030]:

plt.scatter(Type_Mg_ol,Mg_ol, label = "red" ,color='Red', s=25, marker="o")
plt.xlabel('Mg_ol')
plt.ylabel('Type_Mg_ol')
plt.legend()
plt.show()
len(Mg_ol)


# In[1031]:

Mg_train = []
Type_Mg_train = []
#(index>152 && index<158) or (index>164 && index<169) or (index>173 && index<187)
for index,i in enumerate(Mg_ol):
    if index<34:
        Mg_train.append(i)
    if (index>67 and index<101):
        Mg_train.append(i)
    if (index>135 and index<144):
        Mg_train.append(i)
    if (index>152 and index<158):
        Mg_train.append(i)
    if (index>164 and index<169):
        Mg_train.append(i)
    if (index>173 and index<187):
        Mg_train.append(i)
for index,i in enumerate(Type_Mg_ol):
    if index<34:
        Type_Mg_train.append(i)
    if (index>67 and index<101):
        Type_Mg_train.append(i)
    if (index>135 and index<144):
        Type_Mg_train.append(i)
    if (index>152 and index<158):
        Type_Mg_train.append(i)
    if (index>164 and index<169):
        Type_Mg_train.append(i)
    if (index>173 and index<187):
        Type_Mg_train.append(i)        
#Mg_train.append(Mg_ol[:34])
#Mg_train.append(Mg_ol[67:101])
#Mg_train.append(Mg_ol[135:144])
#Mg_train.append(Mg_ol[152:158])
#Mg_train.append(Mg_ol[164:169])
#Mg_train.append(Mg_ol[173:187])
#Mg_train = np.array(Mg_train)
len(Mg_train)


# In[1032]:

Mg_test = []
Type_Mg_test =[]
for index,i in enumerate(Mg_ol):
    if (index>=34 and index<=67) or (index>=101 and index<=135) or (index>=144 and index<=152) or (index>=158 and index<=164) or (index>=169 and index<=173) and index>=187:
        Mg_test.append(i)
len(Mg_test)        
for index,i in enumerate(Type_Mg_ol):
     if (index>=34 and index<=67) or (index>=101 and index<=135) or (index>=144 and index<=152) or (index>=158 and index<=164) or (index>=169 and index<=173) and index>=187:
        Type_Mg_test.append(i)
len(Type_Mg_test)
len(Type_Mg_train)
Type_Mg_train = [int(i) for i in Type_Mg_train]
Type_Mg_test = [int(i) for i in Type_Mg_test]
#Type_Mg_test  


# In[1033]:

Mg_train = np.array(Mg_train)
Mg_train = Mg_train.reshape(-1,1)


# In[1034]:

Type_Mg_train = np.array(Type_Mg_train)
Type_Mg_train = Type_Mg_train.reshape(-1,1)


# In[1035]:

Mg_test = np.array(Mg_test)
Mg_test = Mg_test.reshape(-1,1)


# In[1036]:

Type_Mg_test = np.array(Type_Mg_test)
Type_Mg_test = Type_Mg_test.reshape(-1,1)


# In[ ]:




# In[1037]:

plt.scatter(Al,Type_Al,color='Red', s=25, marker="o")
plt.xlabel('Al')
plt.ylabel('Type_Al')
plt.show()


# In[1038]:

from statistics import median
Al_1=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 1:
        Al_1.append(i["Al"])
median_Al_1=median(Al_1)
a=0
for i in Al_1:
    if (i-median_Al_1)<0:
        a=a-(i-median_Al_1)
    else:
        a=a+(i-median_Al_1)
mad_Al_1=a/len(Al_1)        
print(median_Al_1)
print(mad_Al_1)
print(Al_1)


# In[1039]:


for i in Al_1:
    if i>(median_Al_1 + 0.5747142857142858) or i<(median_Al_1 - 0.5747142857142858):
        print(i)
Al_ol = []
for i in Al:
    Al_ol.append(i)


# In[1040]:

Al_ol.index(0.29)
Al_ol[21]=1.23


# In[1041]:

Al_ol.index(0.47)
Al_ol[38]=1.23


# In[1042]:

Al_ol.index(0.47)
Al_ol[39]=1.23


# In[1043]:

Al_ol.index(0.51)
Al_ol[50]=1.23


# In[1044]:

from statistics import median
Al_2=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 2:
        Al_2.append(i["Al"])
median_Al_2=median(Al_2)
a=0
for i in Al_2:
    if (i-median_Al_2)<0:
        a=a-(i-median_Al_2)
    else:
        a=a+(i-median_Al_2)
mad_Al_2=a/len(Al_2)        
print(median_Al_2)
print(mad_Al_2)
print(Al_2)


# In[1045]:

for i in Al_2:
    if i>(median_Al_2 + 0.7073684210526312) or i<(median_Al_2 - 0.7073684210526312):
        print(i)



# In[1046]:

print(Al_ol.index(0.66))
print(Al_ol.index(0.56))
print(Al_ol.index(0.75))
print(Al_ol.index(0.67))


# In[1047]:

Al_ol[103]=1.46
Al_ol[109]=1.46
Al_ol[111]=1.46
Al_ol[112]=1.46


# In[1048]:

from statistics import median
Al_3=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 3:
        Al_3.append(i["Al"])
median_Al_3=median(Al_3)
a=0
for i in Al_3:
    if (i-median_Al_3)<0:
        a=a-(i-median_Al_3)
    else:
        a=a+(i-median_Al_3)
mad_Al_3=a/len(Al_3)        
print(median_Al_3)
print(mad_Al_3)
print(Al_3)


# In[1049]:

for i in Al_3:
    if i>(median_Al_3 + 0.66470588235294115) or i<(median_Al_3 - 0.66470588235294115):
        print(i)
#no outliars in Al_3


# In[1050]:

from statistics import median
Al_5=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 5:
        Al_5.append(i["Al"])
median_Al_5=median(Al_5)
a=0
for i in Al_5:
    if (i-median_Al_5)<0:
        a=a-(i-median_Al_5)
    else:
        a=a+(i-median_Al_5)
mad_Al_5=a/len(Al_5)        
print(median_Al_5)
print(mad_Al_5)
print(Al_5)


# In[1051]:

for i in Al_5:
    if i>(median_Al_5 + 1.4215384615384617) or i<(median_Al_5 - 1.4215384615384617):
        print(i)


# In[1052]:

Al_ol.index(3.5)
Al_ol[163]=1.76


# In[1053]:

from statistics import median
Al_6=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 6:
        Al_6.append(i["Al"])
median_Al_6=median(Al_6)
a=0
for i in Al_6:
    if (i-median_Al_6)<0:
        a=a-(i-median_Al_6)
    else:
        a=a+(i-median_Al_6)
mad_Al_6=a/len(Al_6)        
print(median_Al_6)
print(mad_Al_6)
print(Al_6)


# In[1054]:

for i in Al_6:
    if i>(median_Al_6 + 1.1599999999999998) or i<(median_Al_6 - 1.1599999999999998):
        print(i)


# In[1055]:

Al_ol.index(0.34)
Al_ol[184]=1.56


# In[1056]:

from statistics import median
Al_7=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 7:
        Al_7.append(i["Al"])
median_Al_7=median(Al_7)
a=0
for i in Al_7:
    if (i-median_Al_7)<0:
        a=a-(i-median_Al_7)
    else:
        a=a+(i-median_Al_7)
mad_Al_7=a/len(Al_7)        
print(median_Al_7)
print(mad_Al_7)
print(Al_7)


# In[1057]:

for i in Al_7:
    if i>(median_Al_7 + 1.03655172413793102) or i<(median_Al_7 - 1.03655172413793102):
        print(i)


# In[1058]:

plt.scatter(Ba,Type_Ba,color='Red', s=25, marker="o")
plt.xlabel('Ba')
plt.ylabel('Type_Ba')
plt.show()


# In[1059]:

from statistics import median,mean
Ba_1=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 1:
        Ba_1.append(i["Ba"])
median_Ba_1=median(Ba_1)
a=0
for i in Ba_1:
    if (i-median_Ba_1)<0:
        a=a-(i-median_Ba_1)
    else:
        a=a+(i-median_Ba_1)
mad_Ba_1=a/len(Ba_1) 
mean_Ba_1=mean(Ba_1)
print(median_Ba_1)
print(mad_Ba_1)
print(Ba_1)


# In[1060]:

for i in Ba_1:
    if i>(median_Ba_1 + 0.038142857142857139) or i<(median_Ba_1 - 0.038142857142857139):
        print(i)#question remain unsolved
Ba_ol = []
for i in Ba:
    Ba_ol.append(i)        


# In[1061]:

from statistics import median,mean
Ba_2=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 2:
        Ba_2.append(i["Ba"])
median_Ba_2=median(Ba_2)
a=0
for i in Ba_2:
    if (i-median_Ba_2)<0:
        a=a-(i-median_Ba_2)
    else:
        a=a+(i-median_Ba_2)
mad_Ba_2=a/len(Ba_2) 
mean_Ba_2=mean(Ba_2)
print(median_Ba_2)
print(mad_Ba_2)
print(Ba_2)


# In[1062]:

Ba_ol.index(3.15)
Ba_ol[106]=0.0


# In[1063]:

from statistics import median,mean
Ba_3=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 3:
        Ba_3.append(i["Ba"])
median_Ba_3=median(Ba_3)
a=0
for i in Ba_3:
    if (i-median_Ba_3)<0:
        a=a-(i-median_Ba_3)
    else:
        a=a+(i-median_Ba_3)
mad_Ba_3=a/len(Ba_3) 
mean_Ba_3=mean(Ba_3)
print(median_Ba_3)
print(mad_Ba_3)
print(Ba_3)


# In[1064]:

Ba_ol.index(0.15)
Ba_ol[161]=0.0


# In[1065]:

from statistics import median,mean
Ba_5=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 5:
        Ba_5.append(i["Ba"])
median_Ba_5=median(Ba_5)
a=0
for i in Ba_5:
    if (i-median_Ba_5)<0:
        a=a-(i-median_Ba_5)
    else:
        a=a+(i-median_Ba_5)
mad_Ba_5=a/len(Ba_5) 
mean_Ba_5=mean(Ba_5)
print(median_Ba_5)
print(mad_Ba_5)
print(Ba_5)


# In[1066]:

Ba_ol.index(2.2)
Ba_ol[163]=0.0


# In[1067]:

from statistics import median,mean
Ba_7=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 7:
        Ba_7.append(i["Ba"])
median_Ba_7=median(Ba_7)
a=0
for i in Ba_7:
    if (i-median_Ba_7)<0:
        a=a-(i-median_Ba_7)
    else:
        a=a+(i-median_Ba_7)
mad_Ba_7=a/len(Ba_7) 
mean_Ba_7=mean(Ba_7)
print(median_Ba_7)
print(mad_Ba_7)
print(Ba_7)


# In[1068]:

for i in Ba_7:
    if i>(median_Ba_7 + 1.6624137931034481) or i<(median_Ba_7 - 1.6624137931034481):
        print(i)


# In[1069]:

Ba_ol.index(2.88)
Ba_ol[207]=0.81


# In[1070]:

plt.scatter(Na,Type_Na,color='Red', s=25, marker="o")
plt.xlabel('Na')
plt.ylabel('Type_Na')
plt.show()


# In[1071]:

from statistics import median,mean
Na_1=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 1:
        Na_1.append(i["Na"])
median_Na_1=median(Na_1)
a=0
for i in Na_1:
    if (i-median_Na_1)<0:
        a=a-(i-median_Na_1)
    else:
        a=a+(i-median_Na_1)
mad_Na_1=a/len(Na_1) 
mean_Na_1=mean(Na_1)
print(median_Na_1)
print(mad_Na_1)
print(Na_1)


# In[1072]:

for i in Na_1:
    if i>(median_Na_1 + 1.1854285714285719) or i<(median_Na_1 - 1.1854285714285719):
        print(i)
Na_ol=[]
for i in Na:
    Na_ol.append(i)


# In[1073]:

Na_ol.index(14.77)


# In[1074]:

Na_ol[21]=13.195


# In[1075]:

from statistics import median,mean
Na_2=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 2:
        Na_2.append(i["Na"])
median_Na_2=median(Na_2)
a=0
for i in Na_2:
    if (i-median_Na_2)<0:
        a=a-(i-median_Na_2)
    else:
        a=a+(i-median_Na_2)
mad_Na_2=a/len(Na_2) 
mean_Na_2=mean(Na_2)
print("median",median_Na_2)
print("MAD",mad_Na_2)
print(Na_2)


# In[1076]:

for i in Na_2:
    if i>(median_Na_2 + 1.3456578947368425) or i<(median_Na_2 - 1.3456578947368425):
        print(i)


# In[1077]:

print(Na_ol.index(14.86))
print(Na_ol.index(11.45))
print(Na_ol.index(10.73))
print(Na_ol.index(11.23))
print(Na_ol.index(11.02))


# In[1078]:

Na_ol[70]=13.155
Na_ol[105]=13.155
Na_ol[106]=13.155
Na_ol[110]=13.155
Na_ol[111]=13.155


# In[1079]:

from statistics import median,mean
Na_3=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 3:
        Na_3.append(i["Na"])
median_Na_3=median(Na_3)
a=0
for i in Na_3:
    if (i-median_Na_3)<0:
        a=a-(i-median_Na_3)
    else:
        a=a+(i-median_Na_3)
mad_Na_3=a/len(Na_3) 
mean_Na_3=mean(Na_3)
print("median",median_Na_3)
print("MAD",mad_Na_3)
print(Na_3)


# In[1080]:

for i in Na_3:
    if i>(median_Na_3 + 1.0570588235294118) or i<(median_Na_3 - 1.0570588235294118):
        print(i)


# In[1081]:

Na_ol.index(12.16)
Na_ol[149]=13.42


# In[1082]:

from statistics import median,mean
Na_5=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 5:
        Na_5.append(i["Na"])
median_Na_5=median(Na_5)
a=0
for i in Na_5:
    if (i-median_Na_5)<0:
        a=a-(i-median_Na_5)
    else:
        a=a+(i-median_Na_5)
mad_Na_5=a/len(Na_5) 
mean_Na_5=mean(Na_5)
print("median",median_Na_5)
print("MAD",mad_Na_5)
print(Na_5)


# In[1083]:

for i in Na_5:
    if i>(median_Na_5 + 1.4884615384615383) or i<(median_Na_5 - 1.4884615384615383):
        print(i)


# In[1084]:

Na_ol.index(11.03)
Na_ol[166]=12.97


# In[1085]:

from statistics import median,mean
Na_6=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 6:
        Na_6.append(i["Na"])
median_Na_6=median(Na_6)
a=0
for i in Na_6:
    if (i-median_Na_6)<0:
        a=a-(i-median_Na_6)
    else:
        a=a+(i-median_Na_6)
mad_Na_6=a/len(Na_6) 
mean_Na_6=mean(Na_6)
print("median",median_Na_6)
print("MAD",mad_Na_6)
print(Na_6)


# In[1086]:

for i in Na_6:
    if i>(median_Na_6 +1.7866666666666671) or i<(median_Na_6 -1.7866666666666671):
        print(i)


# In[1087]:

Na_ol.index(17.38)
Na_ol[184]=14.4


# In[1088]:

from statistics import median,mean
Na_7=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 7:
        Na_7.append(i["Na"])
median_Na_7=median(Na_7)
a=0
for i in Na_7:
    if (i-median_Na_7)<0:
        a=a-(i-median_Na_7)
    else:
        a=a+(i-median_Na_7)
mad_Na_7=a/len(Na_7) 
mean_Na_7=mean(Na_7)
print("median",median_Na_7)
print("MAD",mad_Na_7)
print(Na_7)


# In[1089]:

for i in Na_7:
    if i>(median_Na_7 + 1.42034482758620649) or i<(median_Na_7 - 1.42034482758620649):
        print(i)


# In[1090]:

Na_ol.index(11.95)
Na_ol[201]=14.39


# In[1091]:

from statistics import median,mean
Na_7=[]
for index,i in glass_df.iterrows():
    if i["Type"] == 7:
        Na_7.append(i["Na"])
median_Na_7=median(Na_7)
a=0
for i in Na_7:
    if (i-median_Na_7)<0:
        a=a-(i-median_Na_7)
    else:
        a=a+(i-median_Na_7)
mad_Na_7=a/len(Na_7) 
mean_Na_7=mean(Na_7)
print("median",median_Na_7)
print("MAD",mad_Na_7)
print(Na_7)


# In[1092]:

print(len(Ri))
print(len(Na_ol))
print(len(Mg_ol))
print(len(Al_ol))
print(len(Si))
print(len(K))
print(len(Ca))
print(len(Ba_ol))
print(len(Fe))


# In[ ]:




# In[1093]:

Na_ol_df=pd.DataFrame(Na_ol,columns=["Na"]).astype(np.float32)
Ri_df=pd.DataFrame(Ri,columns=["Ri"]).astype(np.float32)
Mg_ol_df=pd.DataFrame(Mg_ol,columns=["Mg"]).astype(np.float32)
Ba_ol_df=pd.DataFrame(Ba_ol,columns=["Ba"]).astype(np.float32)
Al_ol_df=pd.DataFrame(Al_ol,columns=["Al"]).astype(np.float32)
Si_df=pd.DataFrame(Si,columns=["Si"]).astype(np.float32)
k_df=pd.DataFrame(K,columns=["K"]).astype(np.float32)
Ca_df=pd.DataFrame(Ca,columns=["Ca"]).astype(np.float32)
Fe_df=pd.DataFrame(Fe,columns=["Fe"]).astype(np.float32)


# In[1094]:

for j,i in Mg_ol_df.iterrows():
    print(type(i["Mg"]))


# In[1095]:

df=Ri_df.join(Na_ol_df)


# In[1096]:

df=df.join(Mg_ol_df)
df=df.join(Al_ol_df)
df=df.join(Si_df)
df=df.join(k_df)
df=df.join(Ca_df)
df=df.join(Ba_ol_df)
df=df.join(Fe_df)
df


# In[1097]:

Type = list(map(int, Type))


# In[1098]:

Type_df=pd.DataFrame(Type,columns=["Type"])


# In[1099]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( df, Type_df, test_size = 0.3, random_state = 100)


# In[1100]:

from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

#changing list into numpy array for clustering
#np_Ri = np.array(Ri)
#np_Na = np.array(Na)
#np_Mg = np.array(Mg)
#np_Al = np.array(Al)
#np_Si = np.array(Si)
#np_K = np.array(K)
#np_Ca = np.array(Ca)
#np_Ba = np.array(Ba)
#np_Fe = np.array(Fe)
#np_Type_Ri = np.array(Type_Ri)
#np_Type_Na = np.array(Type_Na)
#np_Type_Mg=np.array(Type_Mg)
#np_Type_Al = np.array(Type_Al)
#np_Type_Si = np.array(Type_Si)
#np_Type_K = np.array(Type_K)
#np_Type_Ca = np.array(Type_Ca)
#np_Type_Ba = np.array(Type_Ba)
#np_Type_Fe = np.array(Type_Fe)


# In[ ]:




# In[ ]:

#making data grouped so as to perform clustering
#cluster_Ri_type= np.insert(np_Type_Ri, np.arange(len(np_Ri)), np_Ri)
#cluster_Ri_type = np.reshape(cluster_Ri_type,(209,2))


# In[ ]:

#making data grouped so as to perform clustering
#cluster_Na_type= np.insert(np_Type_Na, np.arange(len(np_Na)), np_Na)
#cluster_Na_type = np.reshape(cluster_Na_type,(212,2))
#cluster_Mg_type= np.insert(np_Type_Mg, np.arange(len(np_Mg)), np_Mg)
#cluster_Mg_type = np.reshape(cluster_Mg_type,(214,2))
#cluster_Al_type= np.insert(np_Type_Al, np.arange(len(np_Al)), np_Al)
#cluster_Al_type = np.reshape(cluster_Al_type,(214,2))
#cluster_Si_type= np.insert(np_Type_Si, np.arange(len(np_Si)), np_Si)
#cluster_Si_type = np.reshape(cluster_Si_type,(212,2))
#cluster_K_type= np.insert(np_Type_K, np.arange(len(np_K)), np_K)
#cluster_K_type = np.reshape(cluster_K_type,(211,2))
#cluster_Ca_type= np.insert(np_Type_Ca, np.arange(len(np_Ca)), np_Ca)
#cluster_Ca_type = np.reshape(cluster_Ca_type,(214,2))
#cluster_Ba_type= np.insert(np_Type_Ba, np.arange(len(np_Ba)), np_Ba)
#cluster_Ba_type = np.reshape(cluster_Ba_type,(211,2))
#cluster_Fe_type= np.insert(np_Type_Fe, np.arange(len(np_Fe)), np_Fe)
#cluster_Fe_type = np.reshape(cluster_Fe_type,(214,2))


# In[ ]:

#from sklearn.cluster import KMeans
#kmeans = KMeans(n_clusters=3, random_state=0).fit(cluster_Ri_type)


# In[ ]:

#kmeans.labels_


# In[ ]:




# In[ ]:



