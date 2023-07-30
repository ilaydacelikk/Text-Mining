# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:53:12 2023

@author: ilayd
"""

#Ödevim için Bones veri seti kullanacağım.
 
import numpy as np
import pandas as pd
data = pd.read_csv("bone.csv")
#Veri setine genel bakış yapmak için
data.head()
#Tanımlayıcı istatistikler
data.describe().T
#Veri setinde eksik gözlem var mı?
data.isnull().values.any()
# 'True' çıktı yani verimizde eksik gözlem var.
#Eksik gözlemleri veri setinden çıkaralım.
dtnew = data[np.isfinite(data).all(1)]
print(dtnew)
dtnew.isnull().values.any()
#Yeni veriyi incelediğimizde eksik verilerin olmadığı görülüyor.
#Değişkenlerin birbirleri arasındaki korelasyonları incelemek için:
dtnew.corr()
#Bağımlı değişkendeki olayları bağımsız değişkenlerce ne kadar açıklayabiliriz önemli olan bu. O 
#yüzden korelasyon değerlerini incelediğimizde; diğer gözlemlerle ilişkisi güçlü olanı bağımsız 
#değişken olarak alırız. 
#ikili ikili değerleri grafik şeklinde gösterecek pair plotu çizdirelim.
#Pair plot
import seaborn as sbn
sbn.pairplot(dtnew, kind="reg")
#veri setimiz çok büyük olduğu için bu grafiklerin çıkması biraz zaman alabilir.
#Basit doğrusal regresyon için bağımsız değişken Height seçilecektir ve bağımlı değişken için ise 
#Weight değişkeni seçilecektir.
sbn.jointplot(x="Height", y="Weight",data=dtnew,kind="reg")
#Bağımlı ve bağımsız değişken arasındaki saçılım grafiği ile beraber değişkenlerin dağılım bilgilerini de 
#eksenlerde gösterir.
# x eksenindeki Height değeri artarken y eksenindeki Weight değeri de artıyor. Aralarında doğrusal 
#bir ilişki olduğu görülüyor. 
#Ayrıca y ekseninin az da olsa normal dağılımı andırdığı görülüyor.
#Dağılım yoğunluğunun kabaca 150-180 arasında olduğu ve aykırı değerlerinde yer aldığı görülüyor.
# Pythonda regresyon modeli icin birden cok yol vardir. En pratik olan iki yol uzerinden gidecegiz.
# Birincisi istatistiksel modeller kutuphanesi olan statmodels kutuphanesi Ikincisi ve diger tum 
#modellemeleri de iceren en kapsamli kutuphane olan sklearn
x = dtnew[["Height"]]
y = dtnew[["Weight"]]
import statsmodels.api as sm
#Bu kütüphane kullanılırken sabit parametre için 1 lerden oluşan matris kolonu eklenmelidir.
x = sm.add_constant(x)
x.head()
#Sabit parametreye const altında görülen 1 leri eklemiş oluyoruz.
#OLS yapısını kullanarak lineer regresyon modelini kuralım.
new = sm.OLS(y,x)
mod = new.fit()
mod.summary()
#y ve x arasında OLS yani en küçük kareler tahmin edicisi oluşturuyoruz ve bu tahminciyle fit ediyoruz 
#FİT ETMEK: veri setinde dönüşüm yapılacağında veya bir model kurulacağında kullanırız. 
#çıktıyı incelediğimizde Rdeğeri 0,441 görülüyor. Bu bağımlı değişkendeki değişimin %44,1 'sının 
#bağımsız değişken tarafından açıklandığını söyler. Daha iyi model de kurulabilir.
#Bu modele bir bağımlı değişken daha atadığımızda modelin anlamlı olup olmadığını R-squared 
#yerine adj R-squarede bakarak yorumlarız.
#AIC(Akaike ölçütü)= Belirli bir veri kümesi için kaliteli bir istatistiksel 
#göreceli model ölçüsüdür. Yani, veri modelleri koleksiyonu verildiğinde, AIC her
# model kalitesini, diğer modellerin her birini göreceli olarak tahmin ediyor. 
#AIC : 1.058e+04
#BIC : 1.059e+04 
#kuracağımız diğer modellerle karşılaştırırken AIC ve BIC değeri düşük olan modeli tercih ederiz.
#const değeri : (-96,7229) bu grafiğin y eksenini -796,7229 noktasında kestiğini söyler
# P > |t| değeri 0 çıktı anlamlı 
#regresyon modelini dataframe yapısı üzerinden kurmak istersek: direkt dataframe içinden formül 
#üreteceğiz
import statsmodels.formula.api as smf 
model = smf.ols("Weight~Height",data=dtnew)
mod=model.fit()
mod.summary()
# Eger ayni seyi data frame yapisindan yapmak isteseydik
import statsmodels.formula.api as smf
mod.summary() #tüm verileri inceledik
#çıktılar ayrıca da elde edilip incelenebilir. 
mod.params
#güven aralıkları
mod.conf_int()
#güven aralığında 0 varsa anlamsız, 0 yoksa anlamlı 
#bizim modelimiz anlamlı 
#p-value değerini grafikten de bakabiliriz.Kendine ait formülü var onunla da bakabiliriz.
mod.f_pvalue 
#Modelin anlamlı olması için p-value değerinin 0'dan farklı çıkması gerekir.
#Bizim değerimiz '9.49e-187' çıktı yani modelimiz anlamlı.
mod.fvalue
#Aynı şekilde F istatistiğinin değerini tablo çıktısında da görebilirz.
#Kendi formülüyle de görülebilir. F istatistiğimizde '1152.6277136945075' çıktı.
#parametrelerin anlamlılığını kullanmak için T değerleri: 
 
mod.tvalues 
#çıktımız: Intercept -20.020785
#Weight 33.950371
#dtype: float64 şeklinde 
mod.mse_model # '93076.13385888295' çıktı
mod.rsquared # '0.441006845640309' çıktı.
mod.rsquared_adj #'0.4406242356783927' çıktı.
mod.fittedvalues #fit edilen y değerleri 0'dan 1536.sıraya kadar gösteriyor.
#Modelimiz ne kadar iyi fit edilmiş bakalım. 
sbn.regplot(x=y, y=mod.fittedvalues)
pl = sbn.regplot(x=dtnew["Height"], y=dtnew["Weight"], ci = None, scatter_kws = {"color":"r","s":9})
pl.set_title("Regresyon modeli")
#İkinci olarak ve en çok kullanılan kütüphane sklearn kütüphanesidir.
from sklearn.linear_model import LinearRegression
model=LinearRegression()
mod = model.fit(x,y)
# Elde edilecek parametre degerleri vs yine buradan elde edilir
mod.intercept_ 
# array([-96.72289302])
mod.coef_ #model katsayısı
#çıktısı array([[0. , 0.9882592]])
mod.score(x,y) # Modelin R^2 degeri
#0.441006845640309 çıktı 
mod.predict(x) # fit degerleri
#array([[65.35161622],
# [63.37509782],
# ...
# [73.25768985]]) çıktı 
#bilinmeyen bağımsız değişken değerlerine karşılık gelecek bağımlı
#değişkenin değerlerini tahmin etmek için sklearn kütüphanesindeki 
#predict fonksiyonu kullanılabilir.
x_yeni= [[1,5],[1,15],[1,8],[1,30]] #sabit parametreye karşılık gelen 
#değerleri yazıyoruz. Yni (1 e 5,1 e 15,1e 8 falan)
y_tahmin = mod.predict(x_yeni)
#orijinal veri setindeki bilgiyi kullanarak alt grupta yeni çıktılar
#elde ediliyor. 
#mse ve r^2 değerlerini hesplamak için 
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y,mod.predict(x))
#tüm y değerlerinden y şapka değerlerini çıkartıyor .
#karelerini alıyor. Bu kareler ortalamasını en son mse olarak 
#adlandırılıyor. 
#mse çıktısı '80.64086318034609' 
r2_score(y,mod.predict(x)) #bu zaten model çıktısı
# olarak veriliyor. R2 değeri yine 0.441 olarak çıktı.
#y ve y sapkaları matplot kütüphanesini kullanarak çizdirelim.
import matplotlib.pyplot as mp
# model çok da yeterli değil.
mp.scatter(dtnew["Weight"], dtnew["Height"], color='orange')
mp.plot(dtnew["Weight"], mod.predict(x), color='yellow')
# Coklu dogrusal regresyon modeli
x = dtnew.drop("Weight",axis = 1)
y = dtnew["Weight"]
# Veri setini train(eğitim) ve test olarak ayirmak icin
from sklearn.model_selection import train_test_split
x_egitim, x_test, y_egitim, y_test, = train_test_split(x,y, test_size=151) 
x_egitim.shape #(1312, 39)
x_test.shape #(151,39)
y_egitim.shape #(1312,)
y_test.shape #(151,)
mod=model.fit(x_egitim,y_egitim)
mod.intercept_ #-136.2386613025334 çıktı
#Test kümesindeki bağımlı değişken değerlerini tahmin etmek için 
#test kümesindeki bağımsız değişkenler kullanılır.
#stats models kütüphanesiyle kurulan model
model=sm.OLS(y_egitim,x_egitim)
mod=model.fit()
mod.summary() #grafiğin çıktısı 
#çıktıyı incelediğimizde R2 değerinin 0,999 olduğunu görüyoruz. Daha 
#önceki veriye göre artış göstermiş. F istatistiği de artmış ama p-value
#değeri 0 gözüküyor. 
#R2 değerimiz ne kadar yüksek olursa olsun modelin anlamlılığını p-value
#belirlediği için bu modelin anlamsız olduğunu söyleyebiliriz.
# Güven aralıklarına da baktığımızda anlamsız değerler aldığı görülüyor. 
#o halde bazı değişkenleri modelden çıkartarak modelimizi anlamlı hale
#getirebiliriz. 
#sklearn kütüphanesiyle
model=LinearRegression()
mod=model.fit(x_egitim,y_egitim)
mod.intercept_
mod.coef_
fit=mod.predict(x_egitim)
mp.scatter(y_egitim,fit,color="purple") 
#model daha açıklayıcı,daha doğrusal bir hal aldı. 
# Test verisindeki bagimli degisken degerlerini tahminlemek icin
#test kümesindeki bağımsız değişkenler kullanılır.
tahminler = mod.predict(x_test)
mp.scatter(y_test,tahminler,color="red") 
#doğrusal bir grafik çıktı. 
mean_squared_error(y_test,tahminler) 
# çıktısı '0.5333067054445609'
#Bunun yanı sıra çoklu doğrusal regresyonda dikkat edilmesi gereken bir nokta da bağımsız 
#değişkenlerin kendi aralarında ilişkili olma durumları.Yukarıdaki matrisimizi 
#incelediğimizde bağımsız değişkenler ile bağımlı değişken arasında
#az bir oran da olsa aralarında ilişki olduğu görülmektedir. Bu da bize 
#istemediğimiz sorun olan multicolinerty sorununu işaret eder. 
#Diğer regresyon modelleri
#Temel bileşenler Regresyonu (PCR)
#Açıklayıcı değişkenler arasında çoklu doğrusal bağlantı problemi varsa
#(açıklayıcı değişkenler arasında yüksek derecede ilişki varsa)
#EKK tahmincisi elde edilemeyebilir, ya da yanlı parametre
#tahminleri elde edilebilir.
#PCR temelinde eldeki bağımsız değişkenler setini
#decomposition(ayrıştırma) ile temel bileşen ve temel bileşen skorlarına ayırma yatar.
#Temel bileşen skorları, eldeki tüm bağımsız değişkenlerin doğrusal bir kombinasyonu ile
#elde edilir ve skorlar birbirlerinden bağımsızıdr. Regresyon modeli sonrasında
#bağımlı değişken ile skorlar arasında kurularak gerçek regresyon modeli
#yakınsanmaya çalışılır. Skorlar oluşturulurken kullanılan mekanizma, bağımsız
#değişkenlerin arasındaki kovaryansın maksimize edilmesine dayanır.
#PCR yapılırken yapı itibari ile değişkenler ölçeklendirilir(ortalamadan
#arındırılması gerekiyor, isteğe bağlı olarak scale edilmesi gerekir.)
from sklearn.preprocessing import scale 
from sklearn.decomposition import PCA 
#ayrıştırma modülünden temel bileşenler analizini aktif hale getiriyoruz.
from sklearn import model_selection
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
cv = KFold(n_splits = 5, shuffle =True, random_state=42)
y=dtnew["Height"]
x=dtnew.drop("Height",axis =1)
pca =PCA() 
x_egitim_scaled = pca.fit_transform(scale(x_egitim))
#ortalaması0,varyansı 1 olan veri setine dönüştürdük .
#K-fold cross-validation (K=5)
cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state=47)
regr = LinearRegression()
mse=[]
#Sadece sabit parametrenin olduğu modelden MSE değerleri hesaplayalım
#n_repeats=tekrarlama sayısı
#mse değerlerini boş vektör olarak belirledik.
#çok fazla verimiz olduğu için bunları K değerine indirgedik.
#Temel bileşen sayısı arttıkça verinin açıklanma oranı o kadar artar.
#Matrisimizin boyutunu indirgeyeceğiz.
#K-fold cross validation ile mse değerlerinin hesaplanması
for i in np.arange(1,10):
 score = -1*model_selection.cross_val_score(regr, x_egitim_scaled[:,:i], y_egitim,
 cv=cv,scoring="neg_mean_squared_error").mean()
 mse.append(score)
print(mse)
#ortalama hata karesi (mse) ne kadar çok değer alınırsa o kadar çok düşer.
#yani 1 değer alındığında 148,16 çıkıyor. 2 değer alındığında 104,84 oluyor.
#7 değer alındığında 90,46 gibi gibi değer sayısı arttıkça düşüyor.
#bunun grafiğini de inceleyelim.,
#cross-validation sonuçlarına göre
mp.plot(mse)
mp.xlabel("Temel bileşen sayısı")
mp.ylabel("mse")
#grafiktede görüldüğü gibi 1.den sonra,2.temel bileşen skoruu 
#eklediğimizde grafikte çok keskin bir düşüş,kırılma var.
model =LinearRegression()
pcreg=model.fit(x_egitim_scaled, y_egitim)
#Model parametre tahminleri
pcreg.intercept_
pcreg.coef_
y_hat = pcreg.predict(x_egitim_scaled)
sbn.regplot(y_egitim, y_hat)
#test kümesindeki y'yi ne kadar iyi tahminlendiğine bakacağız şimdi.
#Tahminleme aşaması
pca=PCA()
x_test_scaled=pca.fit_transform(scale(x_test))
predicted_y=pcreg.predict(x_test_scaled)
sbn.regplot(y_test,predicted_y)
r2_score(y_test,predicted_y)
#eğitim kümesindeki tahminler çok daha iyi açıklanırken
#test kümesindeki tahminler o kadar da iyi açıklanmadı. 
#temel bileşen sayımıza bağlı olarak. 
#Kısmi en küçük kareler regresyonu(PLS)
#PLS regresyonu da PCR'ye benzer mantıkta çalışır. Bu model kurulurken kısmi EKK
#bileşenleri ve bu bileşenlere karşılık gelecek skorlar iteratif bir şekilde oluşturulur.
#PLS ve PCR arasındaki temel fark, PCR temel bileşenleri hesaplarken değişkenler arasındaki
#kovaryansı maksimum yapmaya çalışırken, PLS, hem bağımlı hem de bağımsız değişkenler arasındaki 
#kovaryansı maksimum yapmaya çalışır. Dolayısı ile, PLS skorları PCR skorlarına kıyasla
#verinin orjinali hakkında daha çok bilgi içerir. PLS iteratif bir yöntemdir ve çözümlemeler
#çeşitli algoritmalara bağlı olarak elde edilir. En çok tercih edilen algoritmalra
#SIMPLS,NIPALS,Kernel PLS,Hybrid PLS,... algoritmalarıdır.
from sklearn.cross_decomposition import PLSRegression
pls_model = PLSRegression().fit(x_egitim,y_egitim)
pls_model.coef_
#beta katsayıları verildi
y_hat_pls0 = pls_model.predict(x_egitim)
mean_squared_error(y_egitim, y_hat_pls0)
#PLS bileşeni belirterek model kurma
pls_model1 = PLSRegression(n_components=4).fit(x_egitim,y_egitim)
y_hat_pls1 = pls_model1.predict(x_egitim)
#Optimum PLS bileşen sayısının belirlenmesi (optimizasyon yapılması) gereklidir.
from sklearn.model_selection import cross_val_predict
def optimum_pls(x,y,ncomp):
 model = PLSRegression(n_components=ncomp) 
 cv.step=cross_val_predict(model,x,y,cv=10) #modeli,değişkenimizi,bağımlı değişkenimizi ve kaça 
böleceğimizi yazdık. 
 rsq = r2_score(y,cv.step) #gerçek değer ve tahmin ettiğimiz değeri yazdık
 mse = mean_squared_error(y, cv.step)
 return(cv.step,rsq,mse) #dışa aktarıyoruz
#10 bileşen sayısı için metrikleri hesapla
#tüm değişken sayılarımızı da kullanabiliriz ama bizim
#değişken sayımız çok olduğu için 10 tane bileşen kullandık.
r2 = []
mse_sonuc = []
for i in np.arange(1, 10):
 cv.step, rsq, mse = optimum_pls(x_egitim,y_egitim,i)
 r2.append(rsq)
 mse_sonuc.append(mse)
#10 tane değer için r2 çıktılar:
#0.7094959958322964,
#0.939860921123082,
#0.9688815111154656,
#0.9802502584778593,
#0.985325238251073,
#0.9924391538277186,
#0.9936171916465956,
#0.9938044470873626,
#0.9938162749701533,
#çıktıları incelediğimizde 1 tane bileşen kullandığımızda
#0,7094..'lük bir açıklama var.2 tane bileşen kullandığımızda
#0,9398..'lik bir açıklama derken 6.bileşenden sonra
# biz 0,99'luk bir açıklama yapıyoruz ki daha fazla 
#oran değişikliği olmuyor. 
#mse_sonuc'un çıktısı : 
#42.700273680216625,
#8.83965484151501,
#4.574009214404211,
#2.9029526478475387,
#2.156997266410897,
#1.11134509741234,
#0.938189008167354,
#0.9106649174348742,
#0.9089263739897618, 
#bu sonuçları incelediğimizde optimum en küçük kısmi 
#regresyon çıktıları için bileşen sayısı 6.
#bunun grafiğini çizelim.
def pls_grafigi(degerler,y_ekseni,fonk):
 with mp.style.context("ggplot"): #en çok kullanılan grafik olan ggplot ı kullanıyoruz.
 mp.plot(np.arange(1,10),np.array(degerler),"-v",color ="black",mfc="green")
 if fonk == "min":
 indeks = np.argmin(degerler)
 else:
 indeks = np.argmax(degerler)
 mp.plot(np.arange(1, 10)[indeks],np.array(degerler)[indeks],"P",ms=10,mfc ="red")
 mp.xlabel("PLS bileşen sayısı")
 mp.title("PLS Grafiği")
 mp.show()
 
pls_grafigi(mse_sonuc,"MSE","min") #fonksiyou minimuma göre çizdirelim
#burada da 2 ye keskin bir düşüş yapmış ardından daha hafif eğrilerle azalmış.
pls_grafigi(r2,"R2","max") #fonksiyonu maximuma göre çizdirdik.
#r2 leri incelediğimzde 1 bileşenle açıklama yaptığımızda
#0,70lik bir açıklama yapıyor. 2 bileşen kullandığımızda keskin bir artış 
#ve 6.bileşenden sonra doğrusallığa yakın bir grafik görüyoruz. 
#RİDGE REGRESYONU
#Çok değişkenli regresyon verilerini analiz etmede kullanılır. 
#Amaç hata kareler toplamını minimize eden katsayıları, bu 
#katsayılara bir ceza uygulayarak bulmaktır.Çok boyutluluğa çözüm sunar.
#Burada yine bir optimizasyon mevcut. 
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha = 0.35).fit(x_egitim,y_egitim) #Ridge() de yazabiliridk. Alpha 1 olarak 
atanmış
ridge_model.coef_ #katsayılar çıktısı
y_hat_alpha1 = ridge_model.predict(x_egitim)
mean_squared_error(y_egitim, y_hat_alpha1)
sbn.regplot(y_egitim,y_hat_alpha1)
ridge_model2 = Ridge().fit(x_egitim,y_egitim)
y_hat_alpha2 = ridge_model2.predict(x_egitim)
mean_squared_error(y_egitim, y_hat_alpha2)
sbn.regplot(y_egitim,y_hat_alpha2)
# grafikleri karşılaştırdığımızda alphanın farlı değerlerine 
#göre farklı model tahminler elde ediyoruz. 
#alphanın optimize değerini bulup en iyi değeri bulmaya çalışırız.
#Lambda optimizasyonu
r2 = []
mse_sonuc = []
alpha_cand = np.array([0,0.05,0.1,0.2,0.5,1,5,10]) #alpha için rastgele değerler tanımladık
for i in alpha_cand:
 model_i = Ridge(alpha=i).fit(x_egitim,y_egitim)
 yhats=model_i.predict(x_egitim)
 r2.append(r2_score(y_egitim,yhats))
 mse_sonuc.append(mean_squared_error(y_egitim, yhats))
#sonuçları incelediğimizde lambdanın 0 olduğu sonuç
#bize en iyi sonucu veriyor. 
#Lasso Regresyonu
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.35).fit(x_egitim,y_egitim)
lasso_model.coef_
y_hat_0 = lasso_model.predict(x_egitim)
mean_squared_error(y_egitim, y_hat_alpha3)
sbn.regplot(y_egitim,y_hat_alpha3)
 
# DOĞRUSAL OLMAYAN REGRESYON MODELİ 
#K- en yakın komşu(KNN) regresyonu
#hem sınıflandırma hem de regresyon problemlerini çözmek 
#için kullanılabilen denetimli makine öğrenimi algoritmasıdır.
#elimize yeni bir gözlem eklendiğinde hangi gruba sınıflandıracağımızı bilemeyiz.
#hangi gözlem kendine yakınsa o gözlemin olduğu gruba atama yapılır. 
y=dtnew["Height"]
X=dtnew.drop("Height", axis = 1)
x_egitim, x_test, y_egitim, y_test = train_test_split(X,y,test_size=0.30,random_state=19)
KNN_model = KNeighborsRegressor().fit(x_egitim,y_egitim)
#K en yakın komşuluk regresyon modelini kuruyor
#K değerine bakmak için
KNN_model.n_neighbors # K değerimiz 5 çıktı
y_hat_knn5 = KNN_model.predict(x_egitim)
mean_squared_error(y_egitim, y_hat_knn5)
r2_score(y_egitim,y_hat_knn5)
mse = []
r2 = []
for k in range(20):
 k = k+1
 KNN_model = KNeighborsRegressor(n_neighbors = k).fit(x_egitim,y_egitim)
 y_hat_k = KNN_model.predict(x_test)
 mse.append(mean_squared_error(y_test, y_hat_k))
 r2.append(r2_score(y_test,y_hat_k))
 
mse
r2
mse_df = pd.DataFrame(mse)
mse_df.plot()
#Grid Search kullanarak en iyi K değerini belirlemek
from sklearn.model_selection import GridSearchCV
KNN_arg = KNeighborsRegressor()
k_params = {"n_neighbors": [2,3,4,5,6,7,8,9,10,11,12,13,14]}
KNN_model = GridSearchCV(KNN_arg,k_params,cv =5)
KNN_model.fit(x_egitim,y_egitim)
KNN_model.best_params_
#Support Vector Machine (SVR) Regresyonu
#Karar Destek Makinaları.
from sklearn.svm import SVR
#Doğrusal olan SVR
SVR_model = SVR(kernel = "linear").fit(x_egitim,y_egitim)
y_hat_svr_l = SVR_model.predict(x_egitim)
mean_squared_error(y_egitim, y_hat_svr_l)
r2_score(y_egitim,y_hat_svr_l)
predicted_y_svr_l = SVR_model.predict(x_test)
mean_squared_error(y_test, predicted_y_svr_l)
r2_score(y_test,predicted_y_svr_l)
#Doğrusal olmayan SVR 
SVR_model_nl = SVR(kernel="rbf").fit(x_egitim,y_egitim)
y_hat_svr_nl = SVR_model_nl.predict(x_egitim)
mean_squared_error(y_egitim, y_hat_svr_nl)
r2_score(y_egitim,y_hat_svr_nl)
predicted_y_svr_nl = SVR_model_nl.predict(x_test)
mean_squared_error(y_test, predicted_y_svr_nl)
r2_score(y_test,predicted_y_svr_nl)
#En iyi modeli belirlemek için Grid Search uygulaması
params_svr = {"C":np.arange(0.1,2,0.4)}
gs_SVR_model_l = GridSearchCV(SVR_model, params_svr,cv=10).fit(x_egitim,y_egitim)
gs_SVR_model_nl = GridSearchCV(SVR_model_nl, params_svr,cv=10).fit(x_egitim,y_egitim)
gs_SVR_model_l.best_params_
gs_SVR_model_nl.best_params_
bp = pd.Series(gs_SVR_model_l.best_params_)[0]
best_l_svr_model = SVR(kernel="linear",C=bp).fit(x_egitim,y_egitim)
best_nl_svr_model = SVR(kernel="rbf",C=bp).fit(x_egitim,y_egitim)
y_hat_l_best = best_l_svr_model.predict(x_test)
y_hat_nl_best = best_nl_svr_model.predict(x_test)
mean_squared_error(y_test, y_hat_l_best)
r2_score(y_test, y_hat_l_best)
mean_squared_error(y_test, y_hat_nl_best)
r2_score(y_test, y_hat_nl_best)
#Yapay Sinir Ağları(ANN)-(NLP)
#Standartlaştırma gerekli
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
scl = StandardScaler()
scl.fit(x_egitim)
x_egitim_scl = scl.transform(x_egitim)
x_test_scl = scl.transform(x_test)
ann_model = MLPRegressor().fit(x_egitim_scl,y_egitim)
ann_model.n_layers_
ann_model.hidden_layer_sizes
y_hat_ann0 = ann_model.predict(x_egitim_scl)
mean_squared_error(y_egitim, y_hat_ann0)
r2_score(y_egitim,y_hat_ann0)
y_predict_ann0 = ann_model.predict(x_test_scl)
mean_squared_error(y_test, y_predict_ann0)
r2_score(y_test,y_predict_ann0)
#Grid Search ile optimizasyon
params_ann = {"alpha":[0.1,0.01,0.02,0.005],
 "hidden_layer_sizes":[(20,20),(100,50,150),(300,200,100)],
 "activation":["relu","logistic"]}
gs_ann_model = GridSearchCV(ann_model, params_ann,cv=5)
gs_ann_model.fit(x_egitim_scl,y_egitim)
gs_ann_model.best_params_
best_ann = MLPRegressor(alpha=0.1, hidden_layer_sizes=[300,200,100],activation="relu")
model_best = best_ann.fit(x_egitim_scl,y_egitim) 
best_yhat = model_best.predict(x_egitim_scl)
mean_squared_error(y_egitim, best_yhat)
r2_score(y_egitim,best_yhat)
best_predicted = model_best.predict(x_test_scl)
mean_squared_error(y_test, best_predicted)
r2_score(y_test,best_predicted)
x_egitim, x_test, y_egitim, y_test = train_test_split(X,y ,test_size = 0.30, random_state=15)
from sklearn.tree import DecisionTreeRegressor
cart_model = DecisionTreeRegressor().fit(x_egitim,y_egitim)
fitted_cart = cart_model.predict(x_egitim)
mean_squared_error(y_egitim, fitted_cart) #0 çıktı
r2_score(y_egitim, fitted_cart) #1 çıktı
preds_cart = cart_model.predict(x_test)
mean_squared_error(y_test, preds_cart) #çıktı '11.117312072892938' 
r2_score(y_test, preds_cart) #çıktı '0.8328543734007364'
#Model optimizasyonu
from sklearn.model_selection import GridSearchCV
cart_pars = {"min_samples_split":range(2,100),"max_leaf_nodes":range(2,10)}
grid_cart_model = GridSearchCV(cart_model, cart_pars,cv=10)
grid_cart_model.fit(x_egitim,y_egitim)
grid_cart_model.best_params_
best_cart_model = 
DecisionTreeRegressor(max_leaf_nodes=9,min_samples_split=34).fit(x_egitim,y_egitim)
preds_best_cart = best_cart_model.predict(x_test)
mean_squared_error(y_test, preds_best_cart)
r2_score(y_test, preds_best_cart)