import pandas as pd
import numpy as np
# Görev 1:
# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("persona.csv")
df.info()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df.groupby("SOURCE").agg({"PRICE": "mean"})

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY").agg({"PRICE": "mean"})

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE").agg({"PRICE": "mean"})

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})


# Görev 2:COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df_pt = df.pivot_table(index=["COUNTRY", "SOURCE", "SEX", "AGE"], values="PRICE", aggfunc=np.mean).reset_index()

# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values
# metodunu azalan olacak şekilde PRICE’a göre uygulayınız ve çıktıyı agg_df olarak kaydediniz.
agg_df = df_pt.sort_values("PRICE", ascending=False)


# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df = agg_df.reset_index(drop=True)


# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici şekilde oluşturunuz.
# Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
agg_df["AGE"] = agg_df["AGE"].astype("category")

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=["0_18", "19_23",
                                                                                "24_30", "31_40", "41_70"])


# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek
# customers_level_based değişkenini oluşturmanız gerekmektedir.
agg_df["customers_level_based"] = (agg_df.COUNTRY.astype("str") + "_" + agg_df.SOURCE.astype("str") +
                                   "_" + agg_df.SEX.astype("str") + "_" + agg_df.AGE_CAT.astype("str")).str.upper()

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()


# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız.
# Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

print(agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]}))


# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
print(agg_df[agg_df["customers_level_based"] == "TUR_ANDROID_FEMALE_31_40"])

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
print(agg_df[agg_df["customers_level_based"] == "FRA_IOS_FEMALE_31_40"])
