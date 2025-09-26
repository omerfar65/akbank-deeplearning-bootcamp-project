# Akbank Derin Öğrenme Bootcamp - Beyin Tümörü Sınıflandırma Projesi

Bu proje, Akbank Derin Öğrenme Bootcamp'i kapsamında gerçekleştirilmiş olup, MR görüntülerinden yararlanarak beyin tümörü türlerini sınıflandıran bir derin öğrenme modelinin geliştirilmesini içermektedir.

## Kaggle Notebook Linki

Projenin tüm kodlarını ve detaylı analizlerini içeren Kaggle Notebook'una aşağıdaki linkten ulaşabilirsiniz:

**[Proje Notebook'u]([https://www.kaggle.com/code/merfarukarl/beyintumorus-n-fland-rmaprojesi])**

## 1. Projenin Amacı

Projenin temel amacı, Evrişimli Sinir Ağları (CNN) kullanarak medikal görüntüler üzerinde bir sınıflandırma problemi çözmektir. Bu kapsamda, veri ön işleme, model oluşturma, model eğitimi, performans değerlendirme ve hiperparametre optimizasyonu gibi derin öğrenme proje adımlarının pratik olarak uygulanması hedeflenmiştir.

## 2. Veri Seti

Projede, Kaggle üzerinde halka açık olarak bulunan **"Brain Tumor MRI Dataset"** kullanılmıştır. Veri seti, 4 farklı sınıfa ayrılmış toplamda 7,000'den fazla MR görüntüsü içermektedir:
* **Glioma Tümörü**
* **Meningioma Tümörü**
* **Hipofiz Tümörü (Pituitary)**
* **Tümör Olmayan (No Tumor)**

## 3. Kullanılan Yöntemler

Proje boyunca aşağıdaki yöntem ve teknolojiler kullanılmıştır:

* **Veri Ön İşleme:**
    * Tüm görseller 150x150 piksel boyutuna getirilmiştir.
    * Piksel değerleri 0-1 aralığında normalize edilmiştir.
    * Modelin genelleme yeteneğini artırmak için **Veri Çoğaltma (Data Augmentation)** teknikleri (döndürme, yakınlaştırma, kaydırma vb.) uygulanmıştır.

* **Modelleme:**
    * Model, esnek bir yapı sunduğu için Keras **Functional API** kullanılarak oluşturulmuştur.
    * Mimaride `Conv2D`, `MaxPooling2D`, `Dropout` ve `Dense` gibi temel CNN katmanları bulunmaktadır.

* **Model Değerlendirme:**
    * Modelin öğrenme süreci, **Accuracy ve Loss grafikleri** ile görselleştirilmiştir.
    * Test verisi üzerindeki performansı detaylı analiz etmek için **Karmaşıklık Matrisi (Confusion Matrix)** ve **Sınıflandırma Raporu (Classification Report)** oluşturulmuştur.
    * Modelin karar verirken görüntünün hangi bölgelerine odaklandığını anlamak için **Grad-CAM** ile ısı haritası görselleştirmesi yapılmıştır.

* **Hiperparametre Optimizasyonu:**
    * Model performansını iyileştirmek amacıyla, `Dropout` oranı üzerinde bir deneme yapılmıştır. Baseline modeldeki `%50`'lik Dropout oranı, ikinci denemede `%30`'a düşürülerek sonuçlar karşılaştırılmıştır.

## 4. Elde Edilen Sonuçlar

Yapılan çalışmalar sonucunda, hiperparametre optimizasyonu ile iyileştirilen model, test verileri üzerinde tatmin edici bir başarı göstermiştir.

* Baseline model (Dropout=0.5), yaklaşık **%84**'lük bir doğrulama doğruluğu (validation accuracy) elde etmiştir.
* Yapılan hiperparametre optimizasyonu denemesinde, `Dropout` oranının **%30`'a düşürülmesiyle **en yüksek doğrulama doğruluğu %86'nın üzerine çıkmış** ve aşırı öğrenme (overfitting) eğiliminde azalma gözlemlenmiştir.
* Grad-CAM analizi, modelin sınıflandırma kararları verirken büyük ölçüde MR görüntülerindeki tümörlü bölgelere odaklandığını göstermiştir.
