# Akbank Derin Öğrenme Bootcamp - Beyin Tümörü Sınıflandırma Projesi

Bu proje, Akbank Derin Öğrenme Bootcamp'i kapsamında gerçekleştirilmiş olup, MR görüntülerinden yararlanarak beyin tümörü türlerini sınıflandıran bir derin öğrenme modelinin geliştirilmesini içermektedir.

## [cite_start]1. Projenin Amacı [cite: 9]

[cite_start]Projenin temel amacı, Evrişimli Sinir Ağları (CNN) kullanarak medikal görüntüler üzerinde bir sınıflandırma problemi çözmektir[cite: 3]. [cite_start]Bu kapsamda, veri ön işleme, model oluşturma, model eğitimi, performans değerlendirme ve hiperparametre optimizasyonu gibi derin öğrenme proje adımlarının pratik olarak uygulanması hedeflenmiştir[cite: 3].

## [cite_start]2. Veri Seti [cite: 10]

Projede, Kaggle üzerinde halka açık olarak bulunan **"Brain Tumor MRI Dataset"** kullanılmıştır. Veri seti, 4 farklı sınıfa ayrılmış toplamda 7,000'den fazla MR görüntüsü içermektedir:
* **Glioma Tümörü**
* **Meningioma Tümörü**
* **Hipofiz Tümörü (Pituitary)**
* **Tümör Olmayan (No Tumor)**

## [cite_start]3. Kullanılan Yöntemler [cite: 11]

Proje boyunca aşağıdaki yöntem ve teknolojiler kullanılmıştır:

* **Veri Ön İşleme:**
    * Tüm görseller 150x150 piksel boyutuna getirilmiştir.
    * Piksel değerleri 0-1 aralığında normalize edilmiştir.
    * [cite_start]Modelin genelleme yeteneğini artırmak için **Veri Çoğaltma (Data Augmentation)** teknikleri (döndürme, yakınlaştırma, kaydırma vb.) uygulanmıştır[cite: 21].

* **Modelleme:**
    * Model, esnek bir yapı sunduğu için Keras **Functional API** kullanılarak oluşturulmuştur.
    * [cite_start]Mimaride `Conv2D`, `MaxPooling2D`, `Dropout` ve `Dense` gibi temel CNN katmanları bulunmaktadır[cite: 30, 31, 32, 33].

* **Model Değerlendirme:**
    * [cite_start]Modelin öğrenme süreci, **Accuracy ve Loss grafikleri** ile görselleştirilmiştir[cite: 37].
    * [cite_start]Test verisi üzerindeki performansı detaylı analiz etmek için **Karmaşıklık Matrisi (Confusion Matrix)** ve **Sınıflandırma Raporu (Classification Report)** oluşturulmuştur[cite: 38].
    * [cite_start]Modelin karar verirken görüntünün hangi bölgelerine odaklandığını anlamak için **Grad-CAM** ile ısı haritası görselleştirmesi yapılmıştır[cite: 39].

* **Hiperparametre Optimizasyonu:**
    * Model performansını iyileştirmek amacıyla, `Dropout` oranı üzerinde bir deneme yapılmıştır. Baseline modeldeki `%50`'lik Dropout oranı, ikinci denemede `%30`'a düşürülerek sonuçlar karşılaştırılmıştır.

## [cite_start]4. Elde Edilen Sonuçlar [cite: 12]

Yapılan çalışmalar sonucunda, hiperparametre optimizasyonu ile iyileştirilen model, test verileri üzerinde tatmin edici bir başarı göstermiştir.

* Baseline model (Dropout=0.5), yaklaşık **%84**'lük bir doğrulama doğruluğu (validation accuracy) elde etmiştir.
* Yapılan hiperparametre optimizasyonu denemesinde, `Dropout` oranının **%30**'a düşürülmesiyle **en yüksek doğrulama doğruluğu %86'nın üzerine çıkmış** ve aşırı öğrenme (overfitting) eğiliminde azalma gözlemlenmiştir.
* Grad-CAM analizi, modelin sınıflandırma kararları verirken büyük ölçüde MR görüntülerindeki tümörlü bölgelere odaklandığını göstermiştir.

## [cite_start]5. Kaggle Notebook Linki [cite: 13]

Projenin tüm kodlarını ve detaylı analizlerini içeren Kaggle Notebook'una aşağıdaki linkten ulaşabilirsiniz:

**[Proje Notebook'u](BURAYA_KENDİ_KAGGLE_NOTEBOOK_LİNKİNİZİ_YAPISTIRIN)**
