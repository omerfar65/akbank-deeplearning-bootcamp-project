# Akbank Derin Öğrenme Bootcamp - Beyin Tümörü Sınıflandırma Projesi

Bu proje, Akbank Derin Öğrenme Bootcamp final projesi olarak geliştirilmiştir. Projenin amacı, hastalara ait MR görüntülerinden yararlanarak, derin öğrenme yöntemleriyle beyin tümörü türünü ("glioma", "meningioma", "pituitary" veya "notumor") sınıflandıran bir model oluşturmaktır. Proje, medikal görüntü analizi gibi pratik bir iş probleminin çözümüne odaklanmakta ve bir derin öğrenme projesinin tüm yaşam döngüsünü (veri analizi, modelleme, değerlendirme, yorumlama) kapsamaktadır.

### İçindekiler Tablosu
* [Canlı Kaggle Not Defteri](#-canl%C4%B1-kaggle-not-defteri)
* [Projenin Amacı ve Stratejisi](#projenin-amac%C4%B1-ve-stratejisi)
* [Kullanılan Teknolojiler](#kullan%C4%B1lan-teknolojiler-ve-k%C3%BCt%C3%BCphaneler)
* [Kurulum ve Çalıştırma](#kurulum-ve-%C3%A7al%C4%B1%C5%9Ft%C4%B1rma)
* [Proje Adımları](#proje-ad%C4%B1mlar%C4%B1)
* [Sonuçlar ve Analiz](#sonu%C3%A7lar-ve-analiz)
* [Gelecek Adımlar](#gelecek-ad%C4%B1mlar)

---

### 🚀 Canlı Kaggle Not Defteri

Bu projenin tüm adımlarını, kodlarını ve detaylı analizlerini içeren final notebook'una **[buradan](https://www.kaggle.com/code/merfarukarl/beyintumorus-n-fland-rmaprojesi)** ulaşabilirsiniz.

---

### Projenin Amacı ve Stratejisi

#### Karşılaşılan Problem

Beyin tümörlerinin teşhisi, radyologlar için bile zaman alıcı ve uzmanlık gerektiren bir süreçtir. Farklı tümör türleri, MR görüntülerinde birbirine çok benzer desenler gösterebilir. Bu durum, teşhis sürecinde otomasyon ve yapay zeka destekli karar mekanizmalarına olan ihtiyacı artırmaktadır. Projenin temel problemi, bu görsel olarak karmaşık ve birbirine yakın sınıfları yüksek doğrulukla ayırt edebilen bir model geliştirmektir.

#### Geliştirilen Çözüm: Uçtan Uca Derin Öğrenme Pipeline'ı

Bu probleme çözüm olarak, bir Evrişimli Sinir Ağı (CNN) tabanlı, uçtan uca bir derin öğrenme süreci geliştirilmiştir. Halka açık bir veri seti kullanılarak, aşağıdaki stratejiler benimsenmiştir:
* **Veri Çoğaltma (Data Augmentation):** Modelin genelleme yeteneğini artırmak ve aşırı öğrenmeyi (overfitting) azaltmak için mevcut veri seti, Keras'ın `ImageDataGenerator` kütüphanesi ile döndürme, yakınlaştırma gibi teknikler kullanılarak yapay olarak zenginleştirilmiştir.
* **Model Yorumlanabilirliği (Grad-CAM):** Geliştirilen modelin bir "kara kutu" olmasını engellemek amacıyla, Grad-CAM tekniği kullanılmıştır. Bu sayede modelin bir tümör teşhisi koyarken MR görüntüsünün hangi bölgelerine odaklandığı görselleştirilerek, modelin kararlarına olan güven artırılmıştır.
* **Hiperparametre Optimizasyonu:** Model performansını iyileştirmek için `Dropout` oranı gibi kritik bir hiperparametre üzerinde deneyler yapılmıştır.

---

### Kullanılan Teknolojiler ve Kütüphaneler

* **Programlama Dili:** Python
* **Derin Öğrenme:** TensorFlow, Keras
* **Görüntü İşleme:** OpenCV, Matplotlib
* **Veri Manipülasyonu:** NumPy
* **Geliştirme Ortamı:** Kaggle Notebooks (GPU ile)
* **Sürüm Kontrolü:** Git ve GitHub

---

### Kurulum ve Çalıştırma

Bu projeyi kendi Kaggle ortamınızda çalıştırmak için:
1.  Yukarıda paylaşılan **[Kaggle Notebook](https://www.kaggle.com/code/merfarukarl/beyintumorus-n-fland-rmaprojesi)** linkine gidin.
2.  Notebook'u kendi profilinize kopyalamak için "Copy and Edit" butonuna tıklayın.
3.  Notebook'un "Settings" -> "Accelerator" menüsünden GPU'nun aktif olduğundan emin olun.
4.  "Run" -> "Run All" komutuyla tüm hücreleri çalıştırın.

---

### Proje Adımları

1.  **Veri Seti Hazırlığı:** Kaggle'daki "Brain Tumor MRI Dataset" incelenmiş ve sınıfların dağılımı analiz edilmiştir.
2.  **Veri Ön İşleme:** Keras `ImageDataGenerator` ile görseller 150x150 boyutuna getirilmiş, normalize edilmiş ve veri çoğaltma teknikleri uygulanmıştır. Veri seti %80 eğitim, %20 doğrulama olmak üzere ayrılmıştır.
3.  **Model Mimarisi:** Keras Functional API kullanılarak standart bir Konvolüsyonlu Sinir Ağı (CNN) modeli inşa edilmiştir.
4.  **Modelin Eğitilmesi:** Hazırlanan veri seti ile model, 25 dönem (epoch) boyunca eğitilmiştir.
5.  **Değerlendirme ve Test:** Eğitim sonuçları doğruluk/kayıp grafikleriyle görselleştirilmiş, test seti üzerinde Karmaşıklık Matrisi ve Sınıflandırma Raporu oluşturulmuştur.
6.  **Hiperparametre Optimizasyonu:** `Dropout` oranı %50'den %30'a düşürülerek model performansı üzerindeki etkisi incelenmiştir.

---

### Sonuçlar ve Analiz

Eğitim süreci sonunda elde edilen en iyi modelin (Dropout=0.3) performans grafikleri aşağıdadır:

<img width="1182" height="578" alt="image" src="https://github.com/user-attachments/assets/ba0c4bdb-c7ec-49d5-bfde-c7a4d25cc4d6" />


* **Eğitim Başarımı (Training Accuracy):** Model, eğitim verisetini **~%94**'ün üzerinde bir başarıyla öğrenmiştir.
* **Doğrulama Başarımı (Validation Accuracy):** Model, daha önce görmediği doğrulama verilerinde **~%86**'lık bir başarı göstermiştir.
* İki sonuç arasındaki fark, modelde hala bir miktar **aşırı öğrenme (overfitting)** olduğunu göstermektedir. Ancak yapılan hiperparametre optimizasyonu, bu durumu ilk modele göre iyileştirmiştir.


---

### Gelecek Adımlar

Bu projenin daha da ileriye taşınması için aşağıdaki adımlar atılabilir:
* **Veri Setini Büyütmek:** Daha fazla veya daha çeşitli MR görüntüleri kullanarak modelin genelleme yeteneğini artırmak.
* **Aşırı Öğrenmeyi Engellemek:** `L1/L2 regularizasyon` veya `Erken Durdurma (Early Stopping)` gibi daha gelişmiş teknikler uygulamak.
* **Transfer Learning:** ImageNet gibi büyük veri setleri üzerinde eğitilmiş VGG16, ResNet gibi hazır modelleri kullanarak, daha az veriyle daha yüksek başarı oranları elde etmeyi denemek.
