# Mesafe Ölçer

Telefonda canlı kamerayla, iki fotoğraftan üçgenleme ile gövde–dal birleşimine mesafe ölçen test uygulaması.

- **Uzak fotoğraf:** kare, kamera konumu ve birleşim noktası saklanır.
- **Yakın fotoğraf:** sonraki her kare; nokta ORB ile eşleştirilir, iki görüşle üçgenlenir.
- Kamera konumu testte kâğıttaki ChArUco tahtasından gelir (robotta kol kontrolcüsü verir).
  Gerçek mesafe de tahtadan hesaplanır; tahmin onu kullanmaz.
- Bütün hesap telefonda yapılır (OpenCV.js). Yapay zekâ modeli yok.

## Kullanım
1. `test_kagidi.pdf` dosyasını %100 ölçekle yazdır, bir karenin kenarını ölç (tasarım 22 mm). Ayarlar'a gir.
2. Kamera kalibrasyonu: bir kez, ölçümde kullanılacak yönde (yatay önerilir).
3. 50–70 cm'den "Uzak fotoğrafı çek", sonra yaklaşırken biraz yana kay. Ekranda tahmin, gerçek ve hata canlı görünür.
4. "Yakın fotoğrafı kaydet" ile ölçümü tabloya ekle.

`?sim` adresiyle sanal kamera modu açılır (kamera gerekmez). Bilgisayardaki simülasyonda
56 → 36 cm yaklaşmada ortalama hata 0.16 cm, en kötü 0.31 cm.

`kagit_olustur.py`: test kâğıdını üretir.
