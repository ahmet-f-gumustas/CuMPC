# CuMPC

**CuMPC**, **ROS 2** için CUDA hızlandırmalı bir **Model Öngörülü Kontrol (MPC)** uygulamasıdır.

Amaç, MPC'nin ağır sayısal işlerini — tahmin ufkunun kurulması, maliyet fonksiyonunun
değerlendirilmesi ve optimizasyon probleminin çözülmesi — GPU'ya taşıyarak, kontrolcünün
çok durumlu, uzun ufuklu veya sıkı gerçek zamanlı sistemlerde yüksek hızda çalışmasını sağlamaktır.

## Nasıl Çalışır

MPC her kontrol adımında bir optimizasyon problemi çözer: sistemin davranışını belirli bir
ufuk boyunca tahmin eder, bir maliyet fonksiyonunu en aza indirir ve ilk kontrol girdisini uygular.
Bu döngünün hesaplama maliyeti ufuk uzunluğu ve durum sayısıyla hızla artar. CuMPC, döngünün
paralelleştirilebilir kısımlarını CUDA çekirdeklerine taşır ve kontrolcüyü bir ROS 2 düğümü
olarak sunar.

## Özellikler

- MPC döngüsünün paralelleştirilebilir kısımları için CUDA çekirdekleri (tahmin, maliyet hesabı, çözücü iterasyonları)
- Kontrolcüyü saran ROS 2 düğümü (durumu dinler, kontrol komutları yayınlar)
- Yapılandırılabilir ufuk, model, maliyet ağırlıkları ve kısıtlar

## Gereksinimler

- CUDA Toolkit (CUDA destekli bir GPU ile)
- ROS 2
- C++17 (veya üzeri) derleyici
- CMake

## Durum

Geliştirme aşamasında. API'ler, düğüm adları ve parametreler hâlâ değişebilir.

## Lisans

GNU General Public License v3.0 ile lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakın.
