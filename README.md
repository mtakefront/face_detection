
#  Kendi Yüzünüzü Algılayan YOLOv8 Modeli

## Proje Tanımı

Bu proje, bir stajyerin kendi yüzünü (Mehmet) diğer insan yüzlerinden ayırt edebilen ve gerçek zamanlı olarak algılayabilen bir nesne algılama modeli geliştirmesini amaçlamaktadır. Projede kişisel fotoğraf verilerinden oluşan özgün bir veri seti toplanmış, etiketlenmiş ve YOLOv8 algoritmasıyla eğitilerek model performansı değerlendirilmiştir.

## Hedefler

- Kendi yüzümden (Mehmet) ve diğer insan yüzlerinden (non_mehmet) oluşan özgün bir veri seti hazırlamak.
- Verileri Roboflow platformunda etiketleyip YOLOv8 formatına dönüştürmek.
- YOLOv8 tabanlı bir yüz algılama modeli eğitmek.
- Modelin doğruluk, kesinlik (precision), duyarlılık (recall) ve mAP değerlerini analiz etmek.
- Gerçek zamanlı kamera görüntüsü üzerinde yüz tespiti yapabilmek.

## Kullanılan Teknolojiler

- **Python**: Proje dili
- **Ultralytics YOLOv8**: Model eğitimi ve tahmini
- **PyTorch**: YOLO’nun arka plan framework’ü
- **Roboflow**: Veri seti yönetimi ve etiketleme
- **OpenCV (cv2)**: Kamera akışı ve görselleştirme

## 📂 Proje Yapısı

```
yazilim_xyz_p1/
├── dataset/
│   ├── train/
│   ├── valid/
│   └── data.yaml
├── outputs/
│   ├── runs_yolo/
│   │   └── face_recognition_two_classes/
│   └── submissions/
├── test_set/
├── README.md
└── Untitled.py
```

## 📊 Veri Toplama ve Hazırlama

### Veri Toplama

- Mehmet'e ait yüz fotoğrafları farklı ışık, açı ve arka planlarda toplandı.
- Ayrıca, çeşitli insanlara ait "non_mehmet" görselleri de veri setine eklendi.

### Etiketleme

- Roboflow Universe kullanıldı.
- `mehmet (class 0)` ve `non_mehmet (class 1)` sınıflarıyla bounding box etiketleme yapıldı.
- Segmentasyon formatı yerine YOLO formatına (class_id x_center y_center width height) dönüştürüldü.

### data.yaml Örneği

```yaml
train: C:/Users/cauff/OneDrive/Desktop/yazilim_xyz_p1/dataset/train/images
val: C:/Users/cauff/OneDrive/Desktop/yazilim_xyz_p1/dataset/valid/images

nc: 2
names: ['mehmet', 'non_mehmet']

roboflow:
  workspace: mtakefront
  project: my-first-project-y0feh
  version: 7
  license: CC BY 4.0
  url: https://universe.roboflow.com/mtakefront/my-first-project-y0feh/dataset/7
```

## 🧠 Model Eğitimi

YOLOv8 mimarisi transfer öğrenme ile `yolov8m.pt` ağırlıkları üzerinden eğitildi. Eğitimde veri büyütme(data augmentation) teknikleri uygulandı.

### Eğitim Ayarları

```python
model.train(
    data = yaml_dir,
    epochs=50,
    imgsz=640,
    workers=4,
    batch=32,
    project=runs_dir,
    name='face_recognition_two_classes',
    patience=50,
    device=0,
    fliplr=0.75,
    flipud=0.0,
    degrees=10.0,
)
```

##  Model Performansı

| Sınıf        | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|--------------|--------|-----------|-----------|--------|--------|-----------|
| **all**      | 25     | 28        | 0.978     | 0.833  | 0.831  | 0.702     |
| **mehmet**   | 25     | 25        | 0.979     | 1.000  | 0.995  | 0.870     |
| **non_mehmet**| 2     | 3         | 0.976     | 0.667  | 0.666  | 0.534     |

### Yorum

- **mehmet** sınıfı için mükemmel sonuçlar elde edildi.
- **non_mehmet** sınıfının recall değeri nispeten düşük; bu sınıfa ait daha fazla ve çeşitli veri gerekmektedir.

##  Kullanım

### Kurulum

```bash
pip install ultralytics opencv-python
```

### Görsel Üzerinde Tahmin

```python
model = YOLO("outputs/runs_yolo/face_recognition_two_classes/weights/best.pt")
results = model.predict(source="test_set/ornek_foto.jpg", conf=0.6, iou=0.7, show=True, save=True)
```

### Gerçek Zamanlı Kamera Akışı

```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model.predict(source=frame, conf=0.6, iou=0.7)
    # Kutuları çizdir...
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## 🛣️ Gelecek Çalışmalar

- `non_mehmet` sınıfı için daha fazla ve dengeli veri toplanması
- YOLOv8 dışında farklı modellerin test edilmesi
- Gerçek zamanlı veri kullanılarak test edilmesi

## Katkıda Bulunma

Bu proje açık kaynaklıdır. Her türlü katkıya, geri bildirime ve öneriye açığız.

## Lisans

Veri seti Roboflow CC BY 4.0 lisansına sahiptir. Proje kodları da uygun açık kaynak lisansları ile paylaşılabilir.
