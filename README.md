# 🚗 Parking Space Detection — YOLOv11

[ 🇬🇧 English Version ](#english) | [ 🇷🇺 Русская версия ](#russian)

---

## 🇬🇧 English Version <a name="english"></a>

**Real-time Parking Space Detection System based on YOLOv11.**

> 🎓 **Project Context:** This project was developed as part of an academic program at Beihang University (BUAA), Beijing, China, in May–June 2026.

### 📌 About the Project

The project addresses a computer vision task of determining the status of parking spaces (occupied/empty) in images and video streams. A key feature of this solution is the implementation of a **two-tier confidence thresholding system** (*Certain* / *Uncertain*). This allows the system not only to detect objects but also to evaluate the reliability of the model's predictions, reducing false positives in challenging lighting conditions or camera angles.

### ✨ Key Features

- 🖼️ **Image Analysis**: Upload and process parking lot photos with bounding box visualization and detailed statistics.
- 🎬 **Video Processing**: Frame-by-frame video analysis with adjustable frame skipping for faster processing and time-based occupancy graphing.
- 📊 **Two-Tier Confidence Classification**:
  - 🟢 **Certain (High confidence)**: Reliable detection (≥ 0.50 for *occupied*, ≥ 0.35 for *empty*).
  - 🟡 **Uncertain (Low confidence)**: Flagged as potential detection with a warning (≥ 0.30 for *occupied*, ≥ 0.20 for *empty*).
- 📈 **Detailed Analytics**: Total space count, occupancy percentage, tabular output of all detections with confidence scores, and video timeline charts.
- 🚀 **Model Export**: Built-in support for exporting model weights to ONNX format for optimized inference and benchmarking.

### 🛠 Tech Stack

- **Language**: Python 3.9+
- **Model**: YOLOv11n (Ultralytics)
- **Interface**: Streamlit
- **Data Processing**: OpenCV, NumPy, Pandas, Pillow (PIL)
- **Optimization**: ONNX Runtime

### 📂 Repository Structure

```text
├── app/                      # Main Streamlit application
│   ├── app.py                # Full-featured interface with analytics and video support
│   ├── easy_app.py           # Simplified version for quick testing
│   ├── export_onnx.py        # Script to export the model to .onnx (imgsz=1280)
│   ├── test_onnx.py          # Script for ONNX model testing and FPS benchmarking
│   ├── best.pt               # Trained PyTorch model weights
│   ├── best.onnx             # Model weights in ONNX format
│   └── test1_*.jpg           # Test images
├── parking_yolo11/           # Model training artifacts and results
│   ├── args.yaml             # Training hyperparameters
│   ├── results.csv           # Training metrics (loss, mAP, etc.)
│   ├── *.png                 # Plots (F1, PR curves, confusion matrix)
│   └── train/val_*.jpg       # Batch visualizations during training and validation
├── proposal/                 # Academic and project documentation
│   ├── Project_Proposal_*.docx # Initial project proposal
│   ├── LJ2606105_final_report.docx # Final project report
│   └── *.pptx / *.pdf        # Presentation slides and speech script
├── weights/                  # Additional/base weights (e.g., yolov8n.pt)
├── yolo_mini.zip             # Archive with a reduced dataset for testing
└── README.md                 # This file
```

### 🚀 Installation and Running

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShadrinDenisBB/parking-detection_YoloV11.git
   cd parking-detection_YoloV11
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   pip install ultralytics streamlit opencv-python numpy pandas Pillow onnxruntime
   ```

3. **Run the application**:
   ```bash
   # Full version with analytics, statistics, and video support
   streamlit run app/app.py
   
   # Or simplified version for quick detection checks
   streamlit run app/easy_app.py
   ```

4. Open the URL provided in the terminal (usually `http://localhost:8501`) in your browser.

### 🧠 Model Training and Export

The model was fine-tuned on a custom parking space dataset with a resolution of `1280x1280`. All training artifacts are saved in the `parking_yolo11/` directory.

To retrain or experiment with hyperparameters:
```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolo11n.pt')

# Train on custom data
model.train(data='path/to/dataset.yaml', epochs=100, imgsz=1280, name='parking_yolo11')
```

#### ONNX Export
To accelerate inference (e.g., for edge device deployment), the model can be exported to ONNX format:
```bash
python app/export_onnx.py
python app/test_onnx.py  # Run inference speed benchmark (FPS)
```

### 📚 Documentation

The `proposal/` folder contains all materials related to the academic reporting of this project:
- **Project Proposal**: Initial problem statement and work plan.
- **Final Report**: Detailed report on the work done, methodology, and results.
- **Presentation**: Slides and script for the project defense.

### 👤 Author

**Denis Shadrin**  
Academic Project, Beihang University (BUAA), Beijing, China (May–June 2026)  
🔗 [GitHub Profile](https://github.com/ShadrinDenisBB)

---
> *Disclaimer: This project was created for educational purposes. For commercial use, fine-tuning on a representative dataset matching the target operating conditions (varying weather, time of day, and camera angles) is recommended.*

<br>
<br>
<br>

---

## 🇷🇺 Русская версия <a name="russian"></a>

**Система детекции свободных и занятых парковочных мест в реальном времени на базе YOLOv11.**

> 🎓 **Контекст проекта:** Данный проект был разработан в рамках учебной программы в Пекинском университете аэронавтики и астронавтики (Beihang University, BUAA), Китай, в период май–июнь 2026 года.

### 📌 О проекте

Проект решает задачу компьютерного зрения по определению статуса парковочных мест (свободно/занято) на изображениях и видеопотоке. Особенностью решения является использование **двухуровневой системы порогов уверенности** (*Certain* / *Uncertain*), что позволяет не только детектировать объекты, но и оценивать надежность предсказания модели, снижая количество ложных срабатываний в сложных условиях освещения или ракурса.

### ✨ Основные возможности

- 🖼️ **Анализ изображений**: Загрузка и обработка фото парковок с визуализацией bounding box и детальной статистикой.
- 🎬 **Обработка видео**: Покадровый анализ видеороликов с возможностью пропуска кадров (frame skip) для ускорения и построением графиков заполненности по времени.
- 📊 **Двухуровневая классификация уверенности**:
  - 🟢 **Certain (Высокая уверенность)**: Надежное детектирование (≥ 0.50 для *occupied*, ≥ 0.35 для *empty*).
  - 🟡 **Uncertain (Низкая уверенность)**: Помечается как потенциальное обнаружение с предупреждением (≥ 0.30 для *occupied*, ≥ 0.20 для *empty*).
- 📈 **Детальная аналитика**: Подсчет общего числа мест, процента заполненности, табличный вывод всех детекций с их confidence score и графики для видео.
- 🚀 **Экспорт модели**: Встроенная поддержка экспорта весов в формат ONNX для оптимизированного инференса и бенчмаркинга.

### 🛠 Стек технологий

- **Язык**: Python 3.9+
- **Модель**: YOLOv11n (Ultralytics)
- **Интерфейс**: Streamlit
- **Обработка данных**: OpenCV, NumPy, Pandas, Pillow (PIL)
- **Оптимизация**: ONNX Runtime

### 📂 Структура репозитория

```text
├── app/                      # Основное Streamlit-приложение
│   ├── app.py                # Полнофункциональный интерфейс с аналитикой и видео
│   ├── easy_app.py           # Упрощенная версия для быстрого тестирования
│   ├── export_onnx.py        # Скрипт для экспорта модели в .onnx (imgsz=1280)
│   ├── test_onnx.py          # Скрипт для проверки и бенчмаркинга ONNX-модели
│   ├── best.pt               # Веса обученной модели PyTorch
│   ├── best.onnx             # Веса модели в формате ONNX
│   └── test1_*.jpg           # Тестовые изображения
├── parking_yolo11/           # Артефакты и результаты обучения модели
│   ├── args.yaml             # Гиперпараметры обучения
│   ├── results.csv           # Метрики обучения (loss, mAP и т.д.)
│   ├── *.png                 # Графики (F1, PR-кривые, матрица ошибок)
│   └── train/val_*.jpg       # Визуализация batch-ей на этапах обучения и валидации
├── proposal/                 # Учебная и отчетная документация
│   ├── Project_Proposal_*.docx # Первоначальное предложение проекта
│   ├── LJ2606105_final_report.docx # Итоговый отчет
│   └── *.pptx / *.pdf        # Презентационные материалы и скрипт выступления
├── weights/                  # Дополнительные/базовые веса (например, yolov8n.pt)
├── yolo_mini.zip             # Архив с уменьшенным набором данных для тестов
└── README.md                 # Этот файл
```

### 🚀 Установка и запуск

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/ShadrinDenisBB/parking-detection_YoloV11.git
   cd parking-detection_YoloV11
   ```

2. **Создайте виртуальное окружение и установите зависимости**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Windows: venv\Scripts\activate
   pip install ultralytics streamlit opencv-python numpy pandas Pillow onnxruntime
   ```

3. **Запустите приложение**:
   ```bash
   # Полная версия с аналитикой, статистикой и поддержкой видео
   streamlit run app/app.py
   
   # Или упрощенная версия для быстрой проверки детекции
   streamlit run app/easy_app.py
   ```

4. Откройте в браузере адрес, указанный в терминале (обычно `http://localhost:8501`).

### 🧠 Обучение и экспорт модели

Модель была дообучена на кастомном датасете парковочных мест с разрешением `1280x1280`. Все артефакты обучения сохранены в директории `parking_yolo11/`.

Для повторного обучения или экспериментов с гиперпараметрами:
```python
from ultralytics import YOLO

# Загрузка базовой модели
model = YOLO('yolo11n.pt')

# Обучение на собственных данных
model.train(data='path/to/dataset.yaml', epochs=100, imgsz=1280, name='parking_yolo11')
```

#### Экспорт в ONNX
Для ускорения инференса (например, для деплоя на edge-устройствах) модель может быть экспортирована в формат ONNX:
```bash
python app/export_onnx.py
python app/test_onnx.py  # Запуск бенчмарка скорости инференса (FPS)
```

### 📚 Документация

В папке `proposal/` находятся все материалы, связанные с академической отчетностью по проекту:
- **Project Proposal** — изначальная постановка задачи и план работ.
- **Final Report** — подробный отчет о проделанной работе, методологии и результатах.
- **Presentation** — слайды и скрипт для защиты проекта.

### 👤 Автор

**Denis Shadrin**  
Учебный проект, Beihang University (BUAA), Пекин, Китай (Май–Июнь 2026)  
🔗 [GitHub Profile](https://github.com/ShadrinDenisBB)

---
> *Проект создан в учебных целях. Для коммерческого использования рекомендуется дообучение на репрезентативном датасете, соответствующем целевым условиям эксплуатации (разные погодные условия, время суток и ракурсы камер).*
