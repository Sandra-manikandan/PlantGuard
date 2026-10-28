# Plant Disease Detection System (Streamlit + MobileNetV2/EfficientNetB0)

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## 1. Get the dataset
Download PlantVillage from Kaggle (https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
and use the `color` folder (38 class sub-folders).

## 2. Train (GPU recommended – Google Colab is free)
```bash
python train.py --data_dir "PlantVillage/color" --model mobilenetv2
# or
python train.py --data_dir "PlantVillage/color" --model efficientnetb0
```
Outputs in `model/`: `plant_disease_model.keras`, `class_names.json`,
`training_curves.png`, `confusion_matrix.png`, `classification_report.txt`.

## 3. Run the website
```bash
streamlit run app.py
```
Opens at http://localhost:8501

## 4. Deploy free (Streamlit Community Cloud)
Push this folder (including `model/`) to GitHub -> https://share.streamlit.io -> New app -> select `app.py`.
