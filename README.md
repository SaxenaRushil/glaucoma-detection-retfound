\# 🧠 Glaucoma Detection using RETFound



\## 🚀 Overview



This project implements a deep learning pipeline for \*\*glaucoma detection from retinal fundus images\*\* using a RETFound-based model.



The system is designed to:



\* Handle multiple datasets

\* Work across different environments (Kaggle + Local)

\* Provide a structured and reproducible workflow



\---



\## 🧠 Features



\* ✔️ RETFound-based model for retinal analysis

\* ✔️ Multi-dataset support (e.g., AIROGS, SMDG)

\* ✔️ Dynamic dataset path handling (Kaggle + Local)

\* ✔️ Data preprocessing and transformation pipeline

\* ✔️ Model evaluation and performance tracking



\---



\## 🛠️ Tech Stack



\* Python

\* PyTorch

\* NumPy, Pandas

\* OpenCV

\* Matplotlib

\* Scikit-learn



\---



\## 📂 Project Structure



```

Glaucoma-RETFound/

│── Glaucoma\_RETfound.ipynb

│── Glaucoma\_RETfound\_fixed.py

│── requirements.txt

│── .gitignore

│── README.md

```



\---



\## ⚙️ Dataset Setup



Datasets are \*\*not included\*\* due to size.



\### 👉 Update dataset path in code:



```python

DATA\_PATH = "data/"  # Local system

```



or (Kaggle auto-detected):



```

/kaggle/input/your-dataset

```



Place your datasets like:



```

data/

&#x20;├── airogs-dataset/

&#x20;├── smdg/

```



\---



\## ▶️ How to Run



\### 1️⃣ Install dependencies



```

pip install -r requirements.txt

```



\### 2️⃣ Run Notebook



```

jupyter notebook Glaucoma\_RETfound.ipynb

```



OR



\### 3️⃣ Run Python file



```

python Glaucoma\_RETfound\_fixed.py

```



\---



\## 📊 Results



\* Model trained and evaluated on retinal datasets

\* Demonstrates ability to generalize across datasets

\* Suitable for medical image classification tasks



\* 0.96 AUC\*



\---



\## 🔮 Future Improvements



\* 🔹 Model deployment (Flask / Streamlit)

\* 🔹 Explainability (Grad-CAM)

\* 🔹 Performance optimization

\* 🔹 Real-time inference



\---



\## ⚠️ Notes



\* Large datasets require high RAM

\* GPU recommended for training

\* Originally developed on Kaggle environment



\---



\## 👤 Author



Your Name

M.Tech Data Science



\---



\## ⭐ Acknowledgment



Inspired by research on RETFound for generalizable retinal disease detection.



