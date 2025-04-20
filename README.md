# AeroCastAI 🌪️  
AeroCastAI is a self-learning tornado forecasting system. It pulls in random real-time weather data from across the US, retrains its model every few days, and saves the progress automatically. The goal is to eventually let users input any location and get a live tornado risk prediction.

---

## 🔧 What It Does
- Fetches real weather data from [Open-Meteo](https://open-meteo.com/)
- Automatically logs and stores new training data
- Retrains a Random Forest model with the updated data
- Saves the trained model to be used for predictions
- Scheduled to run on [Render.com](https://render.com/) every 3 days

---

## 💻 Files in This Repo
| File | Purpose |
|------|---------|
| `daily_run.py` | Main retraining script (runs on schedule) |
| `retrain_model.py` | One-time retraining script |
| `live_predictor.py` | (Coming soon) Will predict based on user input |
| `aerocastai_model.pkl` | Saved model used for prediction |
| `daily_log.csv` | Log of all training data |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Tech Stack
- Python 3  
- scikit-learn  
- pandas  
- requests  
- joblib  
- Render (for hosting scheduled jobs)

---

## 🚧 What's Coming Soon
- Ability to enter a city or lat/long and get a tornado risk forecast  
- Smarter model tuning and evaluation  
- Frontend for users to interact with AeroCastAI

---

## 🧠 Why I Built This
I’ve always been curious about weather systems and how machine learning can be used to predict dangerous conditions like tornadoes. This is my take on building something that learns over time and eventually helps real people make better decisions.

---

## 📫 Reach Out
If you have ideas, want to contribute, or just want to connect — feel free to reach out.

---

