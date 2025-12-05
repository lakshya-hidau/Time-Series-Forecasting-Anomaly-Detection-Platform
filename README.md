# Enterprise-Grade Time-Series Forecasting & Anomaly Detection Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready pipeline for multivariate time-series forecasting and real-time anomaly detection, built with modern deep learning architectures and full MLOps integration.

---

## 🎯 Project Overview

This system provides:

- **Multivariate time-series forecasting** with short & long horizon predictions
- **Real-time anomaly detection** in streaming data
- **End-to-end MLOps pipeline** (feature store, model registry, monitoring)
- **Production-ready deployment** with API, dashboard, and monitoring
- **Demo-ready setup** for technical interviews and presentations

---

## ✨ Key Features

- **Forecasting Models**: Classical (SARIMA), ML (XGBoost), and Deep Learning (LSTM, N-BEATS, Transformer-TS)
- **Anomaly Detection**: Residual-based, Autoencoder/VAE, Isolation Forest
- **MLOps Integration**: MLflow tracking, Feast feature store, experiment management
- **Production Serving**: FastAPI endpoints with Docker containerization
- **Real-time Monitoring**: React dashboard with Grafana integration
- **Scalable Architecture**: Kubernetes-ready deployment

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                             │
│  Historical Data │ Real-time Streams │ External APIs             │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
│  Data Validation │ Preprocessing │ Feature Engineering          │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FEATURE STORE (Feast)                       │
│  Feature Registry │ Online Store │ Offline Store                │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING LAYER                         │
│  Classical Models │ ML Models │ Deep Learning Models            │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│               MODEL REGISTRY & VERSIONING (MLflow)               │
│  Experiment Tracking │ Model Versioning │ A/B Testing           │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING LAYER (FastAPI)                       │
│  Batch Predictions │ Real-time Inference │ Anomaly Detection    │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MONITORING & VISUALIZATION                       │
│  Grafana Dashboards │ Alerting │ Model Performance Tracking     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Data & Modelling
- Python, pandas, NumPy, scikit-learn, statsmodels
- PyTorch, PyTorch Lightning, TensorFlow (optional)
- darts or sktime for time-series modelling
- N-BEATS, Transformer-TS (Informer/Autoformer), Prophet

### Anomaly Detection
- PyOD, PyTorch (autoencoders/VAE)
- IsolationForest, LSTM-AD

### MLOps & Infrastructure
- MLflow (tracking & model registry)
- Feast (feature store)
- Airflow or Prefect (orchestration)
- Docker, Kubernetes (minikube/GKE/EKS)
- Prometheus + Grafana (monitoring)

### Serving & UI
- FastAPI, Uvicorn
- React/Streamlit/Plotly Dash for dashboard
- Redis or Kafka for streaming

### Storage
- PostgreSQL or ClickHouse for time-series data
- S3 for artifacts

---

## 📁 Repository Structure

```
/time-series-project
├── data/                    # Raw and processed sample data
├── notebooks/               # EDA & experiments notebooks
├── src/
│   ├── ingestion/          # Data ingestion modules
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Model implementations
│   ├── training/           # Training scripts
│   ├── serving/            # FastAPI serving layer
│   ├── monitoring/         # Monitoring utilities
│   └── utils/              # Helper functions
├── deployments/
│   ├── docker/             # Docker configurations
│   └── kubernetes/         # K8s manifests (optional)
├── dashboard/              # React/Streamlit dashboard
├── tests/                  # Unit and integration tests
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
├── ROADMAP.md             # 12-week development plan
└── README.md              # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/lakshya-hidau/time-series-forecasting-anomaly-detection-platform.git
cd time-series-forecasting-anomaly-detection-platform
```

2. **Set up Python environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run with Docker (recommended)**

```bash
# Build and start all services
docker-compose up --build

# Or run individual components
docker-compose up api
docker-compose up mlflow
```

4. **Access the services**

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Dashboard**: http://localhost:8500 (if using Streamlit)

---

## 📊 Usage Examples

### 1. Train a Forecasting Model

```python
from src.training.trainer import ModelTrainer
from src.models.nbeats import NBEATSModel

trainer = ModelTrainer()
model = NBEATSModel()
trainer.train(model, data_path='data/train.csv')
```

### 2. Make Predictions via API

```bash
# Batch forecasting
curl -X POST "http://localhost:8000/forecast/batch" \
  -H "Content-Type: application/json" \
  -d '{"model": "nbeats", "horizon": 7, "data": [...]}'

# Real-time anomaly detection
curl -X POST "http://localhost:8000/anomaly/detect" \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-01T00:00:00", "value": 123.45, "features": [...]}'
```

### 3. Monitor with Dashboard

```bash
# Start the dashboard
cd dashboard
streamlit run app.py  # or npm start for React
```

---

## 📈 Evaluation Metrics

### Forecasting
- **Point Forecast**: MAE, RMSE, MAPE (per horizon)
- **Probabilistic**: CRPS, PICP, Pinball loss
- **Validation**: Rolling walk-forward validation

### Anomaly Detection
- Precision, Recall, F1-Score
- Detection delay
- Benchmark on NAB/Yahoo S5 datasets

---

## 🗺️ Development Roadmap

See detailed 12-week implementation plan in [ROADMAP.md](ROADMAP.md):

- **Weeks 1-3**: EDA, baselines, feature engineering
- **Weeks 4-6**: ML & deep learning models
- **Weeks 7-8**: Anomaly detection & explainability
- **Weeks 9-10**: MLOps & serving layer
- **Weeks 11-12**: Dashboard, testing, deployment

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Generate coverage report
pytest --cov=src tests/
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📚 Resources & References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Time Series](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [N-BEATS Paper](https://arxiv.org/abs/1905.10437)
- [NAB Benchmark](https://github.com/numenta/NAB)

---

## 🎥 Demo

[Link to demo video] - Coming soon!

---

## 📧 Contact

For questions or feedback, please open an issue or contact [lakshyahidau2005@gmail.com](mailto:lakshyahidau2005@gmail.com).

---

**Built with ❤️ for the ML/MLOps community**

*Note: This project follows the 12-week roadmap outlined in "Time-series Forecasting + Anomaly Detection — Roadmap & Architecture.pdf"*
