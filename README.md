Employee Attrition Prediction Service

A production-grade Machine Learning system for predicting employee attrition, detecting anomalies, monitoring data drift, and retraining models automatically.
This platform integrates supervised learning, unsupervised learning, deep learning, NLP, real-time inference APIs, and monitoring dashboards into a single end-to-end ML system.

#### What this system solves

Most attrition models fail in production because:

    Input data changes
    User behavior changes
    New employee profiles appear
    Models silently become wrong

This system is designed to survive that.

It does not just predict attrition.
It detects when it should no longer trust itself.

High level architecture
HR System
    |
    v
FastAPI Inference API (v1)
    |
    v
Preprocessing Pipeline
(OneHotEncoder + StandardScaler)
    |
    v
------------------------------------------------
| Supervised Model (Attrition Probability)     |
| KMeans (Employee Segments)                   |
| Isolation Forest (Statistical Anomalies)     |
| Deep Autoencoder (Behavioral Anomalies)      |
------------------------------------------------
    |
    v
Predictions + Anomaly Flags
    |
    v
SQLite Monitoring Database
    |
    v
Drift Detection + Retraining Engine
    |
    v
Metrics Dashboard (Port 8001)



#### Why this design exists

A single ML model cannot survive production.

So this system uses multiple safety layers.

#### Supervised learning

Predicts:

    Probability an employee will leave

This is the business signal.

#### Unsupervised learning (KMeans)
    Clusters employees into behavioral segments.

This allows:

    Segment-level analysis
    Detection of new employee types

#### Isolation Forest

Detects:

    Statistically rare employee profiles

These represent out-of-distribution inputs that supervised models cannot handle safely.

#### Deep Autoencoder

Learns:

    Normal employee behavior patterns

It detects:

    Novel patterns not seen during training

This protects the supervised model from hallucinating.

NLP feature pipeline

Employee feedback text is converted into embeddings using Sentence Transformers.

This provides:

    Soft behavioral signals
    Semantic understanding
    Human-like signal extraction

#### Drift detection

The system tracks:

    Feature distributions
    Incoming traffic patterns

If drift is detected:

    The model retrains automatically
    New baseline statistics are saved

#### Monitoring dashboard

The live dashboard shows:

    Attrition probability distribution
    Isolation Forest anomaly rate
    Deep autoencoder anomaly rate
    Cluster distribution

This lets operators see whether the model is still observing the same world.

#### API design

The inference API is versioned.

    POST /v1/predict


Why:
Models evolve. Clients must not break.

Why preprocessing is saved with the model

The model does not understand raw data.

It understands:

    Encoded categories
    Scaled numbers

If preprocessing changes, predictions become meaningless.

So preprocessing and the model are stored as one pipeline.

Why inference is stateless

    The API does not store user state.
    Every request is independent.

This allows:
    Horizontal scaling
    Safe retries
    No memory leaks
    Deterministic predictions

#### How to run
Start the inference API:

    uvicorn src.api.main:app --port 8000


Start the dashboard:

    uvicorn src.dashboard:app --port 8001


Make a prediction:

    POST http://127.0.0.1:8000/v1/predict


View monitoring:

    http://127.0.0.1:8001/metrics

Author

Anuj Mundu
MCA Student | AI and ML Engineer | Full Stack Developer

Built and deployed a full production-style Machine Learning system combining supervised learning, unsupervised learning, deep learning, NLP, real-time inference APIs, drift detection, automated retraining, and monitoring dashboards.

GitHub: https://github.com/anujmundu

LinkedIn: https://linkedin.com/in/anujmundu
