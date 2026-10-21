# Cora Citation Network Classification with GCN & ONNX

[![GitHub Repository](https://img.shields.io/badge/GitHub-Arvind208%2FCORA--Project-181717?style=for-the-badge&logo=github)](https://github.com/Arvind208/CORA-Project)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.18%2B-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch_Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pyg.org/)

An end-to-end Graph Neural Network (GNN) system for classifying academic research papers on the **Cora Citation Network** into 7 distinct scientific domains. The project includes a trained **SimpleGCN** exported to **ONNX Runtime** for high-throughput, low-latency inference, served via **FastAPI** with a modern interactive frontend and a **Streamlit** explorer dashboard.

---

## 🌟 Key Features

- **Pre-trained ONNX GCN Model:** 2-layer Graph Convolutional Network exported to ONNX (Opset 18) with dynamic node and edge axes for lightning-fast CPU inference.
- **FastAPI Backend:** Production-ready asynchronous REST API supporting both real Cora dataset queries and custom user-defined citation subgraphs.
- **Interactive Web Interface:** Glassmorphic UI featuring a live HTML5 Canvas local citation graph visualizer, probability distributions, raw logits explorer, and real-time latency benchmarking.
- **Streamlit Dashboard:** Alternative analytical explorer for topic distribution analysis, sample exploration, and remote API health monitoring.
- **Full Cora Dataset Integration:** 2,708 scientific papers, 10,556 citation edges, and 1,433-dimensional bag-of-words features.

---

## 🏛️ Classification Topics

The model classifies papers into seven fundamental computer science categories:

| ID | Class Label | Description |
|:--:|:------------|:------------|
| `0` | **Case_Based** | Case-based reasoning and memory systems |
| `1` | **Genetic_Algorithms** | Evolutionary algorithms and genetic programming |
| `2` | **Neural_Networks** | Deep learning, connectionist models, and ANN architectures |
| `3` | **Probabilistic_Methods** | Bayesian networks, graphical models, and Markov processes |
| `4` | **Reinforcement_Learning** | Policy iteration, Q-learning, and MDPs |
| `5` | **Rule_Learning** | Inductive logic programming and rule extraction |
| `6` | **Theory** | Computational complexity and algorithmic theory |

---

## 📁 Repository Structure

```text
CORA-Project/
├── data/
│   └── Planetoid/
│       └── Cora/
│           ├── processed/        # PyG processed graph tensors (data.pt)
│           └── raw/              # Cora citation dataset raw files
├── static/
│   ├── app.js                    # Web UI client-side graph visualizer & API logic
│   ├── index.css                 # Glassmorphic responsive styling
│   └── index.html                # Interactive dashboard
├── cora_citation_network_classification_(1).ipynb  # Training & ONNX export notebook
├── main.py                       # FastAPI application & ONNX runtime inference
├── requirements.txt              # Project dependencies
├── simple_gcn_cora.onnx          # Exported ONNX model graph
├── simple_gcn_cora.onnx.data     # Exported model weights
└── streamlit_app.py              # Streamlit analytical explorer
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Arvind208/CORA-Project.git
cd CORA-Project
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Optional) If you want to use the Streamlit interface:*
```bash
pip install streamlit pandas requests
```

### 4. Run the FastAPI Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive Web App:** Open [http://localhost:8000](http://localhost:8000) in your browser.
- **Interactive Swagger API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. (Optional) Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/` | Serves the web dashboard (`static/index.html`) |
| `GET` | `/health` | Service health status and available ONNX execution providers |
| `GET` | `/info` | ONNX model schema, feature dimensions, and class mappings |
| `POST` | `/predict/cora_node` | Runs GCN inference for node IDs in the Cora dataset |
| `POST` | `/predict` | Runs GCN inference on user-defined feature vectors and edge indices |

### Example Request (`POST /predict/cora_node`)

```json
{
  "node_indices": [0, 42, 100]
}
```

### Example Response

```json
{
  "num_nodes": 2708,
  "num_edges": 10556,
  "predictions": [
    {
      "node_index": 0,
      "predicted_class_id": 2,
      "predicted_class_name": "Neural_Networks",
      "probabilites": [0.0012, 0.0003, 0.9854, 0.0041, 0.0062, 0.0018, 0.0010],
      "logits": [-2.11, -3.54, 4.82, -0.92, -0.51, -1.73, -2.31]
    }
  ]
}
```

---

## 👤 Author

Developed and maintained by **[Arvind208](https://github.com/Arvind208)**.

- **GitHub:** [@Arvind208](https://github.com/Arvind208)
- **Repository:** [https://github.com/Arvind208/CORA-Project](https://github.com/Arvind208/CORA-Project)

---

## 📄 License

This project is licensed under the MIT License - see the repository for details.
