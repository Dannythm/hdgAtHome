# hdg@home

**Distributed AI Training for Everyone**

Train large language models across consumer hardware using heterogeneous pipeline parallelism.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-yellow)

## 🚀 What is hdg@home?

hdg@home is a distributed training platform that allows anyone to contribute their GPU power to train AI models. Similar to projects like Folding@home or BOINC, but for machine learning.

**Key Features:**
- 🖥️ **Heterogeneous Pipeline Parallelism** - Split models across different GPUs
- 🌐 **HTTP-based Coordination** - Simple, NAT-friendly communication
- 📊 **Real-time Dashboards** - Monitor training progress and contributions
- 🔒 **Admin Authentication** - Protected coordinator dashboard

## 📋 Requirements

**For Contributors (Workers):**
- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (CPU mode available but slow)
- Stable internet connection

**For Coordinators:**
- Python 3.10+
- Public IP or domain (for workers to connect)

## 🏃 Quick Start

### As a Contributor

```bash
# Download the worker script
curl -O http://your-coordinator:8002/worker.py

# Install dependencies
pip install torch transformers accelerate requests fastapi uvicorn psutil

# Edit COORDINATOR_URL in worker.py, then run
python worker.py
```

Your local dashboard will be available at `http://localhost:8080`

### As a Coordinator

```bash
# Clone the repository
git clone https://github.com/your-username/hdgAtHome.git
cd hdgAtHome

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set model path (optional)
export HF_HOME=/path/to/models

# Start coordinator
python -m uvicorn server.coordinator:app --host 0.0.0.0 --port 8002
```

**Available Endpoints:**
- `/join` - Public landing page for contributors
- `/dashboard/` - Admin dashboard (requires login)
- `/worker.py` - Downloadable worker script

## 📁 Project Structure

```
hdgAtHome/
├── server/
│   ├── coordinator.py    # Central coordination server
│   ├── partitioner.py    # Model sharding logic
│   ├── trainer.py        # Training loop driver
│   └── static/           # Web UI files
├── client/
│   ├── worker.py         # Worker client
│   ├── shard_engine.py   # Model shard execution
│   └── static/           # Worker dashboard
├── common/
│   └── protocol.py       # Shared data models
└── verification/
    └── test_network.py   # E2E tests
```

## 🔧 Configuration

### Coordinator (`server/coordinator.py`)

```python
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "your-secure-password"
MODEL_NAME = "path/to/model"
```

### Worker (`worker.py`)

```python
COORDINATOR_URL = "http://your-coordinator:8002"
```

## 🧪 Development

```bash
# Run tests
python verification/test_network.py

# Start local cluster for testing
python run_cluster.py
```

## 🛡️ Security Notes

- Change default admin password before deployment
- Use HTTPS in production (reverse proxy recommended)
- API key system available for worker authentication
- Consider VPN/private network for sensitive training

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Inspired by:
- [OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo)
- [Folding@home](https://foldingathome.org/)
- [BOINC](https://boinc.berkeley.edu/)

---

**Built for the community, by the community.**
