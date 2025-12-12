# hdg@home - Admin Guide

**Internal documentation for coordinators/operators.**

## 🔐 Authentication

### Dashboard Access
- **URL:** `http://localhost:8002/dashboard/`
- **Default Credentials:**
  - Username: `admin`
  - Password: `hdghome2024`

⚠️ **Change the password in `server/coordinator.py` before production deployment!**

```python
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "your-new-secure-password"
```

### API Key Management

Generate keys for trusted workers:

```bash
# Generate a new API key (requires admin auth)
curl -u admin:hdghome2024 -X POST http://localhost:8002/admin/generate-key

# List all keys
curl -u admin:hdghome2024 http://localhost:8002/admin/list-keys

# Revoke a key (use first 8 chars of key)
curl -u admin:hdghome2024 -X DELETE http://localhost:8002/admin/revoke-key/abc12345
```

## 🚀 Deployment

### Local Testing

```bash
# Start everything (server + 2 workers)
python run_cluster.py

# Or start components separately:
python -m uvicorn server.coordinator:app --port 8002
python client/worker.py  # In separate terminals
```

### Production Checklist

- [ ] Change admin password
- [ ] Set up HTTPS (nginx/Caddy reverse proxy)
- [ ] Configure firewall (allow 8002)
- [ ] Set `HF_HOME` to model storage path
- [ ] Enable API key enforcement (optional)
- [ ] Set up monitoring/logging

### Running with Docker (Future)

```dockerfile
# TODO: Add Dockerfile
```

## 📊 Monitoring

### Dashboard Features
- Active peer count
- Training step counter
- Real-time loss chart
- Connected peer table with layer assignments
- Network topology visualization

### Logs

Worker logs are saved to `worker_<peer_id>.log` in the working directory.

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | None | Status JSON |
| `/join` | GET | None | Public landing page |
| `/dashboard/` | GET | Basic | Admin dashboard |
| `/admin/generate-key` | POST | Basic | Generate worker API key |
| `/admin/list-keys` | GET | Basic | List all API keys |
| `/admin/revoke-key/{id}` | DELETE | Basic | Revoke an API key |
| `/register` | POST | None* | Worker registration |
| `/heartbeat` | POST | None* | Worker heartbeat |

*Can be locked down with API key requirement.

## 🔧 Configuration Options

### Model Selection

Edit `server/coordinator.py`:

```python
MODEL_NAME = "H:/gen_ai/llm/gpt2"  # Local path
# MODEL_NAME = "gpt2"  # HuggingFace ID
```

### Training Parameters

Currently hardcoded in `server/trainer.py` and `client/shard_engine.py`:
- Learning Rate: `1e-4`
- Optimizer: `AdamW`
- Loss: `output.pow(2).mean()` (L2 norm for testing)

## 🐛 Troubleshooting

### Workers not connecting
1. Check firewall allows port 8002
2. Verify `COORDINATOR_URL` in worker
3. Check coordinator logs for registration attempts

### Out of memory
1. Reduce model size or use smaller split
2. Enable CPU offloading (future feature)

### Slow training
1. Check worker VRAM (4GB+ recommended)
2. Reduce batch size
3. Check network latency between peers

## 📁 File Locations

| File | Purpose |
|------|---------|
| `server/coordinator.py` | Main server logic + auth config |
| `server/static/` | Dashboard HTML/CSS/JS |
| `client/worker.py` | Integrated worker client |
| `client/shard_engine.py` | Model execution engine |
| `common/protocol.py` | Shared data models |

## 🔄 Backup & Recovery

Currently, all state is in-memory. Server restart clears:
- Peer registrations
- Assignments
- API keys

**TODO:** Add persistence layer (SQLite/Redis).
