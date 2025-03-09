# 🚀 Ubuntu Node Docker Compose Setup

This repository contains a **Docker Compose configuration** to set up multiple **Ubuntu-based containers** with Python, networking tools, and a simulated network script.

---

## 📌 Features
- Uses **Ubuntu:latest** as the base image.
- Installs **Python3, pip, and required dependencies** (`requests` and `scapy`).
- Runs a **network simulation script** (`network_simulator.py`).
- Creates a **custom Docker network** with a specified **subnet (172.20.0.0/24)**.
- Supports **replication** (default: 3 containers).

---

## 🏗️ Prerequisites
Ensure you have:
- **Docker** installed 
- **Docker Compose** installed
- Sufficient system resources for running multiple containers

---

## 🚀 Usage

### 1️⃣ Start the Containers
```bash
sudo docker-compose up -d  # Runs in detached mode
```

### 2️⃣ Check Running Containers
```bash
sudo docker ps
```

### 3️⃣ Stop and Remove Containers
```bash
sudo docker-compose down
```

---

## ⚙️ Configuration
### **Modify Replica Count**
By default, **3 Ubuntu containers** are deployed. Adjust the **`replicas`** value in `docker-compose.yml` if needed:
```yaml
deploy:
  replicas: 3  # Adjust based on system resources
```

### **Modify Network Configuration**
You can change the **subnet** by modifying:
```yaml
networks:
  ubuntu_hosts_network:
    ipam:
      config:
        - subnet: 172.20.0.0/24  # Change it according to availability
```

---

## 🛠️ Troubleshooting
- **Check container logs**:  
  ```bash
  docker logs <container_id>
  ```
- **Inspect network**:
  ```bash
  docker network inspect ubuntu_hosts_network
  ```
- **Rebuild without cache**:
  ```bash
  docker compose up --build --force-recreate
  ```