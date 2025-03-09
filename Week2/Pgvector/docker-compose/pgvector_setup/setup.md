# 🚀 PostgreSQL with pgVector & pgAdmin - Docker Compose Setup

This repository provides a **Docker Compose configuration** to set up a **PostgreSQL database with pgVector** for handling vector-based search, along with **pgAdmin** for database management.

---

## 📌 Features
- **PostgreSQL with pgVector** for storing and querying vector embeddings.
- **pgAdmin 4** for managing the PostgreSQL database via a web interface.
- **Persistent storage** for both PostgreSQL and pgAdmin.
- **Environment variables** for easy configuration.
- **Health checks** to ensure database availability.

---

## 🏗️ Prerequisites
Ensure you have:
- **Docker** installed 
- **Docker Compose** installed

---

## 🚀 Usage

### 1️⃣ Create an `.env` File
Before running the compose file, create a `.env` file with the required credentials:
```bash
POSTGRES_DB=your_database
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=your_pgadmin_password
```

### 2️⃣ Start the Containers
```bash
sudo docker-compose up -d  # Runs in detached mode
```

### 3️⃣ Access pgAdmin
Open a browser and go to:
```
http://localhost:5016
```
Login using the credentials from `.env` file.

### 4️⃣ Stop & Remove Containers
```bash
sudo docker-compose down
```

---

## ⚙️ Configuration
### **Modify Database Storage Path**
The PostgreSQL data is stored locally in `./local_pgdata`. Adjust the path in `docker-compose.yml` if needed:
```yaml
volumes:
  - ./local_pgdata:/var/lib/postgresql/data
```

### **Change pgAdmin Storage**
pgAdmin stores settings in `./pgadmin-data`. Modify it as per requirements:
```yaml
volumes:
  - ./pgadmin-data:/var/lib/pgadmin
```

---

## 🛠️ Troubleshooting
- **Check database logs**:  
  ```bash
  docker logs pgvector_db_container
  ```
- **Check pgAdmin logs**:
  ```bash
  docker logs pgadmin4_container
  ```
- **Restart the database**:
  ```bash
  docker restart pgvector_db_container
  ```