# 📚 Book Recommender

A Dockerized, end-to-end book recommendation system that combines classic recommendation models (trained on Goodreads-10k data) with an LLM-powered layer for more “human” suggestions. It exposes a backend API (FastAPI) and a lightweight web UI (Streamlit) to browse and request recommendations. Caching is done via Redis, and model artifacts are mounted into the containers.

> This repo is meant to be a showcase project: data → model → API → UI → Docker.

---

## 🚀 Features

- **Model-based recommendations**  
  Trained recommendation models stored in `models/` and. Can also be built from data in `data/raw/`.

- **LLM-powered suggestions**  
  The stack is ready to use an `OPEN_AI_KEY` (see `compose.yaml`) so the app can generate enriched / mood-based / natural-language recommendations on top of the base model.

- **API + Web app**  
  Containers for:
  - `api`: serves recommendation endpoints
  - `web`: front-end to consume the API
  - `redis`: for caching recommendations / metadata

- **Reproducible with Docker**  
  One command to run everything via `docker compose`.

- **Notebooks for experimentation**  
  `notebooks/` contains exploratory / training notebooks used to build the models.

---

## 🗂️ Repository Structure

```text
book-recommender/
├── compose.yaml        # Docker compose: api, web, redis
├── src/                # Application source code
│   ├── ...             # (backend / frontend code lives here)
├── models/             # Saved recommender models / artifacts
├── data/
│   └── raw/            # Source dataset(s) for training
├── notebooks/          # Model training / EDA notebooks
├── LICENSE
└── README.md           # (you are here)
```
---

## 🧰 Requirements

You can run this project in two ways.

### 1. With Docker (recommended)

- Docker
- Docker Compose

### 2. Locally (dev mode)

- Python 3.10+
- `pip` / `venv`
- Redis running locally (optional but recommended)

---

## 🔑 Environment Variables

The `compose.yaml` shows two important envs:

- `OPEN_AI_KEY` – API key for LLM features.
- `REDIS_URL` – Redis connection string, e.g. `redis://redis:6379/0`

You can create a `.env` file at the project root (not committed) and put:

```env
OPEN_AI_KEY=sk-...
REDIS_URL=redis://redis:6379/0
```

The compose file will pass these into the containers.

---

## ▶️ Running with Docker

From the project root:

```bash
docker compose up --build -d
```

This will:

1. Build the `api` image from `./src/backend`
2. Build the `web` image from `./src/frontend`
3. Start `redis:7`.
4. Mount `./models` into the API container as read-only so the backend can load pretrained recommenders.

Then open the web container’s exposed port in your browser (check the port in `compose.yaml`).

---

## ▶️ Running Locally (without Docker)

This is useful if you just want to tweak the model or API.

1. **Create and activate a virtualenv**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```

2. **Install deps**

   ```bash
   pip install -r requirements.txt
   ```


3. **Run Redis** (optional but used in Docker)
   ```bash
   docker run -p 6379:6379 redis:7
   ```

4. **Run the backend**
   ```bash
   uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Run the frontend**
   ```bash
   streamlit run src/frontend/app.py
   ```

---

## 🧪 Training / Experimentation

- Open any notebook in `notebooks/` (e.g. in VS Code or Jupyter).
- Explore the dataset from `data/raw/`.
- Train a model (ALS / content-based / hybrid).
- Save the trained model artifact into `models/`.
- Restart the API container (or app) so it picks up the new model.

This mirrors a simple MLOps loop: **notebook → artifact → served model**.

---

## 🔌 API

The backend is intended to expose following endpoints:

- `GET /health` – health check
- `POST /recommend` – get top-N recommendations for a user
- `GET /get_titles` – populates book title metadata

---

## 🛡️ Security Notes

- The compose setup expects your secrets **not** to be committed (root-level `.env` ignored by Git).
- In production (e.g. on EKS) you’d map these to Kubernetes Secrets / env vars instead of committing them.

---

## 📝 Roadmap / Ideas

- Add user-based + item-based similarity views
- Add book cover fetching (Open Library / Google Books)
- Add rating history view for a user
- Persist recommendations in Redis for faster re-queries
- CI pipeline to build and push images

---

## 📄 License

This project is licensed under the **MIT License**. See [`LICENSE`](./LICENSE) for details.

---

## 🙌 Author

**Varun Date**  
ML / DS + full-stack-ish demo of a recommender system.
