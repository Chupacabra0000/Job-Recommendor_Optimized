# AI_Job-Recommendor_Optimized_Version

**Link:** [https://github.com/Chupacabra0000/Job-Recommendor_Optimized/tree/main](https://github.com/Chupacabra0000/Job-Recommendor_Optimized/tree/main)

This project builds an **AI-based job recommender system** that suggests relevant job vacancies based on a user's **resume or search query**. It is implemented using **Python + Streamlit + NLP + embeddings**. ([GitHub][1])

---

# 🧠 Main Idea

The system compares **job descriptions with a user's resume** using **natural language processing and semantic similarity**. It ranks jobs that best match the user's skills and experience.

This type of system is called a **recommender system**, which predicts items (jobs in this case) that a user might be interested in based on data patterns. ([Википедия][2])

---

# ⚙️ Key Features

### 1️⃣ Resume-based job matching

* User uploads a **PDF resume**
* Text is extracted from the resume
* System finds jobs matching those skills

### 2️⃣ Live job data

Jobs are fetched **directly from the HH.ru API** (a job portal). ([GitHub][3])

### 3️⃣ Semantic ranking

Uses **Sentence Transformers (multilingual model)** to compute semantic similarity between:

* resume text
* job descriptions

### 4️⃣ Smart search optimization

Improved version includes:

* **TF-IDF keyword extraction**
* **multi-query job fetching**
* **FAISS vector search** for fast ranking. ([GitHub][1])

### 5️⃣ User system

* Login / Registration
* Multiple resumes
* Favorite jobs

### 6️⃣ Caching system

Vacancies and embeddings are cached locally using **SQLite**, improving speed.

---

# 🧩 Project Structure (Important Files)

| File                 | Purpose                        |
| -------------------- | ------------------------------ |
| `app.py`             | Main Streamlit web application |
| `model.py`           | Recommendation model           |
| `hh_client.py`       | API client to fetch jobs       |
| `build_index.py`     | Build search index for jobs    |
| `embedding_store.py` | Stores embeddings              |
| `db.py`              | SQLite database management     |
| `requirements.txt`   | Python dependencies            |

---

# 🚀 How to Run the Project

1️⃣ Clone repository

```bash
git clone https://github.com/Chupacabra0000/Job-Recommendor_Optimized.git
```

2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

3️⃣ Run the app

```bash
streamlit run app.py
```

Then open the browser UI.

---

# 🧠 Technologies Used

* **Python**
* **Streamlit** (web interface)
* **Sentence Transformers**
* **FAISS** (vector similarity search)
* **TF-IDF**
* **SQLite**
* **HH.ru API**

---

# 📊 System Workflow

```
User uploads resume
        ↓
Resume text extraction
        ↓
TF-IDF keywords generated
        ↓
Jobs fetched from API
        ↓
Embeddings generated
        ↓
Similarity ranking (FAISS)
        ↓
Top job recommendations shown
```

---

💡 **In simple words:**
It’s an **AI job search assistant** that reads your resume and automatically finds the most relevant jobs online.



[1]: https://github.com/Chupacabra0000/Job-Recommendor-updated?utm_source=chatgpt.com "GitHub - Chupacabra0000/Job-Recommendor-updated"
[2]: https://en.wikipedia.org/wiki/Recommender_system?utm_source=chatgpt.com "Recommender system - Wikipedia"
[3]: https://github.com/Chupacabra0000/Job-Recommendor?utm_source=chatgpt.com "GitHub - Chupacabra0000/Job-Recommendor"

