# AI Job Recommender

An AI-powered job recommendation system that analyzes a candidate's resume and finds relevant vacancies using **NLP, TF-IDF, Sentence Transformers, and FAISS**.

The application allows users to upload a PDF resume, extract relevant skills and keywords, search vacancies from **hh.ru**, and rank them according to semantic similarity between the resume and job descriptions.

---

## Features

* 📄 Upload and analyze PDF resumes
* 🔎 Extract relevant technical terms using **TF-IDF**
* 💼 Search vacancies through the **hh.ru API**
* 🤖 Semantic vacancy matching using **Sentence Transformers**
* ⚡ Fast similarity search using **FAISS**
* 📊 Rank vacancies according to resume/job similarity
* 💡 Explain why a vacancy matches the resume
* 🌍 Select regions and cities for vacancy searches
* ⭐ Save favorite vacancies
* 🔐 User authentication and sessions
* 📑 Support for multiple resumes
* 🔍 Save and reuse search queries
* 💾 SQLite database for application data
* 🧪 Automated tests with **pytest**
* ♻️ Vacancy and embedding caching
* 🔄 Incremental and periodic rebuilding of the global FAISS index

---

## How It Works

The main recommendation pipeline can be summarized as:

```text
PDF Resume
    ↓
Text Extraction
    ↓
TF-IDF Keyword Extraction
    ↓
hh.ru Vacancy Search
    ↓
Text Embedding
    ↓
FAISS Similarity Search
    ↓
Vacancy Ranking
    ↓
Recommended Jobs
```

The system combines **keyword-based analysis** with **semantic similarity** to identify vacancies that are relevant to the candidate's skills and experience.

---

## Architecture

### `app.py`

The main Streamlit application.

It handles:

* User authentication
* Resume uploading
* Resume processing
* Region/city selection
* Vacancy searching
* Recommendation display
* Favorites
* Saved searches
* Interaction with the global vacancy index

---

### `model.py`

Contains the main `JobRecommendationSystem`.

Responsibilities include:

* Loading vacancy datasets
* Loading/building vacancy embeddings
* Encoding resumes
* Calculating similarity scores
* Ranking vacancies
* Generating keyword-based explanations

The system uses normalized embeddings and calculates similarity between resume and vacancy vectors.

The default model configured in `JobRecommendationSystem` is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

### `hh_client.py`

Client for interacting with the **hh.ru API**.

It provides functionality for:

* Vacancy searching
* Vacancy retrieval
* Vacancy details
* Pagination
* Region filtering
* Search period and ordering
* Retry handling

The client includes retry and exponential backoff logic for temporary API errors such as:

```text
429
500
502
503
504
```

---

### `hh_areas.py`

Handles the hh.ru geographical hierarchy.

It is used to:

* Retrieve regions
* Retrieve cities
* Resolve geographical identifiers
* Cache geographical information

---

### `tfidf_terms.py`

Provides TF-IDF-based keyword extraction.

The tokenizer supports:

* English
* Russian
* Technical terms
* Programming languages
* Frameworks and technologies

Examples of technical tokens that can be preserved include:

```text
C++
C#
.NET
Python
Java
SQL
```

Russian and English stop words are filtered before calculating TF-IDF scores.

---

### `vector_store.py`

Responsible for persistent storage of vacancy embeddings.

It uses NumPy-based storage and metadata to avoid unnecessarily recalculating embeddings.

Embeddings can be stored and loaded between application runs.

---

### `global_faiss_index.py`

Provides the global FAISS index used for efficient similarity search across vacancies.

FAISS allows the system to search large numbers of vectors much faster than comparing every vacancy individually.

---

### `global_index_manager.py`

Manages the global vacancy index.

It supports:

* Adding new vacancies
* Updating existing data
* Incremental index updates
* Full index rebuilding
* Vacancy-to-vector mapping
* Periodic index maintenance

The application can perform incremental updates while also periodically rebuilding the complete index.

---

### `db.py`

Provides SQLite-based persistence.

The database is used for application data such as:

* Users
* Sessions
* Resumes
* Favorite vacancies
* Saved searches
* Saved search results
* Global vacancy information
* Index state

---

## Recommendation Algorithm

The recommendation system uses vector embeddings to represent resumes and vacancies.

For example:

```text
Resume → Embedding Vector
Vacancy → Embedding Vector
             ↓
      Similarity Calculation
             ↓
       Relevance Score
```

Because the vectors are normalized, the system can use their dot product as cosine similarity.

Higher similarity indicates greater semantic similarity between the resume and vacancy.

---

## Vacancy Explanations

The system can additionally explain a recommendation using TF-IDF keywords.

For a resume and vacancy, it can identify:

```text
Resume Keywords
       +
Job Keywords
       ↓
Matched Keywords
```

This makes recommendations easier to interpret instead of presenting only a numerical similarity score.

---

## Caching and Performance

Several mechanisms are used to reduce unnecessary computation and API requests:

* Streamlit caching
* SQLite storage
* Persistent embedding storage
* FAISS indexes
* Cached hh.ru geographical information
* Incremental global-index updates

The hh.ru client also implements retry and backoff logic for temporary API failures.

---

## Testing

The project uses **pytest** for automated testing.

The test suite covers components such as:

* Database operations
* Vacancy searching
* hh.ru API client
* TF-IDF extraction
* Vector storage
* FAISS functionality
* Global index management
* Recommendation model
* Search functionality

Model-related tests use lightweight fake/mock implementations where appropriate, allowing the tests to run without downloading and loading the full transformer model.

Run the tests with:

```bash
pytest
```

---

## Technologies

### Programming Language

* Python

### Machine Learning / NLP

* Sentence Transformers
* scikit-learn
* TF-IDF
* NLP
* Vector embeddings

### Search

* FAISS
* NumPy

### Data Processing

* Pandas
* PyArrow

### Web Application

* Streamlit
* Requests

### Database

* SQLite

### PDF Processing

* PyMuPDF

### Testing

* pytest

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Chupacabra0000/Job-Recommendor_Optimized.git
cd Job-Recommendor_Optimized
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Typical Workflow

1. Launch the Streamlit application.
2. Create an account or log in.
3. Upload a PDF resume.
4. Select the required region or city.
5. Search for vacancies.
6. The application processes the resume.
7. Vacancies are retrieved from hh.ru.
8. Vacancies are converted into embeddings.
9. Similarity between the resume and vacancies is calculated.
10. Vacancies are ranked by relevance.
11. The user can inspect recommendations and explanations.
12. Interesting vacancies can be saved to favorites or search results.

---

## Project Structure

```text
Job-Recommendor_Optimized/
│
├── app.py
├── model.py
├── hh_client.py
├── hh_areas.py
├── tfidf_terms.py
├── vector_store.py
├── global_faiss_index.py
├── global_index_manager.py
├── db.py
│
├── tests/
│   ├── conftest.py
│   ├── test_db.py
│   ├── test_hh_client.py
│   ├── test_hh_areas.py
│   ├── test_model_and_global_index_manager.py
│   ├── test_tfidf_terms.py
│   ├── test_vector_store.py
│   └── ...
│
├── artifacts/
│   └── ...
│
├── requirements.txt
└── README.md
```

---

## Key Technologies Demonstrated

This project demonstrates practical experience with:

* Python application development
* Object-oriented programming
* REST API integration
* NLP
* Machine learning
* Semantic search
* Vector embeddings
* FAISS
* TF-IDF
* SQLite
* Data processing
* Caching
* Error handling and retry mechanisms
* Automated testing
* Streamlit application development
* Git/GitHub

---

## Purpose

The project was developed to explore how artificial intelligence and NLP techniques can be applied to the job-search process.

Instead of relying exclusively on exact keyword matches, the system attempts to understand the semantic relationship between a candidate's resume and available job descriptions.

This makes it possible to identify potentially relevant vacancies even when the wording used in the resume and vacancy is different.

---


