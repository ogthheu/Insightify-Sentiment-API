# 🧠 Insightify API: Dual-Lingual Sentiment & N-Gram Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-RoBERTa-yellow?logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Deployed%20on-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![Status](https://img.shields.io/badge/Status-Operational-success)

## 📌 Overview
**Insightify Sentiment API** is a robust NLP microservice capable of understanding context in both **English** and **Indonesian**.

Unlike basic sentiment tools, this engine goes beyond simple "Positive/Negative" labeling. It features a **Batch Processing Pipeline** that can digest entire Excel/CSV datasets, extract key insights using **N-Gram Analysis**, and calculate text complexity statistics in real-time. Powered by state-of-the-art **RoBERTa Transformers**, it offers high-accuracy inference for diverse use cases.

## ✨ Key Features

### 🌐 Dual-Core Intelligence (EN & ID)
The system intelligently routes requests to specialized models:
* **English Core:** Powered by `cardiffnlp/twitter-roberta-base` for nuance detection in global text.
* **Indonesian Core:** Powered by `w11wo/indonesian-roberta-base` to accurately process local slang and formal Bahasa Indonesia.

### 📊 Batch Insight Extraction (The "Deep Dive")
It doesn't just process rows; it analyzes the *whole picture*. When you upload a dataset:
* **Sentiment Classification:** Labels every row with confidence scores.
* **Keyword Extraction:** Uses N-Gram logic to find the most frequent phrases (e.g., "bad service", "pengiriman lambat") specific to a sentiment group.
* **Statistical Analysis:** Calculates word count and sentence complexity averages per sentiment.

### 🛡️ Defensive Architecture
* **Lazy Loading:** Models are loaded into memory only upon the first request to optimize resource usage.
* **Strict Validation:** Prevents crashes by validating file types (`.csv`, `.xlsx`), encoding (UTF-8/Latin1), and empty inputs before processing.
* **Smart Error Handling:** Returns detailed, actionable HTTP error codes (400 vs 500) so developers know exactly what went wrong.

## 🛠️ Tech Stack
* **Framework:** FastAPI (Asynchronous)
* **NLP Core:** Hugging Face Transformers (PyTorch)
* **Data Processing:** Pandas & NumPy
* **Deployment:** Hugging Face Spaces (Dockerized)

## 🚀 The Processing Pipeline
1.  **Routing:** Request hits specific endpoint (EN or ID).
2.  **Ingestion:**
    * *Single Mode:* Validates JSON string.
    * *Batch Mode:* Reads binary stream -> Converts to DataFrame -> Checks for "komentar" column.
3.  **Inference:** Text is passed through the tokenizer and RoBERTa model.
4.  **Extraction (Batch Only):**
    * Aggregates data by sentiment.
    * Runs N-Gram vectorization to find top keywords.
5.  **Response:** Returns a structured JSON containing predictions, statistics, and preview data.

## 🔌 Integration Guide (API Contract)

### Live Base URL
**[https://silvio0-simple-sentiment-analyst.hf.space](https://silvio0-simple-sentiment-analyst.hf.space)**

### 1. Single Text Analysis (Real-time)
Analyze a single sentence instantly.

* **Endpoints:**
    * 🇬🇧 English: `/predict-sentiment/en`
    * 🇮🇩 Indonesian: `/predict-sentiment/id`
* **Method:** `POST`
* **Body (JSON):**
    ```json
    {
      "text_input": "Pelayanan sangat memuaskan dan cepat!"
    }
    ```

### 2. Batch File Analysis (Deep Insight)
Upload a file to analyze thousands of rows at once.

* **Endpoints:**
    * 🇬🇧 English: `/predict-table-sentiment/en`
    * 🇮🇩 Indonesian: `/predict-table-sentiment/id`
* **Method:** `POST`
* **Body (Form-Data):**
    * `file`: (Binary .csv/.xlsx) *Must have column named 'komentar'*
    * `num`: (Int) Number of keywords to extract (Default: 5)
    * `ngram_min`: (Int) Min word phrase length
    * `ngram_max`: (Int) Max word phrase length
    * `sentiment`: (String) Filter keywords by "positive", "negative", or "neutral"

## 📚 Interactive Documentation (Swagger UI)

Don't write code to test. Use the built-in GUI:

1.  **Access Docs:** [https://silvio0-simple-sentiment-analyst.hf.space/docs](https://silvio0-simple-sentiment-analyst.hf.space/docs)
2.  **Select Endpoint:** Choose between Single Text or Table Sentiment.
3.  **Upload/Type:** Input your data directly in the browser.
4.  **Execute:** See the full analysis JSON response immediately.

## 📦 Local Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/viochris/Insightify-Sentiment-API.git
    cd Insightify-Sentiment-API
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**
    ```bash
    uvicorn main:app --reload
    ```
    *Output: `Uvicorn running on http://127.0.0.1:8000`*

-----

**Author:** [Silvio Christian, Joe](https://www.linkedin.com/in/silvio-christian-joe)
*"Turning raw text into actionable insights."*
