from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
from transformers import pipeline
import pandas as pd
from utils import get_top_n_words_en, get_top_n_words_id, convert_for_download
import torch
import io
import re

app = FastAPI(title="Simple Sentiment Analyst AI", version="1.0.2")

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

class ConditionInput(BaseModel):
    text_input: str

# Global variables for model storage (Lazy Loading)
lan_model_id = None
lan_model_en = None

def load_models():
    """
    Loads the sentiment analysis models into global variables.
    Uses a singleton pattern to ensure models are only loaded once.
    """
    global lan_model_id, lan_model_en
    try:
        # Check if models are empty; if so, load them for the first time
        if lan_model_id is None or lan_model_en is None: 
            print("Mencoba memuat model untuk PERTAMA KALI...")
            lan_model_id = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")
            lan_model_en = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            print("Semua 4 model berhasil dimuat!")
        return True
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return False

@app.get("/")
def home():
    return {
        "meta": {
            "project_name": "Review Sentiment Analyzer API",
            "version": "1.0.1",
            "authors": ["Silvio Christian, Joe"],
            "description": "High-accuracy Sentiment Analysis API (English & Indonesian) using RoBERTa Transformers & N-Gram Extraction.",
            "tech_stack": ["FastAPI", "Hugging Face Transformers", "Pandas", "Scikit-Learn"]
        },
        "documentation": {
            "swagger_ui": "/docs (Interactive Testing)",
            "redoc": "/redoc (Static Documentation)"
        },
        "features": [
            "Real-time Text Sentiment Analysis (EN/ID)",
            "Batch File Processing (CSV/Excel)",
            "Keyword Extraction (N-Gram Analysis)",
            "Text Complexity Statistics"
        ],
        "usage_guide": {
            "text_analysis_endpoints": {
                "description": "Analyze a single sentence.",
                "urls": [
                    "POST /predict-sentiment/en (English)",
                    "POST /predict-sentiment/id (Indonesian)"
                ],
                "payload_format": {
                    "text_input": "I really love this product! (String)"
                }
            },
            "file_analysis_endpoints": {
                "description": "Upload a file for bulk analysis.",
                "urls": [
                    "POST /predict-table-sentiment/en (English)",
                    "POST /predict-table-sentiment/id (Indonesian)"
                ],
                "file_requirement": "File must be .csv or .xlsx and contain a column named 'komentar'.",
                "form_parameters": {
                    "file": "Binary File (.csv or .xlsx)",
                    "num": "Number of top keywords to extract (Default: 5)",
                    "ngram_min": "Min N-Gram size (Default: 1)",
                    "ngram_max": "Max N-Gram size (Default: 1)",
                    "sentiment": "Filter keywords by sentiment (positive/negative/neutral)"
                }
            }
        },
        "status": "🚀 Server is Running Smoothly."
    }

@app.post("/predict-sentiment/en")
def predict(data: ConditionInput):
    global lan_model_en

    # [ERROR 1] Server Model Check
    if not load_models():
        raise HTTPException(status_code=500, detail="Server Error: Failed to load model. Check server logs.")

    text = data.text_input

    # [ERROR 2] Empty Input Validation (CRITICAL!)
    # Prevent users from sending empty strings or just whitespace
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Input Error: Text cannot be empty or just whitespace.")

    try:
        # Call the Model
        result = lan_model_en(text)
        
        # Safety Check: Ensure the model returns a list and has the 'label' key
        if not result or "label" not in result[0] or "score" not in result[0]:
             raise HTTPException(status_code=500, detail="AI Error: Model returned unexpected format.")

        sentiment = result[0]["label"]   
        confidence = result[0]["score"]   
        return {'prediction': sentiment, "confidence": confidence}

    except Exception as e:
        # Catch-all error for unexpected issues
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")


@app.post("/predict-sentiment/id")
def predict(data: ConditionInput):
    global lan_model_id

    # [ERROR 1] Server Model Check
    if not load_models():
        raise HTTPException(status_code=500, detail="Server Error: Failed to load model. Check server logs.")

    text = data.text_input

    # [ERROR 2] Empty Input Validation
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Input Error: Text cannot be empty or just whitespace.")

    try:
        # Call the Indonesian Model
        result = lan_model_id(text)
        
        # Safety Check Output
        if not result or "label" not in result[0] or "score" not in result[0]:
             raise HTTPException(status_code=500, detail="AI Error: Model returned unexpected format.")
             
        sentiment = result[0]["label"]   
        confidence = result[0]["score"]   
        return {'prediction': sentiment, "confidence": confidence}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")


@app.post("/predict-table-sentiment/en")
async def predict(
    file: UploadFile = File(...),     
    num: int = Form(5, ge=1, le=10),
    sentiment: Literal["positive", "negative", "neutral"] = Form("positive"),
    ngram_min: int= Form(1, ge=1, le=3),
    ngram_max: int= Form(1, ge=1, le=3),
):
    global lan_model_en
    
    # [ERROR 1] Server Model Check
    if not load_models():
        raise HTTPException(status_code=500, detail="Server Error: Failed to load model. Check server logs.")

    # [ERROR 2] User Input Logic Validation
    if ngram_min > ngram_max:
        raise HTTPException(status_code=400, detail="Input Error: 'ngram_min' cannot be greater than 'ngram_max'.")

    # [ERROR 3] File Reading Process
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents) # Convert binary content to a file-like object

        if file.filename.endswith('.csv'):
            try:
                data = pd.read_csv(buffer)
            except UnicodeDecodeError:
                # Try other encoding if UTF-8 fails (e.g., latin1 for Excel CSVs)
                buffer.seek(0)
                data = pd.read_csv(buffer, encoding='latin1')
            except pd.errors.EmptyDataError:
                raise HTTPException(status_code=400, detail="CSV file is empty (no data found).")
            except pd.errors.ParserError:
                raise HTTPException(status_code=400, detail="Invalid/Corrupted CSV format. Ensure the delimiter is a comma.")
            print("✅ Successfully read CSV")

        elif file.filename.endswith(('.xlsx', '.xls')):
            try:
                data = pd.read_excel(buffer)
            except ValueError:
                 raise HTTPException(status_code=400, detail="Excel file is corrupted or format not recognized.")
            print("✅ Successfully read EXCEL")

        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .csv or .xlsx file.")

    except HTTPException as he:
        raise he # Re-raise the HTTP exception we just caught
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # [ERROR 4] Required Column Validation
    if "komentar" not in data.columns:
        raise HTTPException(status_code=400, detail=f"Missing Required Column: File must have a column named 'komentar'. Columns found: {list(data.columns)}")

    # [ERROR 5] Empty Data Check after reading
    if data.empty:
        raise HTTPException(status_code=400, detail="File read successfully, but the table is empty.")

    # --- DEFAULT CODE (CORE LOGIC) ---
    try:
        original_data = data.copy()

        # Handle empty rows in 'komentar' to prevent errors during split/len
        data = data.dropna(subset=['komentar'])
        data['komentar'] = data['komentar'].astype(str)

        if "Sentiment" not in data.columns and "Confidence" not in data.columns:
            try:
                # Predict sentiment for each row
                data['Sentiment'] = data['komentar'].apply(lambda x: lan_model_en(x)[0]["label"])
                data['Confidence'] = data['komentar'].apply(lambda x: f"{round(lan_model_en(x)[0]['score'] * 100, 1)}%")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"AI Error: Failed to predict sentiment. Weird characters in data? Detail: {str(e)}")

        sentiment_count = data["Sentiment"].value_counts().reset_index()

        # Handle case if filter result is empty (e.g., no 'positive' sentiment found)
        try:
            corpus_data = data[data["Sentiment"] == sentiment]["komentar"]
            if corpus_data.empty:
                 result = [] # Safe empty list
            else:
                 # Extract Top N-Grams
                 result = get_top_n_words_en(corpus=corpus_data, n=num, ngram_range=(ngram_min, ngram_max))
        except ValueError as ve:
             # Usually n-gram error if corpus is too small
             print(f"N-Gram Warning: {ve}")
             result = [] 

        result_df = pd.DataFrame(result, columns=["Word", "Jumlah"])

        # Statistics error handling (division by zero, regex fail, etc.)
        try:
            # Calculate text length (sentences) and word length
            data['Text Length'] = data["komentar"].apply(lambda x: len([x for x in re.split(r'[.!?]+', x) if x.strip()]))
            data['Word Length'] = data["komentar"].apply(lambda x: len(x.split()))

            text_data = data.groupby("Sentiment")["Text Length"].mean().round().sort_values().reset_index()
            word_data = data.groupby("Sentiment")["Word Length"].mean().round().sort_values().reset_index()
        except Exception as e:
            print(f"Statistics Warning: {e}")
            text_data = pd.DataFrame() # Return empty to prevent crash
            word_data = pd.DataFrame()

        return {
            "status": "Success",
            "filename": file.filename,
            "rows": len(data),
            "data_preview": original_data.to_dict(orient="records"),
            "predict_result": data.to_dict(orient="records"),
            "sentiment_count": sentiment_count.to_dict(orient="records"),
            "top_keywords": result_df.to_dict(orient="records"),
            "text_length": text_data.to_dict(orient="records"),
            "word_length": word_data.to_dict(orient="records")
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"JSON Key Error: {str(e)}. Check data structure.")
    except Exception as e:
        # Catch-All for unexpected internal errors
        raise HTTPException(status_code=500, detail=f"Internal server error during processing: {str(e)}")
    finally:
        await file.close()


@app.post("/predict-table-sentiment/id")
async def predict(
    file: UploadFile = File(...),     
    num: int = Form(5, ge=1, le=10),
    sentiment: Literal["positive", "negative", "neutral"] = Form("positive"),
    ngram_min: int= Form(1, ge=1, le=3),
    ngram_max: int= Form(1, ge=1, le=3),
):
    global lan_model_id  # <--- Using Indonesian Model
    
    # [ERROR 1] Server Model Check
    if not load_models():
        raise HTTPException(status_code=500, detail="Server Error: Failed to load model. Check server logs.")

    # [ERROR 2] User Input Logic Validation
    if ngram_min > ngram_max:
        raise HTTPException(status_code=400, detail="Input Error: 'ngram_min' cannot be greater than 'ngram_max'.")

    # [ERROR 3] File Reading Process
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)

        if file.filename.endswith('.csv'):
            try:
                data = pd.read_csv(buffer)
            except UnicodeDecodeError:
                # Try other encoding if UTF-8 fails
                buffer.seek(0)
                data = pd.read_csv(buffer, encoding='latin1')
            except pd.errors.EmptyDataError:
                raise HTTPException(status_code=400, detail="CSV file is empty (no data found).")
            except pd.errors.ParserError:
                raise HTTPException(status_code=400, detail="Invalid/Corrupted CSV format. Ensure the delimiter is a comma.")
            print("✅ Successfully read CSV")

        elif file.filename.endswith(('.xlsx', '.xls')):
            try:
                data = pd.read_excel(buffer)
            except ValueError:
                 raise HTTPException(status_code=400, detail="Excel file is corrupted or format not recognized.")
            print("✅ Successfully read EXCEL")

        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .csv or .xlsx file.")

    except HTTPException as he:
        raise he # Re-raise the HTTP exception we just caught
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # [ERROR 4] Required Column Validation
    if "komentar" not in data.columns:
        raise HTTPException(status_code=400, detail=f"Missing Required Column: File must have a column named 'komentar'. Columns found: {list(data.columns)}")

    # [ERROR 5] Empty Data Check after reading
    if data.empty:
        raise HTTPException(status_code=400, detail="File read successfully, but the table is empty.")

    # --- DEFAULT CODE (CORE LOGIC) ---
    try:
        original_data = data.copy()

        # Handle empty rows in 'komentar' to prevent errors during split/len
        data = data.dropna(subset=['komentar'])
        data['komentar'] = data['komentar'].astype(str)

        if "Sentiment" not in data.columns and "Confidence" not in data.columns:
            try:
                # Using Indonesian Model here
                data['Sentiment'] = data['komentar'].apply(lambda x: lan_model_id(x)[0]["label"])
                data['Confidence'] = data['komentar'].apply(lambda x: f"{round(lan_model_en(x)[0]['score'] * 100, 1)}%")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"AI Error: Failed to predict sentiment. Weird characters in data? Detail: {str(e)}")

        sentiment_count = data["Sentiment"].value_counts().reset_index()

        # Handle case if filter result is empty
        try:
            corpus_data = data[data["Sentiment"] == sentiment]["komentar"]
            if corpus_data.empty:
                 result = [] 
            else:
                 # Note: Assuming you use the same helper function name 'get_top_n_words_en' or 'get_top_n_words_id'
                 # (I kept your original function call as requested)
                 result = get_top_n_words_en(corpus=corpus_data, n=num, ngram_range=(ngram_min, ngram_max))
        except ValueError as ve:
             print(f"N-Gram Warning: {ve}")
             result = [] 

        result_df = pd.DataFrame(result, columns=["Word", "Jumlah"])

        # Statistics error handling
        try:
            data['Text Length'] = data["komentar"].apply(lambda x: len([x for x in re.split(r'[.!?]+', x) if x.strip()]))
            data['Word Length'] = data["komentar"].apply(lambda x: len(x.split()))

            text_data = data.groupby("Sentiment")["Text Length"].mean().round().sort_values().reset_index()
            word_data = data.groupby("Sentiment")["Word Length"].mean().round().sort_values().reset_index()
        except Exception as e:
            print(f"Statistics Warning: {e}")
            text_data = pd.DataFrame()
            word_data = pd.DataFrame()

        return {
            "status": "Success",
            "filename": file.filename,
            "rows": len(data),
            "data_preview": original_data.to_dict(orient="records"),
            "predict_result": data.to_dict(orient="records"),
            "sentiment_count": sentiment_count.to_dict(orient="records"),
            "top_keywords": result_df.to_dict(orient="records"),
            "text_length": text_data.to_dict(orient="records"),
            "word_length": word_data.to_dict(orient="records")
        }

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"JSON Key Error: {str(e)}. Check data structure.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during processing: {str(e)}")
    finally:
        await file.close()