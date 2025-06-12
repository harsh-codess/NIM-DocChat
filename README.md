# **Nvidia NIM RAG Demo**

**👤 Created by: Harsh Gidwani**  
**🔗 LinkedIn: [https://www.linkedin.com/in/harsh-gidwani-497a63243/](https://www.linkedin.com/in/harsh-gidwani-497a63243/)**

---

## 📋 Project Overview

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system using **Nvidia NIM (Neural Inference Microservice)** endpoints. The application allows users to upload PDF documents, create vector embeddings, and ask questions based on the document content using advanced language models.

## 🚀 Features

- **Document Processing**: Automatic PDF loading and text extraction
- **Vector Embeddings**: Utilizes Nvidia's embedding models for semantic search
- **Question Answering**: Context-aware responses using Nvidia's LLM models
- **Interactive UI**: Clean Streamlit interface for easy interaction
- **Similarity Search**: Shows relevant document chunks for transparency
- **Error Handling**: Robust error management with user-friendly messages

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **LLM Provider**: Nvidia NIM API
- **Embedding Models**: Nvidia NV-Embed models
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: LangChain + PyPDF
- **Environment Management**: Python dotenv

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Nvidia-NIM-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```
   NVIDIA_API_KEY=your_nvidia_api_key_here
   ```

4. **Create documents directory**:
   ```bash
   mkdir us_census
   ```
   Place your PDF documents in the `us_census` folder.

## 🏃‍♂️ Usage

1. **Start the application**:
   ```bash
   streamlit run app1.py
   ```

2. **Access the interface**:
   Open your browser and navigate to `http://localhost:8501`

3. **Create embeddings**:
   - Click "Documents Embedding" to process your PDF files
   - Wait for the vector database to be created

4. **Ask questions**:
   - Enter your question in the text input
   - Get AI-powered answers based on your documents
   - View relevant document chunks in the expander

## 📁 Project Structure

```
Nvidia-NIM-main/
├── app1.py              # Main Streamlit application
├── app.py               # Basic Nvidia API test script
├── .env                 # Environment variables (API keys)
├── us_census/           # Directory for PDF documents
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## 🔧 Configuration

### Supported Models

- **Chat Model**: `meta/llama3-70b-instruct`
- **Embedding Models**: 
  - `nvidia/nv-embedqa-e5-v5` (primary)
  - `nvidia/nv-embed-v1` (fallback)

### Parameters

- **Chunk Size**: 700 characters
- **Chunk Overlap**: 50 characters
- **Batch Size**: 10 documents (for rate limiting)
- **Document Limit**: 30 documents maximum

## ⚡ API Endpoints

The application uses Nvidia's NIM API endpoints:
- **Chat Completions**: `https://integrate.api.nvidia.com/v1/chat/completions`
- **Embeddings**: Nvidia embedding service endpoints

## 🔒 Security

- API keys are stored in environment variables
- No hardcoded credentials in the source code
- Proper error handling to prevent key exposure

## 🐛 Troubleshooting

### Common Issues

1. **403 Forbidden Error**: 
   - Verify your Nvidia API key is valid
   - Check if your account has access to embedding endpoints
   - Ensure you haven't exceeded rate limits

2. **No Documents Found**:
   - Verify PDF files are in the `us_census` directory
   - Check file permissions and formats

3. **Memory Issues**:
   - Reduce the number of documents processed
   - Adjust chunk size parameters

## 📈 Performance

- **Response Time**: Typically 2-5 seconds for queries
- **Batch Processing**: Handles large document sets efficiently
- **Memory Usage**: Optimized with document chunking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [GPL V3 License](LICENSE).

## 🙏 Acknowledgments

- **Nvidia** for providing the NIM API endpoints
- **LangChain** for the RAG framework
- **Streamlit** for the web interface
- **FAISS** for efficient vector search

---

## 📞 Contact

**👤 Harsh Gidwani**  
**🔗 LinkedIn: [https://www.linkedin.com/in/harsh-gidwani-497a63243/](https://www.linkedin.com/in/harsh-gidwani-497a63243/)**

*Feel free to connect for collaborations, questions, or feedback!*


