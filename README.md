# Hackathon Idea Evaluator

A comprehensive AI-powered system for evaluating hackathon project ideas against available bounty descriptions using advanced RAG (Retrieval-Augmented Generation) techniques.

## 🚀 Features

- **Multi-Event Support**: Analyze ideas across multiple hackathon events
- **Intelligent Bounty Matching**: Find relevant bounties using semantic similarity
- **AI-Powered Evaluation**: Comprehensive scoring across 8 key metrics
- **Evidence-Based Feedback**: Detailed analysis with specific recommendations
- **Real-Time Processing**: Fast vector search and evaluation
- **Interactive UI**: Clean, modern Streamlit interface

## 🏗️ Architecture

### Core Components

1. **FAISS Vector Store** (`faiss_vector_store.py`)
   - High-performance similarity search using Facebook's FAISS
   - Sentence transformer embeddings for semantic matching
   - Incremental indexing with tracking

2. **RAG Evaluator** (`rag_evaluator.py`)
   - Retrieval-Augmented Generation pipeline
   - Evidence-based evaluation with confidence scoring
   - Anthropic Claude integration for detailed analysis

3. **LangGraph Evaluator** (`langgraph_evaluator.py`)
   - Multi-step evaluation workflow
   - OpenAI GPT-4 integration
   - Comprehensive metrics calculation

4. **Vectorization Tracker** (`vectorization_tracker.py`)
   - SQLite-based tracking system
   - Prevents duplicate vectorization
   - Incremental updates support

5. **Main Application** (`app.py`)
   - Streamlit web interface
   - Event and company selection
   - Real-time evaluation results

## 📊 Evaluation Metrics

The system evaluates hackathon ideas across 8 weighted criteria:

1. **Problem Significance** (20%) - Meaningfulness of the problem being solved
2. **Novelty/Uniqueness** (20%) - Innovation compared to existing solutions
3. **User Value** (15%) - Tangible benefits for end users
4. **Crypto-Nativeness** (15%) - Effective use of blockchain/Web3 technologies
5. **Feasibility** (10%) - Realistic implementation within hackathon constraints
6. **Technical Innovation** (10%) - Technical depth and sophistication
7. **Market Potential** (10%) - Adoption likelihood and scalability

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hackarag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp env_template.txt .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

5. **Prepare hackathon data**
   - Place JSON files in `hackathon_data/` directory
   - Format: `EventName_Location_Year_Month.json`
   - Example: `EthGlobal_New-Delhi_2025_September.json`

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Workflow

1. **Select Events**: Choose hackathon events to analyze
2. **Select Companies**: Pick companies and bounties of interest
3. **Enter Idea**: Describe your hackathon project idea
4. **Get Evaluation**: Receive comprehensive AI-powered analysis

### Data Format

Hackathon data should be structured as JSON files with the following format:

```json
{
  "CompanyName": [
    {
      "title": "Bounty Title",
      "description": "Detailed description of the bounty",
      "prizes": "Prize information",
      "requirements": ["requirement1", "requirement2"]
    }
  ]
}
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for GPT-4 evaluation
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude evaluation
- `PYTORCH_CUDA_ALLOC_CONF`: PyTorch memory configuration
- `TOKENIZERS_PARALLELISM`: Tokenizer parallelism setting

### Model Configuration

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **FAISS Index**: `IndexFlatIP` for cosine similarity
- **LLM Models**: GPT-4o-mini, Claude-3.5-Sonnet

## 📁 Project Structure

```
hackarag/
├── app.py                          # Main Streamlit application
├── faiss_vector_store.py           # FAISS vector store implementation
├── rag_evaluator.py               # RAG evaluation system
├── langgraph_evaluator.py         # LangGraph evaluation workflow
├── langgraph_evaluator_simple.py  # Simplified LangGraph evaluator
├── vectorization_tracker.py       # SQLite tracking system
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── hackathon_data/                # Hackathon JSON data files
├── faiss_index/                   # FAISS index storage
├── data/                          # Application data storage
└── venv/                          # Virtual environment
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
docker-compose up -d
```

### Manual Docker Build

```bash
docker build -t hackathon-evaluator .
docker run -p 8501:8501 hackathon-evaluator
```

## 🔍 API Integration

### OpenAI Integration
- Model: `gpt-4o-mini`
- Temperature: 0.1
- Used for: Comprehensive idea evaluation

### Anthropic Integration
- Model: `claude-3-5-sonnet-20241022`
- Temperature: 0.1
- Used for: Detailed analysis and recommendations

## 📈 Performance

- **Vector Search**: Sub-second similarity search across thousands of bounties
- **Evaluation Time**: 10-30 seconds for comprehensive analysis
- **Memory Usage**: ~2GB RAM for full dataset
- **Storage**: ~100MB for FAISS index per 1000 bounties

## 🛡️ Security

- API keys stored in environment variables
- No sensitive data logged
- Input validation and sanitization
- Error handling with graceful degradation

## 🧪 Testing

### Manual Testing
1. Load sample hackathon data
2. Test idea evaluation workflow
3. Verify bounty matching accuracy
4. Check error handling scenarios

### Performance Testing
- Vector search speed benchmarks
- Memory usage monitoring
- Concurrent user simulation

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify API keys in `.env` file
   - Check API key permissions and quotas

2. **Model Loading Issues**
   - Ensure stable internet connection
   - Check PyTorch installation

3. **Memory Issues**
   - Reduce batch sizes in configuration
   - Use CPU-only mode if GPU memory insufficient

4. **Vector Search Problems**
   - Rebuild FAISS index
   - Check data format consistency

### Debug Mode

Enable verbose logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Facebook AI Research for FAISS
- Hugging Face for sentence transformers
- OpenAI for GPT models
- Anthropic for Claude models
- Streamlit for the web framework

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Built with ❤️ for the hackathon community**