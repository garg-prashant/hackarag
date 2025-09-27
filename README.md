# 🚀 Hackathon Idea Evaluator

A Streamlit application that helps hackathon participants evaluate their ideas against available partner bounties. The app analyzes and matches ideas to bounties using well-defined evaluation metrics.

## Features

- **URL Processing**: Upload bounty URLs and automatically scrape relevant information
- **Data Storage**: Persistent storage of bounty data and chat history
- **Chat Interface**: Interactive AI-powered evaluation and Q&A
- **Evaluation Metrics**: Six core metrics for comprehensive idea assessment
- **Company Management**: Track multiple hackathon partners and their bounties

## Core Evaluation Metrics

1. **Innovation** – How original and creative is the idea?
2. **Uniqueness** – Does it stand out compared to other solutions?
3. **Correctness** – Is the idea technically sound and logically valid?
4. **Feasibility** – Can the idea realistically be implemented within hackathon constraints?
5. **Risk & Time** – What's the likelihood of completing it within the hackathon deadline?
6. **Future Viability** – Does the idea have potential beyond the hackathon?

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables** (Optional for OpenAI integration):
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Open your browser and go to `http://localhost:8501`

## Usage

1. **Upload Bounty URL**: Enter the URL of a hackathon bounty page
2. **Process URL**: The app will scrape and extract relevant information
3. **Chat Interface**: Describe your idea and get AI-powered feedback
4. **View Companies**: See all processed bounties in the sidebar
5. **Evaluation**: Get detailed feedback based on the six core metrics

## Data Storage

- Bounty data is stored in JSON format in the `data/` directory
- Chat history is preserved for each bounty URL
- Data persists between sessions

## Future Enhancements

- Integration with OpenAI API for more sophisticated AI responses
- RAG (Retrieval-Augmented Generation) system for better knowledge management
- Advanced web scraping for complex bounty pages
- Export functionality for evaluation reports
- Team collaboration features

## Requirements

- Python 3.8+
- Streamlit
- BeautifulSoup4
- Requests
- Pandas

## License

MIT License - Feel free to use and modify for your hackathon projects!
