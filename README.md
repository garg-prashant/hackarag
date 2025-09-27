# 🚀 Hackathon Idea Evaluator

A Streamlit application that helps hackathon participants evaluate their ideas against available partner bounties. The app analyzes and matches ideas to bounties using well-defined evaluation metrics and leverages Fluence Network's decentralized compute resources for enhanced analysis.

## Features

- **URL Processing**: Upload bounty URLs and automatically scrape relevant information
- **Data Storage**: Persistent storage of bounty data and chat history
- **Chat Interface**: Interactive AI-powered evaluation and Q&A
- **Evaluation Metrics**: Six core metrics for comprehensive idea assessment
- **Company Management**: Track multiple hackathon partners and their bounties
- **🌐 Fluence Network Integration**: Deploy decentralized compute resources for enhanced evaluation
- **Advanced Analysis**: Run complex algorithms and testing on rented VMs
- **Real-time Monitoring**: Track VM usage, costs, and performance

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

3. **Configure Fluence Network** (Optional for enhanced compute):

   - Sign up at [Fluence Console](https://console.fluence.network)
   - Get your API key from the console
   - Configure in the app's sidebar or edit `fluence_config.json`

4. **Run the Application**:

   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   Open your browser and go to `http://localhost:8501`

## Usage

1. **Select Hackathon Event**: Choose from available hackathon events
2. **Select Companies & Bounties**: Pick which companies and bounties to evaluate against
3. **Describe Your Idea**: Enter your hackathon project idea
4. **Get Similar Bounties**: Find bounties that match your idea using AI
5. **Enhanced Analysis** (Optional): Deploy Fluence VMs for advanced compute analysis
6. **View Results**: Get comprehensive evaluation and recommendations

### Fluence Network Features

- **Deploy VMs**: Rent decentralized compute resources for heavy analysis
- **Advanced Testing**: Run complex algorithms and simulations
- **Real-time Monitoring**: Track VM usage and costs
- **Web Interface**: Access deployed evaluation environments
- **Auto Cleanup**: Automatically terminate VMs to save costs

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
- Enhanced Fluence integration with custom evaluation algorithms
- Multi-cloud deployment options
- Real-time collaboration on VM resources

## Requirements

- Python 3.8+
- Streamlit
- BeautifulSoup4
- Requests
- Pandas

## License

MIT License - Feel free to use and modify for your hackathon projects!
