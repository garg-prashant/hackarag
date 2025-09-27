import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime
import pandas as pd
from urllib.parse import urlparse
import hashlib
import re
import time
import glob
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Fluence Network Integration
from fluence_ui import render_fluence_sidebar, render_fluence_integration_main


# Page configuration
st.set_page_config(
    page_title="Hackathon Idea Evaluator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, step-by-step CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    .stStatusWidget {display: none;}
    
    /* Step Container */
    .step-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .step-container.active {
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    .step-container.completed {
        border-left: 4px solid #10b981;
        background: #f0fdf4;
    }
    
    /* Step Header */
    .step-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .step-number {
        background: #3b82f6;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 1rem;
        font-size: 14px;
    }
    
    .step-number.completed {
        background: #10b981;
    }
    
    .step-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0;
    }
    
    .step-description {
        color: #64748b;
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }
    
    /* Company Cards */
    .company-card {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .company-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
    }
    
    .company-card.selected {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    
    .company-name {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    
    .company-title {
        color: #64748b;
        font-size: 0.875rem;
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        max-width: 80%;
    }
    
    .user-message {
        background: #3b82f6;
        color: white;
        margin-left: auto;
    }
    
    .bot-message {
        background: #f1f5f9;
        color: #1e293b;
        margin-right: auto;
    }
    
    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }
    
    /* Progress Bar */
    .progress-container {
        background: #e2e8f0;
        border-radius: 8px;
        height: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        background: #3b82f6;
        height: 100%;
        transition: width 0.3s ease;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-pending {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-active {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .status-completed {
        background: #d1fae5;
        color: #065f46;
    }
    
    /* Sidebar */
    .sidebar-content {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Main Header */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: #64748b;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class HackathonDataLoader:
    """Load and manage hackathon data from JSON files"""
    
    def __init__(self, data_dir="hackathon_data"):
        self.data_dir = data_dir
        self.events = {}
        self.load_all_events()
    
    def load_all_events(self):
        """Load all hackathon events from JSON files"""
        try:
            json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
            for file_path in json_files:
                filename = os.path.basename(file_path)
                # Extract event info from filename: event-name_location-name_year_month.json
                name_parts = filename.replace('.json', '').split('_')
                if len(name_parts) >= 4:
                    event_name = name_parts[0]
                    location = name_parts[1]
                    year = name_parts[2]
                    month = name_parts[3]
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    event_key = f"{event_name}_{location}_{year}_{month}"
                    self.events[event_key] = {
                        'name': event_name,
                        'location': location,
                        'year': year,
                        'month': month,
                        'filename': filename,
                        'companies': data
                    }
        except Exception as e:
            st.error(f"Error loading hackathon data: {str(e)}")
    
    def get_all_events(self):
        """Get all available events"""
        return self.events
    
    def get_event_companies(self, event_key):
        """Get all companies for a specific event"""
        if event_key in self.events:
            return self.events[event_key]['companies']
        return {}
    
    def get_company_bounties(self, event_key, company_name):
        """Get all bounties for a specific company in an event"""
        companies = self.get_event_companies(event_key)
        if company_name in companies:
            return companies[company_name]
        return []

class BountyVectorizer:
    """Handle vectorization and similarity search for bounty data"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize ChromaDB and sentence transformer model"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Download and initialize sentence transformer model at boot time
            with st.spinner("🔄 Downloading AI model (this may take a moment on first run)..."):
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="hackathon_bounties",
                metadata={"hnsw:space": "cosine"}
            )
            
            st.success("✅ Vector database and AI model initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing vector database: {str(e)}")
    
    def vectorize_bounties(self, event_key, companies_data, force_revectorize=False):
        """Vectorize all bounties for an event"""
        try:
            documents = []
            metadatas = []
            ids = []
            
            # Check which bounties are already vectorized
            existing_ids = set()
            if not force_revectorize:
                try:
                    # Get all existing documents for this event
                    existing_results = self.collection.get(
                        where={"event_key": event_key}
                    )
                    existing_ids = set(existing_results['ids']) if existing_results['ids'] else set()
                except:
                    pass  # If collection is empty or doesn't exist, continue
            
            for company_name, bounties in companies_data.items():
                for i, bounty in enumerate(bounties):
                    # Create unique ID
                    bounty_id = f"{event_key}_{company_name}_{i}"
                    
                    # Skip if already vectorized and not forcing re-vectorization
                    if not force_revectorize and bounty_id in existing_ids:
                        continue
                    
                    # Create document text from bounty data
                    doc_text = f"""
                    Company: {company_name}
                    Title: {bounty.get('title', '')}
                    Description: {bounty.get('description', '')}
                    Prizes: {bounty.get('prizes', '')}
                    """
                    
                    # Create metadata
                    metadata = {
                        'event_key': event_key,
                        'company': company_name,
                        'title': bounty.get('title', ''),
                        'bounty_index': i
                    }
                    
                    documents.append(doc_text.strip())
                    metadatas.append(metadata)
                    ids.append(bounty_id)
            
            # Add to collection
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                st.success(f"✅ Vectorized {len(documents)} bounties for {event_key}")
            else:
                st.info(f"ℹ️ All bounties for {event_key} are already vectorized")
            
        except Exception as e:
            st.error(f"Error vectorizing bounties: {str(e)}")
    
    def get_vectorized_events(self):
        """Get list of events that have been vectorized"""
        try:
            # Get all unique event keys from the collection
            results = self.collection.get()
            if results['metadatas']:
                event_keys = set()
                for metadata in results['metadatas']:
                    if 'event_key' in metadata:
                        event_keys.add(metadata['event_key'])
                return list(event_keys)
            return []
        except Exception as e:
            st.error(f"Error getting vectorized events: {str(e)}")
            return []
    
    def get_vectorization_status(self, event_key):
        """Get vectorization status for a specific event"""
        try:
            results = self.collection.get(
                where={"event_key": event_key}
            )
            return len(results['ids']) if results['ids'] else 0
        except Exception as e:
            return 0
    
    def search_similar_bounties(self, query, event_key=None, n_results=5):
        """Search for similar bounties based on query"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Search parameters
            where_clause = {"event_key": event_key} if event_key else None
            
            # Perform search
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=where_clause
            )
            
            return results
        except Exception as e:
            st.error(f"Error searching similar bounties: {str(e)}")
            return None
    
    def get_bounty_by_id(self, bounty_id):
        """Get bounty details by ID"""
        try:
            result = self.collection.get(ids=[bounty_id])
            if result['ids']:
                return {
                    'document': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
            return None
        except Exception as e:
            st.error(f"Error getting bounty by ID: {str(e)}")
            return None

class HackathonEvaluator:
    def __init__(self):
        self.data_dir = "data"
        self.ensure_data_directory()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def get_url_hash(self, url):
        """Generate a hash for the URL to use as filename"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def scrape_bounty_page(self, url):
        """Scrape bounty information using enhanced static crawler"""
        try:
            st.info("🚀 Starting enhanced web crawler...")
            return self._enhanced_static_scrape(url)
        except Exception as e:
            st.error(f"❌ Error in scraping: {str(e)}")
            return self._enhanced_static_scrape(url)
    
    def _convert_to_bounty_format(self, scraped_data, url):
        """Convert scraped data to bounty format"""
        return {
            'url': url,
            'title': scraped_data.get('event', 'Untitled'),
            'scraped_at': datetime.now().isoformat(),
            'content': f"Event: {scraped_data.get('event', 'Unknown')}\nPrizes: {scraped_data.get('prizes_counted', 0)}",
            'links': [url],
            'requirements': scraped_data.get('requirements', []),
            'prizes': [prize.get('total_prize', 'Unknown') for prize in scraped_data.get('prizes', [])],
            'deadline': None,
            'company': self.extract_company_name(url),
            'structured_data': scraped_data
        }
    
    def _enhanced_static_scrape(self, url):
        """Enhanced static scraper with better extraction"""
        try:
            st.info(f"🌐 Fetching: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            st.success("✅ Page fetched successfully")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            st.info(f"📄 Page title: {soup.title.string if soup.title else 'No title'}")
            
            # Extract event name
            event_name = self._extract_event_name(soup, url)
            st.info(f"🎯 Event name: {event_name}")
            
            # Extract prizes with multiple methods
            prizes_data = self._extract_prizes_enhanced(soup, response.text)
            st.info(f"💰 Found {len(prizes_data)} prizes")
            
            # Extract requirements
            requirements = self._extract_requirements_enhanced(soup)
            st.info(f"📋 Found {len(requirements)} requirements")
            
            # Create structured data
            structured_data = {
                "event": event_name,
                "page_url": url,
                "source_citation": f"{event_name} ({urlparse(url).netloc})",
                "prizes_counted": len(prizes_data),
                "prizes": prizes_data,
                "requirements": requirements
            }
            
            return self._convert_to_bounty_format(structured_data, url)
            
        except Exception as e:
            st.error(f"❌ Enhanced static scraping failed: {str(e)}")
            return self._fallback_scrape(url)
    
    def _extract_event_name(self, soup, url):
        """Extract event name from various sources"""
        # Try multiple methods to get event name
        methods = [
            lambda: soup.find('title').string if soup.find('title') else None,
            lambda: soup.find('h1').get_text().strip() if soup.find('h1') else None,
            lambda: soup.find('h2').get_text().strip() if soup.find('h2') else None,
            lambda: soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else None,
        ]
        
        for method in methods:
            try:
                result = method()
                if result and len(result.strip()) > 3:
                    return result.strip()
            except:
                continue
        
        # Fallback to URL-based name
        domain = urlparse(url).netloc
        if 'ethglobal' in domain.lower():
            return "ETHGlobal Event"
        elif 'devpost' in domain.lower():
            return "Devpost Hackathon"
        else:
            return "Hackathon Event"
    
    def _extract_prizes_enhanced(self, soup, page_text):
        """Enhanced prize extraction using multiple methods"""
        prizes_data = []
        
        # Method 1: Look for specific prize patterns in text
        st.info("🔍 Method 1: Pattern-based extraction...")
        import re
        
        # Enhanced regex patterns for prize amounts
        prize_patterns = [
            r'\$[\d,]+(?:,\d{3})*(?:\.\d{2})?',  # $20,000, $1,500, etc.
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',  # 20000 dollars
            r'prize.*?(\$[\d,]+(?:,\d{3})*(?:\.\d{2})?)',  # prize $20,000
            r'bounty.*?(\$[\d,]+(?:,\d{3})*(?:\.\d{2})?)',  # bounty $20,000
        ]
        
        all_amounts = []
        for pattern in prize_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            all_amounts.extend(matches)
        
        st.info(f"💰 Found {len(all_amounts)} prize amounts: {all_amounts[:10]}")
        
        # Method 2: Look for company names near prize amounts
        st.info("🔍 Method 2: Company name extraction...")
        
        # Find elements that might contain prize information
        prize_containers = []
        
        # Look for common prize-related selectors
        selectors = [
            'div[class*="prize"]',
            'div[class*="bounty"]', 
            'div[class*="sponsor"]',
            'div[class*="company"]',
            'section[class*="prize"]',
            'li:contains("$")',
            'p:contains("$")',
            'span:contains("$")'
        ]
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                prize_containers.extend(elements)
            except:
                continue
        
        st.info(f"🔍 Found {len(prize_containers)} potential prize containers")
        
        # Process containers to extract structured data
        processed_prizes = set()
        
        for i, container in enumerate(prize_containers[:30]):  # Limit to first 30
            try:
                text = container.get_text().strip()
                if len(text) < 10:
                    continue
                
                st.info(f"🔍 Processing container {i+1}: {text[:100]}...")
                
                # Extract company name (usually first meaningful line)
                company_name = "Unknown"
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines:
                    if (len(line) > 2 and 
                        not line.startswith('$') and 
                        not any(skip in line.lower() for skip in ['prize', 'bounty', 'requirement', 'qualification', 'workshop', 'click', 'learn more'])):
                        company_name = line
                        break
                
                # Extract prize amount
                prize_amount = "Unknown"
                for line in lines:
                    if '$' in line:
                        amounts = re.findall(r'\$[\d,]+(?:,\d{3})*(?:\.\d{2})?', line)
                        if amounts:
                            prize_amount = amounts[0]
                            break
                
                # Extract description
                description = ""
                for line in lines:
                    if (len(line) > 20 and 
                        not line.startswith('$') and 
                        'requirement' not in line.lower() and
                        line != company_name):
                        description = line
                        break
                
                # Create unique key
                prize_key = f"{company_name}_{prize_amount}"
                
                if (prize_key not in processed_prizes and 
                    company_name != "Unknown" and 
                    prize_amount != "Unknown"):
                    
                    processed_prizes.add(prize_key)
                    
                    prize_data = {
                        "name": company_name,
                        "total_prize": prize_amount,
                        "about": description
                    }
                    
                    # Try to extract links
                    links = container.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        if href.startswith('http'):
                            if 'twitter.com' in href or 'x.com' in href:
                                prize_data["twitter"] = href
                            else:
                                prize_data["website"] = href
                    
                    prizes_data.append(prize_data)
                    st.success(f"✅ Prize added: {company_name} - {prize_amount}")
                
            except Exception as e:
                st.warning(f"⚠️ Error processing container {i+1}: {str(e)}")
                continue
        
        # If we didn't find structured prizes, create basic ones from amounts
        if len(prizes_data) < 3 and all_amounts:
            st.info("🔄 Creating basic prizes from amounts...")
            for i, amount in enumerate(all_amounts[:15]):
                if amount and '$' in str(amount):
                    prizes_data.append({
                        "name": f"Prize {i+1}",
                        "total_prize": amount if str(amount).startswith('$') else f"${amount}",
                        "about": f"Prize amount: {amount}"
                    })
        
        return prizes_data
    
    def _extract_requirements_enhanced(self, soup):
        """Enhanced requirement extraction"""
        requirements = []
        
        # Look for requirement-related elements
        req_keywords = ['requirement', 'criteria', 'qualification', 'must', 'should', 'need', 'expect']
        
        # Method 1: Look for specific requirement sections
        for keyword in req_keywords:
            elements = soup.find_all(['div', 'section', 'ul', 'ol'], 
                                   class_=re.compile(keyword, re.I))
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) > 10:
                    # Split into individual requirements
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if (len(line) > 15 and 
                            any(kw in line.lower() for kw in req_keywords)):
                            requirements.append(line)
        
        # Method 2: Look for list items
        lists = soup.find_all(['ul', 'ol'])
        for lst in lists:
            items = lst.find_all('li')
            for item in items:
                text = item.get_text().strip()
                if (len(text) > 15 and 
                    any(kw in text.lower() for kw in req_keywords)):
                    requirements.append(text)
        
        # Method 3: Look for paragraphs with requirement keywords
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text().strip()
            if (len(text) > 20 and 
                any(kw in text.lower() for kw in req_keywords)):
                requirements.append(text)
        
        # Remove duplicates and limit
        requirements = list(set(requirements))[:10]
        return requirements
    
    def _fallback_scrape(self, url):
        """Fallback scraping method using basic requests"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract basic information
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Untitled"
            
            # Look for common bounty/prize related content
            bounty_info = {
                'url': url,
                'title': title_text,
                'scraped_at': datetime.now().isoformat(),
                'content': soup.get_text()[:5000],  # First 5000 characters
                'links': [link.get('href') for link in soup.find_all('a', href=True)][:20],  # First 20 links
                'requirements': [],
                'prizes': [],
                'deadline': None,
                'company': self.extract_company_name(url)
            }
            
            # Try to extract more specific information
            # Look for prize amounts, requirements, etc.
            text_content = soup.get_text().lower()
            
            # Extract potential prize information
            prize_patterns = [
                r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',
                r'prize.*?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            ]
            
            for pattern in prize_patterns:
                matches = re.findall(pattern, text_content)
                if matches:
                    bounty_info['prizes'].extend(matches)
            
            # Extract potential requirements
            requirement_keywords = ['requirement', 'criteria', 'must', 'should', 'need', 'expect']
            sentences = soup.get_text().split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in requirement_keywords):
                    if len(sentence.strip()) > 20:  # Filter out very short sentences
                        bounty_info['requirements'].append(sentence.strip())
            
            return bounty_info
            
        except Exception as e:
            st.error(f"Error in fallback scraping: {str(e)}")
            return None
    
    def extract_company_name(self, url):
        """Extract company name from URL"""
        try:
            domain = urlparse(url).netloc
            # Remove common prefixes
            domain = domain.replace('www.', '').replace('blog.', '').replace('dev.', '')
            # Take the main domain part
            return domain.split('.')[0].title()
        except:
            return "Unknown Company"
    
    def save_bounty_data(self, bounty_data):
        """Save bounty data to JSON file"""
        url_hash = self.get_url_hash(bounty_data['url'])
        file_path = os.path.join(self.data_dir, f"{url_hash}.json")
        
        with open(file_path, 'w') as f:
            json.dump(bounty_data, f, indent=2)
        
        return file_path
    
    def load_bounty_data(self, url):
        """Load existing bounty data"""
        url_hash = self.get_url_hash(url)
        file_path = os.path.join(self.data_dir, f"{url_hash}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_all_bounties(self):
        """Get all stored bounty data"""
        bounties = []
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json') and not filename.endswith('_chat.json'):
                    file_path = os.path.join(self.data_dir, filename)
                    try:
                        with open(file_path, 'r') as f:
                            bounty_data = json.load(f)
                            # Ensure it's a dictionary and has required fields
                            if isinstance(bounty_data, dict) and 'url' in bounty_data:
                                bounties.append(bounty_data)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        continue
        return bounties
    
    def save_chat_history(self, url, message, response):
        """Save chat history"""
        url_hash = self.get_url_hash(url)
        chat_file = os.path.join(self.data_dir, f"{url_hash}_chat.json")
        
        chat_data = []
        if os.path.exists(chat_file):
            with open(chat_file, 'r') as f:
                chat_data = json.load(f)
        
        chat_data.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'bot_response': response
        })
        
        with open(chat_file, 'w') as f:
            json.dump(chat_data, f, indent=2)
    
    def load_chat_history(self, url):
        """Load chat history for a URL"""
        url_hash = self.get_url_hash(url)
        chat_file = os.path.join(self.data_dir, f"{url_hash}_chat.json")
        
        if os.path.exists(chat_file):
            with open(chat_file, 'r') as f:
                return json.load(f)
        return []
    
    def display_structured_data(self, bounty_data):
        """Display the structured data in a nice format"""
        if not bounty_data or 'structured_data' not in bounty_data:
            return
        
        structured_data = bounty_data['structured_data']
        
        st.markdown("### 🎯 Structured Data Preview")
        
        # Display event information
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Event:** {structured_data.get('event', 'Unknown')}")
            st.markdown(f"**Source:** {structured_data.get('source_citation', 'Unknown')}")
        with col2:
            st.markdown(f"**Prizes Count:** {structured_data.get('prizes_counted', 0)}")
        
        # Display prizes in a structured format
        if structured_data.get('prizes'):
            st.markdown("#### 💰 Prizes")
            for i, prize in enumerate(structured_data['prizes'][:5]):  # Show first 5 prizes
                with st.expander(f"Prize {i+1}: {prize.get('name', 'Unknown')} - {prize.get('total_prize', 'Unknown')}"):
                    if prize.get('about'):
                        st.write(prize['about'])
                    if prize.get('website'):
                        st.write(f"Website: {prize['website']}")
                    if prize.get('twitter'):
                        st.write(f"Twitter: {prize['twitter']}")
        
        # Display requirements if available
        if structured_data.get('requirements'):
            st.markdown("#### 📋 Requirements")
            for req in structured_data['requirements'][:3]:  # Show first 3 requirements
                st.write(f"• {req}")
    
    def clear_all_data(self):
        """Clear all stored data and reset session state"""
        try:
            # Clear all JSON files in data directory
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    if filename.endswith('.json'):
                        file_path = os.path.join(self.data_dir, filename)
                        os.remove(file_path)
            
            # Reset session state
            st.session_state.current_url = None
            st.session_state.bounty_data = None
            st.session_state.chat_history = []
            st.session_state.current_step = 1
            
            return True
        except Exception as e:
            st.error(f"Error clearing data: {str(e)}")
            return False

def main():
    # Initialize data loader and vectorizer first
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = HackathonDataLoader()
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = BountyVectorizer()
    
    # Clean header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 Hackathon Idea Evaluator</h1>
        <p class="main-subtitle">Transform your ideas into winning solutions with AI-powered evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick data management section at the top
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📊 System Status")
        vectorized_events = st.session_state.vectorizer.get_vectorized_events()
        all_events = list(st.session_state.data_loader.get_all_events().keys())
        vectorized_count = len(vectorized_events)
        total_count = len(all_events)
        
        st.metric("Vectorized Events", f"{vectorized_count}/{total_count}")
    
    with col2:
        if st.button("📥 Load All Data", type="primary", use_container_width=True, key="load_all_data_main"):
            with st.spinner("Loading all unvectorized data..."):
                loaded_count = 0
                for event_key in all_events:
                    if event_key not in vectorized_events:
                        companies = st.session_state.data_loader.get_event_companies(event_key)
                        st.session_state.vectorizer.vectorize_bounties(event_key, companies)
                        loaded_count += 1
                if loaded_count > 0:
                    st.success(f"✅ Loaded {loaded_count} events!")
                else:
                    st.info("All events are already vectorized!")
                st.rerun()
    
    with col3:
        if st.button("🔄 Re-vectorize All", type="secondary", use_container_width=True, key="revectorize_all_main"):
            with st.spinner("Re-vectorizing all events..."):
                for event_key in all_events:
                    companies = st.session_state.data_loader.get_event_companies(event_key)
                    st.session_state.vectorizer.vectorize_bounties(event_key, companies, force_revectorize=True)
                st.success("All events re-vectorized!")
                st.rerun()
    
    st.markdown("---")
    
    evaluator = HackathonEvaluator()
    
    # Initialize session state
    if 'selected_event' not in st.session_state:
        st.session_state.selected_event = None
    if 'selected_companies' not in st.session_state:
        st.session_state.selected_companies = []
    if 'selected_bounties' not in st.session_state:
        st.session_state.selected_bounties = []
    if 'user_idea' not in st.session_state:
        st.session_state.user_idea = ""
    if 'similar_bounties' not in st.session_state:
        st.session_state.similar_bounties = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    # Progress tracking
    def get_progress():
        if not st.session_state.selected_event:
            return 0
        elif st.session_state.selected_event and not st.session_state.selected_companies:
            return 25
        elif st.session_state.selected_companies and not st.session_state.user_idea:
            return 50
        elif st.session_state.user_idea and not st.session_state.similar_bounties:
            return 75
        else:
            return 100
    
    progress = get_progress()
    
    # Progress bar
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress}%"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Select Event
    step1_class = "step-container active" if st.session_state.current_step == 1 else "step-container completed" if st.session_state.selected_event else "step-container"
    st.markdown(f"""
    <div class="{step1_class}">
        <div class="step-header">
            <div class="step-number {'completed' if st.session_state.selected_event else ''}">1</div>
            <div>
                <h3 class="step-title">Select Hackathon Event</h3>
                <p class="step-description">Choose the hackathon event you want to participate in</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.current_step == 1:
        events = st.session_state.data_loader.get_all_events()
        
        if events:
            # Display events in a nice format
            event_options = []
            for event_key, event_data in events.items():
                display_name = f"{event_data['name']} - {event_data['location']} ({event_data['month']} {event_data['year']})"
                event_options.append((display_name, event_key))
            
            selected_display = st.selectbox(
                "Choose an event:",
                options=[opt[0] for opt in event_options],
                index=0 if not st.session_state.selected_event else None
            )
            
            if selected_display:
                selected_event_key = next(opt[1] for opt in event_options if opt[0] == selected_display)
                
                if st.button("🚀 Select Event", type="primary"):
                    st.session_state.selected_event = selected_event_key
                    st.session_state.current_step = 2
                    st.rerun()
        else:
            st.warning("No hackathon events found. Please add JSON files to the hackathon_data folder.")
    
    # Step 2: Select Companies and Bounties
    if st.session_state.current_step >= 2 and st.session_state.selected_event:
        step2_class = "step-container active" if st.session_state.current_step == 2 else "step-container completed"
        st.markdown(f"""
        <div class="{step2_class}">
            <div class="step-header">
                <div class="step-number {'completed' if st.session_state.current_step > 2 else ''}">2</div>
                <div>
                    <h3 class="step-title">Select Companies & Bounties</h3>
                    <p class="step-description">Choose which companies and bounties you're interested in</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_step == 2:
            companies = st.session_state.data_loader.get_event_companies(st.session_state.selected_event)
            
            if companies:
                st.markdown("### 🏢 Select Companies")
                
                # Select All Companies option
                col1, col2 = st.columns([1, 4])
                with col1:
                    select_all_companies = st.checkbox("Select All", key="select_all_companies")
                with col2:
                    if select_all_companies:
                        st.info("All companies will be selected")
                
                # Company selection
                if select_all_companies:
                    selected_companies = list(companies.keys())
                else:
                    selected_companies = st.multiselect(
                        "Choose companies:",
                        options=list(companies.keys()),
                        default=st.session_state.selected_companies
                    )
                
                if selected_companies:
                    st.session_state.selected_companies = selected_companies
                    
                    # Display bounties for selected companies
                    st.markdown("### 🎯 Available Bounties")
                    
                    all_bounties = []
                    for company in selected_companies:
                        bounties = st.session_state.data_loader.get_company_bounties(st.session_state.selected_event, company)
                        for i, bounty in enumerate(bounties):
                            bounty_id = f"{st.session_state.selected_event}_{company}_{i}"
                            all_bounties.append({
                                'id': bounty_id,
                                'company': company,
                                'title': bounty.get('title', ''),
                                'description': bounty.get('description', ''),
                                'prizes': bounty.get('prizes', ''),
                                'bounty_data': bounty
                            })
                    
                    # Select All Bounties option
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        select_all_bounties = st.checkbox("Select All Bounties", key="select_all_bounties")
                    with col2:
                        if select_all_bounties:
                            st.info(f"All {len(all_bounties)} bounties will be selected")
                    
                    # Display bounties with checkboxes
                    selected_bounty_ids = []
                    for bounty in all_bounties:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if select_all_bounties:
                                selected = True
                            else:
                                selected = st.checkbox("", key=f"bounty_{bounty['id']}")
                            if selected:
                                selected_bounty_ids.append(bounty['id'])
                        with col2:
                            st.markdown(f"""
                            <div class="company-card">
                                <div class="company-name">{bounty['company']}</div>
                                <div class="company-title">{bounty['title']}</div>
                                <p style="font-size: 0.8rem; color: #666; margin: 0.5rem 0;">{bounty['description'][:200]}...</p>
                                <p style="font-size: 0.8rem; color: #059669; margin: 0;">💰 {bounty['prizes']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.session_state.selected_bounties = selected_bounty_ids
                    
                    if st.button("🚀 Continue to Idea Evaluation", type="primary"):
                        # Vectorize the selected event's bounties (only unvectorized ones)
                        with st.spinner("🔍 Vectorizing bounties..."):
                            st.session_state.vectorizer.vectorize_bounties(st.session_state.selected_event, companies)
                        st.session_state.current_step = 3
                        st.rerun()
            else:
                st.warning("No companies found for this event.")
    
    # Step 3: Enter Your Idea
    if st.session_state.current_step >= 3 and st.session_state.selected_companies:
        step3_class = "step-container active" if st.session_state.current_step == 3 else "step-container completed"
        st.markdown(f"""
        <div class="{step3_class}">
            <div class="step-header">
                <div class="step-number {'completed' if st.session_state.current_step > 3 else ''}">3</div>
                <div>
                    <h3 class="step-title">Describe Your Idea</h3>
                    <p class="step-description">Tell us about your hackathon project idea</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_step == 3:
            user_idea = st.text_area(
                "Describe your hackathon project idea:",
                placeholder="I'm building a decentralized voting system that uses zero-knowledge proofs to ensure privacy while maintaining transparency...",
                height=150,
                value=st.session_state.user_idea
            )
            
            if st.button("🔍 Find Similar Bounties", type="primary", disabled=not user_idea):
                if user_idea:
                    st.session_state.user_idea = user_idea
                    
                    # Count total bounties for selected companies
                    total_bounties = 0
                    for company in st.session_state.selected_companies:
                        bounties = st.session_state.data_loader.get_company_bounties(st.session_state.selected_event, company)
                        total_bounties += len(bounties)
                    
                    st.info(f"📊 Total bounties selected: {total_bounties}")
                    
                    if total_bounties < 5:
                        st.info("ℹ️ Bounty count is less than 5. Skipping vector database search and showing all selected bounties.")
                        
                        # Create a mock results structure for consistency
                        all_bounties = []
                        for company in st.session_state.selected_companies:
                            bounties = st.session_state.data_loader.get_company_bounties(st.session_state.selected_event, company)
                            for i, bounty in enumerate(bounties):
                                bounty_id = f"{st.session_state.selected_event}_{company}_{i}"
                                all_bounties.append({
                                    'id': bounty_id,
                                    'company': company,
                                    'title': bounty.get('title', ''),
                                    'description': bounty.get('description', ''),
                                    'prizes': bounty.get('prizes', ''),
                                    'bounty_data': bounty
                                })
                        
                        # Create mock results with all bounties
                        mock_results = {
                            'ids': [bounty['id'] for bounty in all_bounties],
                            'distances': [[0.0] * len(all_bounties)],  # All have perfect match
                            'documents': [f"Company: {bounty['company']}\nTitle: {bounty['title']}\nDescription: {bounty['description']}\nPrizes: {bounty['prizes']}" for bounty in all_bounties],
                            'metadatas': [{'company': bounty['company'], 'title': bounty['title'], 'bounty_index': i} for i, bounty in enumerate(all_bounties)]
                        }
                        
                        st.session_state.similar_bounties = mock_results
                        st.session_state.current_step = 4
                        st.rerun()
                    else:
                        with st.spinner("🔍 Finding similar bounties using vector database..."):
                            # Search for similar bounties using vector database
                            n_results = min(10, total_bounties)  # Fetch 5-10 closest ideas
                            results = st.session_state.vectorizer.search_similar_bounties(
                                user_idea, 
                                st.session_state.selected_event, 
                                n_results=n_results
                            )
                            
                            if results and results['ids']:
                                st.session_state.similar_bounties = results
                                st.session_state.current_step = 4
                                st.rerun()
                            else:
                                st.warning("No similar bounties found in vector database.")
    
    # Step 4: Show Similar Bounties and Recommendations
    if st.session_state.current_step >= 4 and st.session_state.similar_bounties:
        step4_class = "step-container active" if st.session_state.current_step == 4 else "step-container completed"
        st.markdown(f"""
        <div class="{step4_class}">
            <div class="step-header">
                <div class="step-number completed">4</div>
                <div>
                    <h3 class="step-title">Similar Bounties & Recommendations</h3>
                    <p class="step-description">Here are the bounties that match your idea</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_step == 4:
            st.markdown("### 🎯 Your Idea")
            st.info(f"**{st.session_state.user_idea}**")
            
            st.markdown("### 🔍 Similar Bounties")
            
            results = st.session_state.similar_bounties
            
            # Check if this is from vector database or mock results
            is_vector_search = 'distances' in results and len(results['distances']) > 0 and len(results['distances'][0]) > 0 and results['distances'][0][0] > 0
            
            for i, bounty_id in enumerate(results['ids']):
                if is_vector_search:
                    # Vector database results
                    distance = results['distances'][0][i] if i < len(results['distances'][0]) else 0
                    bounty_data = st.session_state.vectorizer.get_bounty_by_id(bounty_id)
                    if bounty_data:
                        metadata = bounty_data['metadata']
                        similarity_score = 1 - distance  # Convert distance to similarity
                        
                        st.markdown(f"""
                        <div class="company-card" style="margin: 1rem 0; padding: 1.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <div class="company-name">{metadata['company']}</div>
                                <div style="background: #10b981; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">
                                    {similarity_score:.1%} Match
                                </div>
                            </div>
                            <div class="company-title">{metadata['title']}</div>
                            <p style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">{bounty_data['document']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Mock results (bounty count < 5)
                    if i < len(results['metadatas']):
                        metadata = results['metadatas'][i]
                        document = results['documents'][i] if i < len(results['documents']) else ""
                        
                        st.markdown(f"""
                        <div class="company-card" style="margin: 1rem 0; padding: 1.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <div class="company-name">{metadata['company']}</div>
                                <div style="background: #3b82f6; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">
                                    Selected Bounty
                                </div>
                            </div>
                            <div class="company-title">{metadata['title']}</div>
                            <p style="font-size: 0.9rem; color: #666; margin: 0.5rem 0;">{document}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            
            if is_vector_search:
                st.markdown("""
                Based on the similarity analysis, here are some recommendations:
                
                1. **Focus on the most similar bounties** - These have the highest alignment with your idea
                2. **Consider combining elements** - You might be able to address multiple bounties with one project
                3. **Check requirements carefully** - Make sure your technical approach aligns with what's expected
                4. **Plan your timeline** - Consider the complexity and scope of the bounties you're targeting
                """)
            else:
                st.markdown("""
                Since you have fewer than 5 bounties selected, here are some recommendations:
                
                1. **Review all selected bounties** - You can see all the bounties you've chosen
                2. **Consider expanding your selection** - You might want to select more companies/bounties for better coverage
                3. **Focus on quality over quantity** - Make sure the selected bounties align well with your idea
                4. **Check requirements carefully** - Make sure your technical approach aligns with what's expected
                """)
            
            if st.button("🔄 Start Over", type="secondary"):
                # Reset session state
                for key in ['selected_event', 'selected_companies', 'selected_bounties', 'user_idea', 'similar_bounties', 'current_step']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Fluence Network Integration Section
    if st.session_state.current_step >= 4 and st.session_state.similar_bounties:
        render_fluence_integration_main()
    
    # Sidebar with event info and data management
    with st.sidebar:
        st.markdown("### 📊 Event Information")
        
        if st.session_state.selected_event:
            event_data = st.session_state.data_loader.events[st.session_state.selected_event]
            st.markdown(f"""
            **Event:** {event_data['name']}  
            **Location:** {event_data['location']}  
            **Date:** {event_data['month']} {event_data['year']}  
            **Companies:** {len(st.session_state.selected_companies) if st.session_state.selected_companies else 0} selected
            """)
            
            if st.session_state.selected_companies:
                st.markdown("### 🏢 Selected Companies")
                for company in st.session_state.selected_companies:
                    st.markdown(f"• {company}")
        
        st.markdown("---")
        st.markdown("### 🔄 Data Management")
        
        # Show vectorization status
        st.markdown("#### 📈 Vectorization Status")
        vectorized_events = st.session_state.vectorizer.get_vectorized_events()
        all_events = list(st.session_state.data_loader.get_all_events().keys())
        
        for event_key in all_events:
            event_data = st.session_state.data_loader.events[event_key]
            is_vectorized = event_key in vectorized_events
            status_icon = "✅" if is_vectorized else "⏳"
            status_text = "Vectorized" if is_vectorized else "Not vectorized"
            
            # Get count of vectorized bounties for this event
            if is_vectorized:
                bounty_count = st.session_state.vectorizer.get_vectorization_status(event_key)
                status_text += f" ({bounty_count} bounties)"
            
            st.markdown(f"{status_icon} **{event_data['name']}**: {status_text}")
        
        st.markdown("---")
        
        # Load Data section
        st.markdown("#### 🚀 Load Data")
        
        # Show events that need vectorization
        unvectorized_events = [e for e in all_events if e not in vectorized_events]
        
        if unvectorized_events:
            st.markdown("**Events to vectorize:**")
            for event_key in unvectorized_events:
                event_data = st.session_state.data_loader.events[event_key]
                if st.button(f"📥 Load {event_data['name']}", key=f"load_{event_key}", use_container_width=True):
                    with st.spinner(f"Loading {event_data['name']}..."):
                        companies = st.session_state.data_loader.get_event_companies(event_key)
                        st.session_state.vectorizer.vectorize_bounties(event_key, companies)
                        st.rerun()
        else:
            st.info("All events are vectorized!")
        
        # Force re-vectorize option
        if st.button("🔄 Re-vectorize All", type="secondary", use_container_width=True, key="revectorize_all_sidebar"):
            with st.spinner("Re-vectorizing all events..."):
                for event_key in all_events:
                    companies = st.session_state.data_loader.get_event_companies(event_key)
                    st.session_state.vectorizer.vectorize_bounties(event_key, companies, force_revectorize=True)
                st.success("All events re-vectorized!")
                st.rerun()
        
        # Fluence Network Integration Sidebar
        render_fluence_sidebar()
        
        st.markdown("---")
        st.markdown("### 🗑️ Reset")
        if st.button("🔄 Reset All", type="secondary", use_container_width=True, key="reset_all_sidebar"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key not in ['data_loader', 'vectorizer']:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Cleanup when app closes
        pass
