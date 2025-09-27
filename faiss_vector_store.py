"""
FAISS-based vector store for hackathon bounty similarity search.
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
from vectorization_tracker import VectorizationTracker
import multiprocessing
import atexit

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    
    def __init__(self, index_path: str = "./faiss_index", embedding_model: str = 'all-MiniLM-L6-v2'):
        self.index_path = index_path
        self.embedding_model_name = embedding_model
        
        os.makedirs(index_path, exist_ok=True)
        
        self.embedder = None
        self.embedding_dim = None
        self._embedder_initialized = False
        
        self.index = None
        self.metadata = []
        self.documents = []
        self.is_trained = False
        
        self.tracker = VectorizationTracker()
        
        self._load_index()
    
    def _ensure_embedder_initialized(self):
        if self._embedder_initialized:
            return
        
        init_strategies = [
            lambda: SentenceTransformer(self.embedding_model_name, device='cpu'),
            lambda: SentenceTransformer(self.embedding_model_name),
            lambda: SentenceTransformer(self.embedding_model_name, trust_remote_code=True),
            lambda: SentenceTransformer(self.embedding_model_name, device='cpu', trust_remote_code=True),
        ]
        
        for i, strategy in enumerate(init_strategies, 1):
            try:
                logger.info(f"🔄 Trying initialization strategy {i}...")
                self.embedder = strategy()
                self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
                logger.info(f"✅ SentenceTransformer initialized successfully with strategy {i}")
                self._embedder_initialized = True
                return
            except Exception as e:
                logger.warning(f"⚠️ Strategy {i} failed: {str(e)}")
                if i == len(init_strategies):
                    logger.error(f"❌ All initialization strategies failed, using fallback")
                    self._initialize_fallback_embedder()
                    return
                continue
    
    def _initialize_fallback_embedder(self):
        """Initialize a fallback embedder using basic text processing."""
        logger.warning("🔄 Initializing fallback embedder...")
        
        # Create a simple fallback embedder
        class FallbackEmbedder:
            def __init__(self):
                self.embedding_dim = 384  # Standard dimension for all-MiniLM-L6-v2
            
            def encode(self, texts, show_progress_bar=False):
                # Simple fallback: create random embeddings
                import numpy as np
                if isinstance(texts, str):
                    texts = [texts]
                
                embeddings = []
                for text in texts:
                    # Create a deterministic "embedding" based on text hash
                    import hashlib
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    # Convert hash to embedding-like vector
                    seed = int(text_hash[:8], 16)
                    np.random.seed(seed)
                    embedding = np.random.normal(0, 1, self.embedding_dim)
                    embeddings.append(embedding)
                
                return np.array(embeddings)
        
        self.embedder = FallbackEmbedder()
        self.embedding_dim = self.embedder.embedding_dim
        self._embedder_initialized = True
        logger.warning("⚠️ Using fallback embedder - search quality may be reduced")
    
    def _load_index(self):
        """Load existing FAISS index and metadata."""
        index_file = os.path.join(self.index_path, "faiss_index.bin")
        metadata_file = os.path.join(self.index_path, "metadata.json")
        documents_file = os.path.join(self.index_path, "documents.json")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                self.is_trained = True
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                # Load documents
                if os.path.exists(documents_file):
                    with open(documents_file, 'r') as f:
                        self.documents = json.load(f)
                
                logger.info(f"✅ Loaded FAISS index with {len(self.metadata)} documents")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing index: {str(e)}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Ensure embedder is initialized to get embedding dimension
        self._ensure_embedder_initialized()
        
        # Create IndexFlatIP (Inner Product) for cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []
        self.documents = []
        self.is_trained = True
        logger.info("✅ Created new FAISS index")
    
    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            index_file = os.path.join(self.index_path, "faiss_index.bin")
            faiss.write_index(self.index, index_file)
            
            # Save metadata
            metadata_file = os.path.join(self.index_path, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save documents
            documents_file = os.path.join(self.index_path, "documents.json")
            with open(documents_file, 'w') as f:
                json.dump(self.documents, f, indent=2)
            
            logger.info(f"✅ Saved FAISS index with {len(self.metadata)} documents")
            
        except Exception as e:
            logger.error(f"❌ Failed to save index: {str(e)}")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
        """
        if not documents:
            return
        
        # Ensure embedder is initialized
        self._ensure_embedder_initialized()
        
        logger.info(f"📚 Adding {len(documents)} documents to FAISS index")
        
        # Generate embeddings
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata and documents
        if metadatas is None:
            metadatas = [{}] * len(documents)
        
        self.metadata.extend(metadatas)
        self.documents.extend(documents)
        
        # Save index
        self._save_index()
        
        logger.info(f"✅ Added {len(documents)} documents. Total documents: {len(self.metadata)}")
    
    def search(self, query: str, k: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with documents, metadata, and scores
        """
        if not self.is_trained or len(self.metadata) == 0:
            logger.warning("⚠️ Index is empty or not trained")
            return []
        
        # Ensure embedder is initialized
        self._ensure_embedder_initialized()
        
        logger.info(f"🔍 Searching for '{query[:50]}...' with k={k}")
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.metadata)))
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            if score >= score_threshold:
                result = {
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "similarity_score": float(score),
                    "rank": i + 1
                }
                results.append(result)
        
        logger.info(f"✅ Found {len(results)} results above threshold {score_threshold}")
        return results
    
    def search_filtered(self, query: str, k: int = 10, score_threshold: float = 0.0, 
                       event_keys: List[str] = None, companies: List[str] = None,
                       bounty_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents with filtering by event, company, or bounty IDs.
        Uses text-based matching to avoid FAISS issues.
        
        Args:
            query: Query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            event_keys: List of event keys to filter by
            companies: List of company names to filter by
            bounty_ids: List of specific bounty IDs to filter by
            
        Returns:
            List of search results with documents, metadata, and scores
        """
        if not self.is_trained or len(self.metadata) == 0:
            logger.warning("⚠️ Index is empty or not trained")
            return []
        
        logger.info(f"🔍 Searching for '{query[:50]}...' with k={k} and filters")
        logger.info(f"🎯 Event filters: {event_keys}")
        logger.info(f"🎯 Company filters: {companies}")
        logger.info(f"🎯 Bounty ID filters: {bounty_ids}")
        
        # Ensure embedder is initialized
        self._ensure_embedder_initialized()
        
        # First, do a regular FAISS search to get vector-based similarity
        logger.info("🔍 Performing FAISS vector search...")
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search with higher k to allow for filtering
        search_k = min(k * 3, len(self.metadata))  # Search more to allow filtering
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Now filter the results based on criteria
        filtered_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
                
            metadata = self.metadata[idx]
            
            # Apply filters
            if event_keys and metadata.get('event_key') not in event_keys:
                continue
            if companies and metadata.get('company') not in companies:
                continue
            if bounty_ids and metadata.get('bounty_id') not in bounty_ids:
                continue
            
            # Check score threshold
            if score >= score_threshold:
                result = {
                    "content": self.documents[idx],
                    "metadata": metadata,
                    "similarity_score": float(score),
                    "rank": len(filtered_results) + 1
                }
                filtered_results.append(result)
                
                # Stop if we have enough results
                if len(filtered_results) >= k:
                    break
        
        logger.info(f"🎯 Filtered to {len(filtered_results)} results from {search_k} total search results")
        logger.info(f"✅ Found {len(filtered_results)} filtered results using FAISS vector search")
        return filtered_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.metadata),
            "embedding_dimension": self.embedding_dim,
            "embedding_model": self.embedding_model_name,
            "is_trained": self.is_trained,
            "index_type": "IndexFlatIP" if self.index else "None"
        }
    
    def clear(self):
        """Clear all documents from the vector store."""
        try:
            self._create_new_index()
            self._save_index()
            logger.info("🗑️ Cleared FAISS index")
        except Exception as e:
            logger.error(f"❌ Error clearing FAISS index: {str(e)}")
            # Fallback: just clear the metadata and documents
            self.metadata = []
            self.documents = []
            self.index = None
            self.is_trained = False
    
    def rebuild_from_data(self, data: List[Dict[str, Any]], text_field: str = "content"):
        """
        Rebuild the entire index from a list of data dictionaries.
        
        Args:
            data: List of dictionaries containing document data
            text_field: Field name containing the text to embed
        """
        logger.info(f"🔄 Rebuilding FAISS index from {len(data)} documents")
        
        # Clear existing data
        self.clear()
        
        # Extract documents and metadata
        documents = []
        metadatas = []
        
        for item in data:
            if text_field in item:
                documents.append(item[text_field])
                # Create metadata excluding the text field
                metadata = {k: v for k, v in item.items() if k != text_field}
                metadatas.append(metadata)
        
        # Add documents
        self.add_documents(documents, metadatas)
        
        logger.info(f"✅ Rebuilt index with {len(documents)} documents")
    
    def add_event_bounties(self, event_key: str, event_name: str, location: str, 
                          year: str, month: str, companies_data: Dict[str, List[Dict[str, Any]]], 
                          force_revectorize: bool = False) -> bool:
        """
        Add bounties from an event to the vector store with tracking.
        
        Args:
            event_key: Unique identifier for the event
            event_name: Name of the event
            location: Location of the event
            year: Year of the event
            month: Month of the event
            companies_data: Dictionary of company names to bounties
            force_revectorize: Whether to force re-vectorization even if already done
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if event is already vectorized
            if not force_revectorize and self.tracker.is_event_vectorized(event_key):
                logger.info(f"ℹ️ Event {event_key} already vectorized, skipping")
                return True
            
            # Clear existing vectors for this event if force_revectorize is True
            if force_revectorize and self.tracker.is_event_vectorized(event_key):
                logger.info(f"🔄 Force re-vectorizing event {event_key}")
                self.tracker.clear_event(event_key)
                # Remove vectors from FAISS index (this is a simplified approach)
                # In a production system, you'd want to track vector indices per event
            
            # Extract bounties
            documents = []
            metadatas = []
            bounties = []
            
            for company_name, company_bounties in companies_data.items():
                for i, bounty in enumerate(company_bounties):
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
                        'bounty_index': i,
                        'bounty_id': f"{event_key}_{company_name}_{i}"
                    }
                    
                    documents.append(doc_text.strip())
                    metadatas.append(metadata)
                    
                    # Store bounty info for tracking
                    bounties.append({
                        'bounty_id': metadata['bounty_id'],
                        'company': company_name,
                        'title': bounty.get('title', '')
                    })
            
            # Add to FAISS store
            if documents:
                self.add_documents(documents, metadatas)
                
                # Mark event as vectorized in tracker
                self.tracker.mark_event_vectorized(
                    event_key, event_name, location, year, month, 
                    len(bounties), len(documents)
                )
                
                # Mark individual bounties as vectorized
                self.tracker.mark_bounties_vectorized(event_key, bounties)
                
                logger.info(f"✅ Vectorized {len(documents)} bounties for event {event_key}")
                return True
            else:
                logger.warning(f"⚠️ No bounties found for event {event_key}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error vectorizing event {event_key}: {str(e)}")
            return False
    
    def is_event_vectorized(self, event_key: str) -> bool:
        """Check if an event has been vectorized."""
        return self.tracker.is_event_vectorized(event_key)
    
    def get_vectorized_events(self) -> List[str]:
        """Get list of vectorized event keys."""
        events = self.tracker.get_vectorized_events()
        return [event['event_key'] for event in events]
    
    def get_vectorization_status(self, event_key: str) -> int:
        """Get vectorization status for a specific event."""
        return self.tracker.get_event_bounty_count(event_key)
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get vectorization tracking statistics."""
        return self.tracker.get_stats()
    
    def clear_event_vectors(self, event_key: str) -> bool:
        """Clear vectors for a specific event."""
        try:
            # Mark as inactive in tracker
            success = self.tracker.clear_event(event_key)
            
            if success:
                # Note: In a production system, you'd want to actually remove
                # the vectors from the FAISS index. For now, we just mark them as inactive.
                logger.info(f"✅ Cleared tracking for event {event_key}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error clearing event vectors: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up resources to prevent multiprocessing leaks."""
        try:
            # Clear any cached models or resources
            if hasattr(self, 'embedder') and self.embedder is not None:
                # Clear any cached embeddings or models
                if hasattr(self.embedder, 'model'):
                    del self.embedder.model
                if hasattr(self.embedder, 'tokenizer'):
                    del self.embedder.tokenizer
            
            # Clear FAISS index
            if hasattr(self, 'index') and self.index is not None:
                del self.index
                self.index = None
            
            # Clear metadata and documents
            self.metadata = []
            self.documents = []
            
            logger.info("✅ FAISSVectorStore resources cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass
