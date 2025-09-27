"""
RAG-driven Hackathon Idea Evaluation System

This module implements a comprehensive RAG pipeline for evaluating hackathon ideas
against existing bounty descriptions with evidence-backed feedback.
"""

import streamlit as st
import json
import re
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable ChromaDB telemetry globally
os.environ["ANONYMIZED_TELEMETRY"] = "False"


@dataclass
class BountyMatch:
    """Represents a bounty match with similarity score and evidence"""
    bounty_id: str
    company_name: str
    event_name: str
    title: str
    description: str
    prizes: str
    similarity_score: float
    evidence_phrases: List[str]
    alignment_score: float
    gaps: List[str]


@dataclass
class EvaluationResult:
    """Comprehensive evaluation result with evidence"""
    idea: str
    bounty_matches: List[BountyMatch]
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    evidence: Dict[str, str]
    recommendations: List[str]
    verdict: str
    confidence_level: str


class RAGEvaluator:
    """
    RAG-driven hackathon idea evaluator that provides critical, constructive,
    and evidence-backed feedback by comparing ideas against bounty descriptions.
    """
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.model = None
        self.openai_client = None
        self.initialize()
        
        # Evaluation criteria with weights
        self.evaluation_criteria = {
            "Problem Significance": {
                "weight": 0.20,
                "description": "Is this a trivial gimmick or a meaningful pain point?",
                "keywords": ["problem", "pain point", "real world", "utility", "need", "challenge"]
            },
            "Novelty / Uniqueness": {
                "weight": 0.20,
                "description": "Is this new, or already solved better by 5 other dApps?",
                "keywords": ["novel", "unique", "innovative", "new", "original", "breakthrough", "first"]
            },
            "User Value": {
                "weight": 0.15,
                "description": "Does this save time, money, or create opportunities?",
                "keywords": ["value", "benefit", "save", "efficiency", "opportunity", "utility", "advantage"]
            },
            "Crypto-Nativeness": {
                "weight": 0.15,
                "description": "Does it require Web3, or could it be a Web2 SaaS with a wallet connect?",
                "keywords": ["blockchain", "decentralized", "web3", "crypto", "defi", "nft", "dao", "smart contract"]
            },
            "Feasibility": {
                "weight": 0.10,
                "description": "Can this actually be built in hackathon constraints?",
                "keywords": ["feasible", "buildable", "practical", "implementable", "realistic", "achievable"]
            },
            "Technical Innovation": {
                "weight": 0.10,
                "description": "Does it showcase technical depth and innovation?",
                "keywords": ["technical", "innovation", "advanced", "complex", "sophisticated", "cutting-edge"]
            },
            "Market Potential": {
                "weight": 0.10,
                "description": "Would users or DAOs actually try this?",
                "keywords": ["adoption", "market", "users", "demand", "popular", "mainstream", "scalable"]
            }
        }
    
    def initialize(self):
        """Initialize ChromaDB, sentence transformer model, and OpenAI client"""
        try:
            # Initialize ChromaDB client with telemetry disabled
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self._log_info("✅ ChromaDB client initialized")
            
            # Initialize sentence transformer model
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self._log_info("✅ Sentence transformer model loaded")
            except Exception as model_error:
                self._log_error(f"Failed to load sentence transformer: {str(model_error)}")
                self.model = None
            
            # Initialize OpenAI client
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                try:
                    # Initialize with minimal parameters to avoid proxy issues
                    self.openai_client = OpenAI(api_key=api_key)
                    self._log_success("✅ OpenAI client initialized successfully!")
                except Exception as openai_error:
                    self._log_error(f"Failed to initialize OpenAI client: {str(openai_error)}")
                    self.openai_client = None
            else:
                self._log_warning("⚠️ OpenAI API key not found. LLM evaluation will be disabled. Please set OPENAI_API_KEY in your .env file.")
                self.openai_client = None
            
            # Use the same collection as the main vectorizer
            try:
                self.collection = self.client.get_or_create_collection(
                    name="hackathon_bounties",  # Same as main vectorizer
                    metadata={"hnsw:space": "cosine"}
                )
                self._log_info("✅ ChromaDB collection created/retrieved")
            except Exception as collection_error:
                self._log_error(f"Failed to create/retrieve collection: {str(collection_error)}")
                self.collection = None
            
            self._log_success("✅ RAG evaluator initialized successfully!")
        except Exception as e:
            self._log_error(f"Error initializing RAG evaluator: {str(e)}")
            # Ensure collection is None if initialization fails
            self.collection = None
    
    def _log_info(self, message):
        """Log info message, handling both Streamlit and non-Streamlit contexts"""
        try:
            st.info(message)
        except:
            print(f"INFO: {message}")
    
    def _log_success(self, message):
        """Log success message, handling both Streamlit and non-Streamlit contexts"""
        try:
            st.success(message)
        except:
            print(f"SUCCESS: {message}")
    
    def _log_warning(self, message):
        """Log warning message, handling both Streamlit and non-Streamlit contexts"""
        try:
            st.warning(message)
        except:
            print(f"WARNING: {message}")
    
    def _log_error(self, message):
        """Log error message, handling both Streamlit and non-Streamlit contexts"""
        try:
            st.error(message)
        except:
            print(f"ERROR: {message}")
    
    # Note: RAG evaluator now uses the same collection as the main vectorizer
    # No need for separate vectorization - data is already available
    
    def validate_input(self, user_input: str) -> Tuple[bool, str]:
        """
        Strict input validation - only accept hackathon ideas
        
        Args:
            user_input: User's input text
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not user_input or not user_input.strip():
            return False, "Please provide your hackathon idea for evaluation."
        
        # Check for non-idea inputs
        question_patterns = [
            r'\?',  # Questions
            r'what is', r'how does', r'can you', r'could you',  # Questions
            r'explain', r'tell me', r'help me',  # Help requests
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, user_input.lower()):
                return False, "Please provide your hackathon idea for evaluation."
        
        # Check for chit-chat patterns
        chat_patterns = [
            r'hello', r'hi there', r'good morning', r'good afternoon',
            r'thanks', r'thank you', r'bye', r'goodbye'
        ]
        
        for pattern in chat_patterns:
            if re.search(pattern, user_input.lower()) and len(user_input.split()) < 10:
                return False, "Please provide your hackathon idea for evaluation."
        
        # Check minimum length
        if len(user_input.split()) < 5:
            return False, "Please provide a more detailed hackathon idea (at least 5 words)."
        
        return True, "Valid hackathon idea input."
    
    def retrieve_similar_bounties(self, idea: str, n_results: int = 7) -> List[BountyMatch]:
        """
        Retrieve most relevant bounties using semantic similarity
        
        Args:
            idea: User's hackathon idea
            n_results: Maximum number of results to return
            
        Returns:
            List of BountyMatch objects
        """
        try:
            if not self.collection:
                st.error("ChromaDB collection not initialized")
                return []
            
            # Generate query embedding
            query_embedding = self.model.encode([idea])
            
            # Search for similar bounties
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            bounty_matches = []
            
            if results and results['ids']:
                for i, bounty_id in enumerate(results['ids']):
                    if i < len(results['distances'][0]):
                        distance = results['distances'][0][i]
                        similarity_score = 1 - distance
                        
                        metadata = results['metadatas'][i]
                        document = results['documents'][i]
                        
                        # Extract evidence phrases
                        evidence_phrases = self._extract_evidence_phrases(idea, document)
                        
                        # Calculate alignment score
                        alignment_score = self._calculate_alignment_score(idea, document)
                        
                        # Identify gaps
                        gaps = self._identify_gaps(idea, document)
                        
                        bounty_match = BountyMatch(
                            bounty_id=bounty_id,
                            company_name=metadata['company_name'],
                            event_name=metadata['event_name'],
                            title=metadata['title'],
                            description=metadata['description'],
                            prizes=metadata['prizes'],
                            similarity_score=similarity_score,
                            evidence_phrases=evidence_phrases,
                            alignment_score=alignment_score,
                            gaps=gaps
                        )
                        
                        bounty_matches.append(bounty_match)
            
            return bounty_matches
            
        except Exception as e:
            st.error(f"Error retrieving similar bounties: {str(e)}")
            return []
    
    def _extract_evidence_phrases(self, idea: str, bounty_text: str) -> List[str]:
        """Extract specific phrases from bounty that support or contradict the idea"""
        evidence_phrases = []
        
        # Split bounty text into sentences
        sentences = re.split(r'[.!?]+', bounty_text)
        
        # Look for relevant phrases
        idea_words = set(idea.lower().split())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            # Check for keyword matches
            sentence_lower = sentence.lower()
            matches = 0
            
            for word in idea_words:
                if word in sentence_lower and len(word) > 3:  # Avoid short words
                    matches += 1
            
            if matches >= 2:  # At least 2 word matches
                evidence_phrases.append(sentence)
        
        return evidence_phrases[:5]  # Limit to top 5 phrases
    
    def _calculate_alignment_score(self, idea: str, bounty_text: str) -> float:
        """Calculate how well the idea aligns with the bounty requirements"""
        # Simple keyword-based alignment scoring
        idea_lower = idea.lower()
        bounty_lower = bounty_text.lower()
        
        # Count matching technical terms
        tech_terms = ['blockchain', 'defi', 'nft', 'dao', 'smart contract', 'web3', 'crypto', 'decentralized']
        matches = sum(1 for term in tech_terms if term in idea_lower and term in bounty_lower)
        
        # Count matching problem/solution keywords
        problem_terms = ['problem', 'solution', 'challenge', 'pain', 'issue', 'need']
        problem_matches = sum(1 for term in problem_terms if term in idea_lower and term in bounty_lower)
        
        # Calculate alignment score (0-1)
        total_possible = len(tech_terms) + len(problem_terms)
        total_matches = matches + problem_matches
        
        return min(total_matches / total_possible, 1.0) if total_possible > 0 else 0.0
    
    def _identify_gaps(self, idea: str, bounty_text: str) -> List[str]:
        """Identify gaps between the idea and bounty requirements"""
        gaps = []
        
        idea_lower = idea.lower()
        bounty_lower = bounty_text.lower()
        
        # Check for missing technical requirements
        tech_requirements = ['smart contract', 'blockchain', 'defi', 'nft', 'dao']
        for req in tech_requirements:
            if req in bounty_lower and req not in idea_lower:
                gaps.append(f"Missing {req} component mentioned in bounty")
        
        # Check for missing problem focus
        if 'problem' in bounty_lower and 'problem' not in idea_lower:
            gaps.append("Bounty focuses on solving specific problems - clarify the problem your idea addresses")
        
        # Check for missing user focus
        if 'user' in bounty_lower and 'user' not in idea_lower:
            gaps.append("Bounty emphasizes user experience - specify who your target users are")
        
        return gaps[:3]  # Limit to top 3 gaps
    
    def evaluate_idea(self, idea: str, bounty_matches: List[BountyMatch]) -> EvaluationResult:
        """
        Comprehensive evaluation of the hackathon idea against retrieved bounties
        
        Args:
            idea: User's hackathon idea
            bounty_matches: List of similar bounty matches
            
        Returns:
            EvaluationResult with comprehensive analysis
        """
        # Calculate scores for each criterion
        criterion_scores = {}
        evidence = {}
        
        for criterion, config in self.evaluation_criteria.items():
            score, criterion_evidence = self._evaluate_criterion(idea, bounty_matches, criterion, config)
            criterion_scores[criterion] = score
            evidence[criterion] = criterion_evidence
        
        # Calculate overall score
        overall_score = sum(
            score * config['weight'] 
            for criterion, score in criterion_scores.items() 
            for config in [self.evaluation_criteria[criterion]]
        )
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(criterion_scores, evidence)
        weaknesses = self._identify_weaknesses(criterion_scores, evidence)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(idea, bounty_matches, criterion_scores, evidence)
        
        # Determine verdict and confidence
        verdict, confidence_level = self._determine_verdict(overall_score, criterion_scores)
        
        return EvaluationResult(
            idea=idea,
            bounty_matches=bounty_matches,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            evidence=evidence,
            recommendations=recommendations,
            verdict=verdict,
            confidence_level=confidence_level
        )
    
    def _evaluate_criterion(self, idea: str, bounty_matches: List[BountyMatch], 
                          criterion: str, config: Dict) -> Tuple[float, str]:
        """Evaluate a specific criterion against the idea and bounty matches"""
        idea_lower = idea.lower()
        keywords = config['keywords']
        
        # Count keyword matches in the idea
        keyword_matches = sum(1 for keyword in keywords if keyword in idea_lower)
        keyword_score = min(keyword_matches / len(keywords), 1.0)
        
        # Check alignment with bounty matches
        alignment_scores = [match.alignment_score for match in bounty_matches]
        avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
        
        # Combine keyword and alignment scores
        final_score = (keyword_score * 0.6 + avg_alignment * 0.4) * 10  # Scale to 0-10
        
        # Generate evidence
        evidence_parts = []
        
        if keyword_matches > 0:
            evidence_parts.append(f"Found {keyword_matches} relevant keywords: {', '.join([k for k in keywords if k in idea_lower])}")
        
        if avg_alignment > 0.5:
            evidence_parts.append(f"Strong alignment with {len([s for s in alignment_scores if s > 0.5])} bounty requirements")
        elif avg_alignment > 0.2:
            evidence_parts.append(f"Moderate alignment with bounty requirements")
        else:
            evidence_parts.append("Limited alignment with available bounty requirements")
        
        evidence_text = ". ".join(evidence_parts) if evidence_parts else "No clear evidence found"
        
        return final_score, evidence_text
    
    def _identify_strengths(self, criterion_scores: Dict[str, float], evidence: Dict[str, str]) -> List[str]:
        """Identify strengths based on high-scoring criteria"""
        strengths = []
        
        for criterion, score in criterion_scores.items():
            if score >= 7:  # High score threshold
                strengths.append(f"**{criterion}** (Score: {score:.1f}/10): {evidence[criterion]}")
        
        return strengths
    
    def _identify_weaknesses(self, criterion_scores: Dict[str, float], evidence: Dict[str, str]) -> List[str]:
        """Identify weaknesses based on low-scoring criteria"""
        weaknesses = []
        
        for criterion, score in criterion_scores.items():
            if score < 4:  # Low score threshold
                weaknesses.append(f"**{criterion}** (Score: {score:.1f}/10): {evidence[criterion]}")
        
        return weaknesses
    
    def _generate_recommendations(self, idea: str, bounty_matches: List[BountyMatch], 
                                criterion_scores: Dict[str, float], evidence: Dict[str, str]) -> List[str]:
        """Generate specific recommendations for improvement"""
        recommendations = []
        
        # Recommendations based on low-scoring criteria
        for criterion, score in criterion_scores.items():
            if score < 5:
                if criterion == "Problem Significance":
                    recommendations.append("Clearly articulate the specific problem your idea solves and why it matters")
                elif criterion == "Novelty / Uniqueness":
                    recommendations.append("Research existing solutions and highlight what makes your approach unique")
                elif criterion == "User Value":
                    recommendations.append("Specify concrete benefits users will gain from your solution")
                elif criterion == "Crypto-Nativeness":
                    recommendations.append("Explain why blockchain/Web3 is essential for your solution")
                elif criterion == "Feasibility":
                    recommendations.append("Break down your idea into achievable milestones for a hackathon timeline")
                elif criterion == "Technical Innovation":
                    recommendations.append("Highlight the technical challenges and innovations in your approach")
                elif criterion == "Market Potential":
                    recommendations.append("Identify your target users and explain why they would adopt your solution")
        
        # Recommendations based on bounty matches
        if bounty_matches:
            best_match = max(bounty_matches, key=lambda x: x.similarity_score)
            if best_match.similarity_score > 0.7:
                recommendations.append(f"Your idea aligns well with {best_match.company_name}'s bounty - consider focusing on their specific requirements")
            
            # Address gaps
            for match in bounty_matches[:3]:  # Top 3 matches
                for gap in match.gaps:
                    recommendations.append(f"Consider addressing: {gap}")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _determine_verdict(self, overall_score: float, criterion_scores: Dict[str, float]) -> Tuple[str, str]:
        """Determine final verdict and confidence level"""
        # Calculate confidence based on score consistency
        scores = list(criterion_scores.values())
        score_variance = np.var(scores) if len(scores) > 1 else 0
        confidence = "high" if score_variance < 5 else "medium" if score_variance < 10 else "low"
        
        # Determine verdict
        if overall_score >= 8:
            verdict = "🎉 **Excellent!** Your idea shows strong potential and aligns well with available bounties."
        elif overall_score >= 6:
            verdict = "👍 **Good!** Your idea has solid potential with some areas for improvement."
        elif overall_score >= 4:
            verdict = "⚠️ **Needs Work.** Your idea has potential but requires significant refinement."
        else:
            verdict = "🚨 **Major Concerns.** Consider pivoting or significantly revising your approach."
        
        return verdict, confidence
    
    def generate_structured_response(self, evaluation_result: EvaluationResult) -> str:
        """Generate a structured, evidence-backed response"""
        response_parts = []
        
        # Header
        response_parts.append("## 🚀 Hackathon Idea Evaluation Report")
        response_parts.append(f"**Idea:** {evaluation_result.idea}")
        response_parts.append(f"**Overall Score:** {evaluation_result.overall_score:.1f}/10")
        response_parts.append(f"**Confidence Level:** {evaluation_result.confidence_level.title()}")
        response_parts.append("")
        
        # Verdict
        response_parts.append("### ⭐ Verdict")
        response_parts.append(evaluation_result.verdict)
        response_parts.append("")
        
        # Strengths
        if evaluation_result.strengths:
            response_parts.append("### ✅ Strengths")
            for strength in evaluation_result.strengths:
                response_parts.append(f"- {strength}")
            response_parts.append("")
        
        # Weaknesses
        if evaluation_result.weaknesses:
            response_parts.append("### ⚠️ Weaknesses")
            for weakness in evaluation_result.weaknesses:
                response_parts.append(f"- {weakness}")
            response_parts.append("")
        
        # Evidence
        response_parts.append("### 📌 Evidence")
        for criterion, evidence_text in evaluation_result.evidence.items():
            response_parts.append(f"**{criterion}:** {evidence_text}")
        response_parts.append("")
        
        # Bounty Matches
        if evaluation_result.bounty_matches:
            response_parts.append("### 🎯 Relevant Bounty Matches")
            for i, match in enumerate(evaluation_result.bounty_matches[:3], 1):
                response_parts.append(f"**{i}. {match.company_name} - {match.title}**")
                response_parts.append(f"   - Similarity: {match.similarity_score:.1%}")
                response_parts.append(f"   - Alignment: {match.alignment_score:.1%}")
                if match.evidence_phrases:
                    response_parts.append(f"   - Key Evidence: {match.evidence_phrases[0]}")
                response_parts.append("")
        
        # Recommendations
        if evaluation_result.recommendations:
            response_parts.append("### 💡 Recommendations")
            for i, rec in enumerate(evaluation_result.recommendations, 1):
                response_parts.append(f"{i}. {rec}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def generate_llm_evaluation(self, idea: str, bounty_matches: List[BountyMatch]) -> str:
        """
        Generate comprehensive evaluation using OpenAI LLM
        
        Args:
            idea: User's hackathon idea
            bounty_matches: List of similar bounty matches
            
        Returns:
            LLM-generated evaluation report
        """
        if not self.openai_client:
            return "❌ **LLM Evaluation Unavailable**\n\nOpenAI API key not configured. Please set OPENAI_API_KEY in your .env file to enable LLM-powered evaluation."
        
        try:
            # Prepare bounty descriptions for the prompt
            bounty_descriptions = []
            for i, match in enumerate(bounty_matches[:5], 1):  # Limit to top 5 matches
                bounty_descriptions.append(f"""
Bounty {i}: {match.company_name} - {match.title}
Description: {match.description}
Prizes: {match.prizes}
Similarity Score: {match.similarity_score:.1%}
""")
            
            bounty_text = "\n".join(bounty_descriptions)
            
            # Create comprehensive prompt
            prompt = f"""You are an expert hackathon judge evaluating a project idea against available bounty descriptions. Provide a comprehensive, critical, and constructive evaluation.

CANDIDATE IDEA:
{idea}

AVAILABLE BOUNTY DESCRIPTIONS:
{bounty_text}

EVALUATION CRITERIA:
1. Problem Significance (20% weight): Is this a trivial gimmick or a meaningful pain point?
2. Novelty/Uniqueness (20% weight): Is this new, or already solved better by 5 other dApps?
3. User Value (15% weight): Does this save time, money, or create opportunities?
4. Crypto-Nativeness (15% weight): Does it require Web3, or could it be a Web2 SaaS with a wallet connect?
5. Feasibility (10% weight): Can this actually be built in hackathon constraints?
6. Technical Innovation (10% weight): Does it showcase technical depth and innovation?
7. Market Potential (10% weight): Would users or DAOs actually try this?

Please provide a structured evaluation report with:

## 🚀 Hackathon Idea Evaluation Report

**Idea:** [Restate the idea briefly]
**Overall Score:** [X.X/10]
**Confidence Level:** [High/Medium/Low]

### ⭐ Verdict
[Clear verdict: Excellent/Good/Needs Work/Major Concerns with brief reasoning]

### ✅ Strengths
- [List 2-3 key strengths with specific evidence from bounty matches]

### ⚠️ Weaknesses  
- [List 2-3 key weaknesses with specific evidence from bounty matches]

### 📌 Evidence & Analysis
[For each criterion, provide specific evidence and reasoning based on the bounty descriptions]

### 🎯 Bounty Alignment
[Analyze how well the idea aligns with the most relevant bounties, citing specific requirements]

### 💡 Recommendations
[Provide 3-5 specific, actionable recommendations for improvement]

Be critical but constructive. Focus on evidence from the bounty descriptions. Use specific examples and concrete suggestions."""

            # Get model configuration
            model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
            max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert hackathon judge with deep knowledge of blockchain, Web3, and startup evaluation. Provide critical, constructive, and evidence-backed feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"❌ **LLM Evaluation Error**\n\nError generating LLM evaluation: {str(e)}\n\nFalling back to rule-based evaluation."
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            if not self.collection:
                return 0
            results = self.collection.get()
            return len(results['ids']) if results['ids'] else 0
        except Exception as e:
            st.error(f"Error getting collection count: {str(e)}")
            return 0
    
    def evaluate_hackathon_idea(self, idea: str) -> str:
        """
        Main evaluation function - validates input and returns structured evaluation
        
        Args:
            idea: User's hackathon idea
            
        Returns:
            Structured evaluation response
        """
        # Validate input
        is_valid, message = self.validate_input(idea)
        if not is_valid:
            return f"❌ **Input Validation Error**\n\n{message}\n\nPlease provide your hackathon idea for evaluation."
        
        # Check if collection has data
        collection_count = self.get_collection_count()
        if collection_count == 0:
            return "❌ **No Bounty Data Available**\n\nNo bounty data found in the RAG evaluation system. Please ensure bounty data is properly loaded before running AI evaluation."
        
        # Retrieve similar bounties
        with st.spinner("🔍 Finding relevant bounties..."):
            bounty_matches = self.retrieve_similar_bounties(idea)
        
        if not bounty_matches:
            return f"❌ **No Relevant Bounties Found**\n\nUnable to find relevant bounties for your idea. The system has {collection_count} bounties loaded. Please try a more specific description or check if your idea matches the available bounty categories."
        
        # Evaluate the idea
        if self.openai_client:
            # Use LLM-powered evaluation
            with st.spinner("🤖 AI is analyzing your idea with LLM..."):
                return self.generate_llm_evaluation(idea, bounty_matches)
        else:
            # Fall back to rule-based evaluation
            with st.spinner("📊 Evaluating your idea with rule-based system..."):
                evaluation_result = self.evaluate_idea(idea, bounty_matches)
            
            # Generate structured response
            return self.generate_structured_response(evaluation_result)
