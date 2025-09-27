"""
LangGraph-based intelligent hackathon idea evaluator.
Implements a multi-step flow for idea validation, bounty matching, and metrics calculation.
"""

import os
import time
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
from faiss_vector_store import FAISSVectorStore
import streamlit as st
import logging
from datetime import datetime

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('langgraph_evaluator.log')
    ]
)
logger = logging.getLogger(__name__)

# Create a separate verbose logger for detailed operations
verbose_logger = logging.getLogger('langgraph_verbose')
verbose_handler = logging.StreamHandler()
verbose_handler.setLevel(logging.DEBUG)
verbose_formatter = logging.Formatter('🔍 VERBOSE: %(asctime)s - %(message)s')
verbose_handler.setFormatter(verbose_formatter)
verbose_logger.addHandler(verbose_handler)
verbose_logger.setLevel(logging.DEBUG)


class IdeaEvaluationState(TypedDict):
    """State for the LangGraph evaluation flow."""
    user_idea: str
    selected_companies: List[str]
    selected_bounties: List[str]
    selected_event: Optional[str]
    similar_bounties: List[Dict[str, Any]]
    bounty_descriptions: str
    final_result: str
    error_message: Optional[str]
    step_progress: Dict[str, Any]


class LangGraphIdeaEvaluator:
    """LangGraph-based intelligent idea evaluator with full verbose logging."""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the evaluator with OpenAI API key."""
        verbose_logger.info("🚀 Initializing LangGraph Idea Evaluator...")
        start_time = time.time()
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            verbose_logger.error("❌ OpenAI API key is required but not provided")
            raise ValueError("OpenAI API key is required")
        
        verbose_logger.info(f"✅ OpenAI API key found: {self.openai_api_key[:10]}...")
        
        # Initialize LLM with verbose logging
        verbose_logger.info("🤖 Initializing OpenAI ChatOpenAI model...")
        llm_start = time.time()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=self.openai_api_key
        )
        llm_time = time.time() - llm_start
        verbose_logger.info(f"✅ OpenAI model initialized in {llm_time:.2f}s")
        
        # Test API connection
        verbose_logger.info("🔍 Testing OpenAI API connection...")
        try:
            test_response = self.llm.invoke([HumanMessage(content="Hello, respond with 'OK'")])
            verbose_logger.info(f"✅ OpenAI API connection test successful: {test_response.content[:50]}")
        except Exception as test_error:
            verbose_logger.warning(f"⚠️ OpenAI API connection test failed: {str(test_error)}")
            # Don't raise error here, let it fail during actual usage for better error messages
        
        # Initialize FAISS vector store with verbose logging
        verbose_logger.info("🗄️ Initializing FAISS vector store...")
        faiss_start = time.time()
        self.vector_store = FAISSVectorStore(
            index_path="./faiss_index",
            embedding_model='all-MiniLM-L6-v2'
        )
        faiss_time = time.time() - faiss_start
        verbose_logger.info(f"✅ FAISS vector store initialized in {faiss_time:.2f}s")
        
        # Log vector store stats
        try:
            stats = self.vector_store.get_stats()
            verbose_logger.info(f"📊 Vector store stats: {stats}")
        except Exception as e:
            verbose_logger.warning(f"⚠️ Could not get vector store stats: {e}")
        
        total_init_time = time.time() - start_time
        verbose_logger.info(f"🎉 LangGraph Idea Evaluator fully initialized in {total_init_time:.2f}s")
    
    def _log_verbose_step(self, step_name: str, message: str, data: Any = None):
        """Log verbose step information with timestamp and data."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        verbose_logger.info(f"[{timestamp}] {step_name}: {message}")
        if data is not None:
            verbose_logger.debug(f"[{timestamp}] {step_name} DATA: {json.dumps(data, indent=2, default=str)}")
    
    def _log_api_call(self, api_name: str, request_data: Any, response_data: Any, duration: float, tokens_used: int = 0):
        """Log detailed API call information."""
        verbose_logger.info(f"🌐 API CALL: {api_name} completed in {duration:.2f}s")
        verbose_logger.debug(f"🌐 API REQUEST: {json.dumps(request_data, indent=2, default=str)}")
        verbose_logger.debug(f"🌐 API RESPONSE: {json.dumps(response_data, indent=2, default=str)}")
        if tokens_used > 0:
            verbose_logger.info(f"🌐 TOKENS USED: {tokens_used}")
    
    def _update_step_progress(self, state: IdeaEvaluationState, step: str, progress: Dict[str, Any]):
        """Update step progress in state."""
        if "step_progress" not in state:
            state["step_progress"] = {}
        state["step_progress"][step] = {
            "timestamp": datetime.now().isoformat(),
            "progress": progress
        }
        verbose_logger.info(f"📈 PROGRESS UPDATE [{step}]: {json.dumps(progress, indent=2)}")
    
    def _parse_openai_response(self, response_content: str, step_name: str) -> Dict[str, Any]:
        """Parse OpenAI response with robust error handling and fallback mechanisms."""
        if not response_content or not response_content.strip():
            raise ValueError("OpenAI returned empty response")
        
        # Log raw response for debugging
        self._log_verbose_step(step_name, "Raw OpenAI response", {
            "response_length": len(response_content),
            "response_preview": response_content[:200] + "..." if len(response_content) > 200 else response_content
        })
        
        # Try to clean the response if it has markdown formatting
        cleaned_content = response_content.strip()
        
        # Remove markdown code blocks if present
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
        
        cleaned_content = cleaned_content.strip()
        
        # Try parsing the cleaned content
        try:
            result = json.loads(cleaned_content)
            self._log_verbose_step(step_name, "Successfully parsed JSON response")
            return result
        except json.JSONDecodeError as json_err:
            self._log_verbose_step(step_name, f"JSON parsing failed: {str(json_err)}", {
                "raw_response": response_content,
                "cleaned_response": cleaned_content,
                "response_length": len(response_content)
            })
            
            # Try to extract JSON from the response if it's embedded in text
            import re
            json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json_match.group(0)
                    result = json.loads(extracted_json)
                    self._log_verbose_step(step_name, "Successfully extracted and parsed JSON from response")
                    return result
                except json.JSONDecodeError:
                    pass
            
            raise ValueError(f"Failed to parse OpenAI response as JSON: {str(json_err)}. Raw response: {response_content[:500]}")
    
    def step1_find_similar_bounties(self, state: IdeaEvaluationState) -> IdeaEvaluationState:
        """
        Step 1: Validate if the user input is a real, complete idea.
        """
        step_start_time = time.time()
        user_idea = state["user_idea"]
        
        # Update state tracking
        state["current_step"] = "validate_idea"
        state["api_call_count"] = state.get("api_call_count", 0)
        state["total_tokens_used"] = state.get("total_tokens_used", 0)
        
        # Verbose logging
        self._log_verbose_step("STEP1_VALIDATE_IDEA", f"Starting idea validation for: '{user_idea[:100]}...'")
        self._log_verbose_step("STEP1_VALIDATE_IDEA", f"Full user idea length: {len(user_idea)} characters")
        self._log_verbose_step("STEP1_VALIDATE_IDEA", f"User idea word count: {len(user_idea.split())} words")
        
        # Update progress
        self._update_step_progress(state, "validate_idea", {
            "status": "starting",
            "idea_length": len(user_idea),
            "word_count": len(user_idea.split())
        })
        
        logger.info(f"🔍 STEP 1: Starting idea validation for: '{user_idea[:100]}...'")
        st.info("🔍 **Step 1**: Validating idea completeness and quality...")
        
        # Prepare validation prompt with verbose logging
        validation_prompt = f"""
        You are an expert hackathon judge. Analyze the following text to determine if it's a valid, complete hackathon project idea.

        Text to analyze: "{user_idea}"

        Please evaluate:
        1. Is this a real project idea (not spam, random text, or gibberish)?
        2. Is the idea complete enough to be evaluated (not just a single word or incomplete thought)?
        3. Does it describe a potential software/tech solution or project?

        Respond with a JSON object containing:
        {{
            "is_valid_idea": true/false,
            "reason": "Detailed explanation of your decision",
            "completeness_score": 0-10,
            "clarity_score": 0-10
        }}

        Be strict but fair. An idea should be at least 2-3 sentences describing a potential project.
        """
        
        self._log_verbose_step("STEP1_VALIDATE_IDEA", "Validation prompt prepared", {
            "prompt_length": len(validation_prompt),
            "prompt_preview": validation_prompt[:200] + "..."
        })
        
        try:
            # Update progress
            self._update_step_progress(state, "validate_idea", {
                "status": "calling_openai",
                "prompt_length": len(validation_prompt)
            })
            
            logger.info("🤖 Calling OpenAI API for idea validation...")
            self._log_verbose_step("STEP1_VALIDATE_IDEA", "Initiating OpenAI API call")
            
            # Make API call with timing and error handling
            api_start = time.time()
            try:
                response = self.llm.invoke([HumanMessage(content=validation_prompt)])
                api_duration = time.time() - api_start
            except Exception as api_error:
                api_duration = time.time() - api_start
                self._log_verbose_step("STEP1_VALIDATE_IDEA", f"OpenAI API call failed: {str(api_error)}", {
                    "error_type": type(api_error).__name__,
                    "api_duration": api_duration
                })
                raise ValueError(f"OpenAI API call failed: {str(api_error)}")
            
            # Log API call details
            self._log_api_call(
                "OpenAI_Idea_Validation",
                {"prompt": validation_prompt, "model": "gpt-4o-mini"},
                {"response": response.content},
                api_duration
            )
            
            # Parse response with robust error handling
            self._log_verbose_step("STEP1_VALIDATE_IDEA", "Parsing OpenAI response")
            result = self._parse_openai_response(response.content, "STEP1_VALIDATE_IDEA")
            
            # Log detailed validation result
            self._log_verbose_step("STEP1_VALIDATE_IDEA", "Validation analysis complete", {
                "is_valid_idea": result["is_valid_idea"],
                "completeness_score": result.get("completeness_score", "N/A"),
                "clarity_score": result.get("clarity_score", "N/A"),
                "reason_length": len(result["reason"])
            })
            
            # Update state
            state["is_valid_idea"] = result["is_valid_idea"]
            state["validation_reason"] = result["reason"]
            state["api_call_count"] += 1
            
            # Update step timing
            if "step_timings" not in state:
                state["step_timings"] = {}
            state["step_timings"]["validate_idea"] = time.time() - step_start_time
            
            if not result["is_valid_idea"]:
                logger.warning(f"❌ Idea validation failed: {result['reason']}")
                self._log_verbose_step("STEP1_VALIDATE_IDEA", "Idea validation FAILED", {
                    "reason": result["reason"],
                    "completeness_score": result.get("completeness_score"),
                    "clarity_score": result.get("clarity_score")
                })
                
                # Update progress
                self._update_step_progress(state, "validate_idea", {
                    "status": "failed",
                    "reason": result["reason"],
                    "scores": {
                        "completeness": result.get("completeness_score"),
                        "clarity": result.get("clarity_score")
                    }
                })
                
                st.error(f"❌ **Invalid Idea**\n\n{result['reason']}\n\nPlease provide a complete, valid hackathon project idea.")
                state["error_message"] = f"❌ **Invalid Idea**\n\n{result['reason']}\n\nPlease provide a complete, valid hackathon project idea."
                state["final_result"] = "terminated"
            else:
                logger.info("✅ Idea validation passed!")
                self._log_verbose_step("STEP1_VALIDATE_IDEA", "Idea validation PASSED", {
                    "completeness_score": result.get("completeness_score"),
                    "clarity_score": result.get("clarity_score"),
                    "reason": result["reason"][:100] + "..."
                })
                
                # Update progress
                self._update_step_progress(state, "validate_idea", {
                    "status": "passed",
                    "scores": {
                        "completeness": result.get("completeness_score"),
                        "clarity": result.get("clarity_score")
                    }
                })
                
                st.success("✅ **Idea validation passed!** Proceeding to bounty matching...")
            
        except Exception as e:
            logger.error(f"❌ Validation error: {str(e)}")
            self._log_verbose_step("STEP1_VALIDATE_IDEA", f"ERROR occurred: {str(e)}", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            # Update progress
            self._update_step_progress(state, "validate_idea", {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            st.error(f"❌ **Validation Error**\n\nError validating idea: {str(e)}")
            state["is_valid_idea"] = False
            state["error_message"] = f"❌ **Validation Error**\n\nError validating idea: {str(e)}"
            state["final_result"] = "error"
        
        # Log step completion
        step_duration = time.time() - step_start_time
        self._log_verbose_step("STEP1_VALIDATE_IDEA", f"Step completed in {step_duration:.2f}s", {
            "duration": step_duration,
            "api_calls": state["api_call_count"],
            "final_status": "passed" if state["is_valid_idea"] else "failed"
        })
        
        return state
    
    def step2_find_similar_bounties(self, state: IdeaEvaluationState) -> IdeaEvaluationState:
        """
        Step 2a: Find similar bounties using vector similarity.
        """
        step_start_time = time.time()
        
        # Update state tracking
        state["current_step"] = "find_similar_bounties"
        
        if not state["is_valid_idea"]:
            logger.info("⏭️ Skipping bounty search - idea validation failed")
            self._log_verbose_step("STEP2_FIND_BOUNTIES", "Skipping - idea validation failed")
            return state
        
        user_idea = state["user_idea"]
        logger.info(f"🔍 STEP 2: Starting bounty search for: '{user_idea[:100]}...'")
        st.info("🔍 **Step 2**: Searching for similar bounties using vector similarity...")
        
        # Verbose logging
        self._log_verbose_step("STEP2_FIND_BOUNTIES", f"Starting bounty search for: '{user_idea[:100]}...'")
        self._log_verbose_step("STEP2_FIND_BOUNTIES", f"User idea length: {len(user_idea)} characters")
        
        # Update progress
        self._update_step_progress(state, "find_similar_bounties", {
            "status": "starting",
            "idea_length": len(user_idea)
        })
        
        try:
            # Search for similar bounties using FAISS with verbose logging
            logger.info("🔍 Searching FAISS vector store for similar bounties...")
            self._log_verbose_step("STEP2_FIND_BOUNTIES", "Searching FAISS vector store")
            
            # Update progress
            self._update_step_progress(state, "find_similar_bounties", {
                "status": "searching_vector_store",
                "query": user_idea[:100]
            })
            
            search_start = time.time()
            # Search for more bounties initially to allow for filtering
            all_similar_bounties = self.vector_store.search(
                query=user_idea,
                k=50,  # Increased from 10 to allow for filtering
                score_threshold=0.0
            )
            search_duration = time.time() - search_start
            
            # Filter bounties by selected events and companies
            selected_events = state.get("selected_events", [])
            selected_companies = state.get("selected_companies", [])
            
            filtered_bounties = []
            for bounty in all_similar_bounties:
                metadata = bounty.get('metadata', {})
                company = metadata.get('company', '')
                event_key = metadata.get('event_key', '')
                
                # Check if this bounty matches our selected criteria
                if selected_events and event_key not in selected_events:
                    continue
                if selected_companies and company not in selected_companies:
                    continue
                    
                filtered_bounties.append(bounty)
            
            # Use filtered results, fallback to all if no matches
            similar_bounties = filtered_bounties if filtered_bounties else all_similar_bounties[:10]
            
            logger.info(f"📊 Found {len(all_similar_bounties)} total bounties, {len(filtered_bounties)} from selected events/companies")
            self._log_verbose_step("STEP2_FIND_BOUNTIES", f"FAISS search completed in {search_duration:.2f}s", {
                "total_results": len(all_similar_bounties),
                "filtered_results": len(filtered_bounties),
                "final_results": len(similar_bounties),
                "search_time": search_duration,
                "vector_store_type": "FAISS",
                "selected_events": selected_events,
                "selected_companies": selected_companies
            })
            
            # Log each bounty with details
            self._log_verbose_step("STEP2_FIND_BOUNTIES", "Processing search results")
            
            for i, bounty in enumerate(similar_bounties):
                self._log_verbose_step("STEP2_FIND_BOUNTIES", f"Bounty {i+1} processed", {
                    "similarity_score": bounty["similarity_score"],
                    "title": bounty["metadata"].get('title', 'Untitled')[:50],
                    "content_length": len(bounty["content"]),
                    "metadata_keys": list(bounty["metadata"].keys()) if bounty["metadata"] else []
                })
                
                logger.info(f"  Bounty {i+1}: Similarity {bounty['similarity_score']:.3f} - {bounty['metadata'].get('title', 'Untitled')[:50]}...")
            
            # Log summary of results
            self._log_verbose_step("STEP2_FIND_BOUNTIES", "Results processing complete", {
                "total_bounties": len(similar_bounties),
                "similarity_scores": [b["similarity_score"] for b in similar_bounties],
                "avg_similarity": sum(b["similarity_score"] for b in similar_bounties) / len(similar_bounties) if similar_bounties else 0,
                "max_similarity": max(b["similarity_score"] for b in similar_bounties) if similar_bounties else 0,
                "min_similarity": min(b["similarity_score"] for b in similar_bounties) if similar_bounties else 0
            })
            
            state["similar_bounties"] = similar_bounties
            
            # Update progress
            self._update_step_progress(state, "find_similar_bounties", {
                "status": "completed",
                "bounties_found": len(similar_bounties),
                "avg_similarity": sum(b["similarity_score"] for b in similar_bounties) / len(similar_bounties) if similar_bounties else 0
            })
            
            st.success(f"✅ Found {len(similar_bounties)} similar bounties! Proceeding to relevance validation...")
            
        except Exception as e:
            logger.error(f"❌ Database error: {str(e)}")
            self._log_verbose_step("STEP2_FIND_BOUNTIES", f"ERROR occurred: {str(e)}", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            # Update progress
            self._update_step_progress(state, "find_similar_bounties", {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            st.error(f"❌ **Database Error**\n\nError searching for similar bounties: {str(e)}")
            state["error_message"] = f"❌ **Database Error**\n\nError searching for similar bounties: {str(e)}"
            state["final_result"] = "error"
        
        # Log step completion
        step_duration = time.time() - step_start_time
        self._log_verbose_step("STEP2_FIND_BOUNTIES", f"Step completed in {step_duration:.2f}s", {
            "duration": step_duration,
            "bounties_found": len(state.get("similar_bounties", [])),
            "final_status": "completed" if not state.get("error_message") else "error"
        })
        
        # Update step timing
        if "step_timings" not in state:
            state["step_timings"] = {}
        state["step_timings"]["find_similar_bounties"] = step_duration
        
        return state
    
    def step3_validate_relevance(self, state: IdeaEvaluationState) -> IdeaEvaluationState:
        """
        Step 2b: Validate if the similar bounties are actually relevant to the idea.
        """
        step_start_time = time.time()
        
        # Update state tracking
        state["current_step"] = "validate_relevance"
        
        if not state["is_valid_idea"] or not state["similar_bounties"]:
            logger.info("⏭️ Skipping relevance validation - no valid idea or bounties")
            self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Skipping - no valid idea or bounties", {
                "is_valid_idea": state.get("is_valid_idea", False),
                "similar_bounties_count": len(state.get("similar_bounties", []))
            })
            return state
        
        user_idea = state["user_idea"]
        similar_bounties = state["similar_bounties"]
        logger.info(f"🔍 STEP 3: Starting relevance validation for {len(similar_bounties)} bounties")
        st.info("🔍 **Step 3**: Validating relevance of found bounties using AI...")
        
        # Verbose logging
        self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"Starting relevance validation for {len(similar_bounties)} bounties")
        self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"User idea: '{user_idea[:100]}...'")
        
        # Update progress
        self._update_step_progress(state, "validate_relevance", {
            "status": "starting",
            "bounties_to_analyze": len(similar_bounties),
            "idea_length": len(user_idea)
        })
        
        # Prepare bounty descriptions for analysis with verbose logging
        bounty_descriptions = []
        self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Preparing bounty descriptions for AI analysis")
        
        for i, bounty in enumerate(similar_bounties[:5]):  # Analyze top 5
            bounty_desc = f"Bounty {i+1}: {bounty['content'][:200]}..."
            bounty_descriptions.append(bounty_desc)
            self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"Bounty {i+1} description prepared", {
                "bounty_index": i,
                "similarity_score": bounty.get("similarity_score", 0),
                "content_length": len(bounty['content']),
                "description_length": len(bounty_desc)
            })
        
        self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"Prepared {len(bounty_descriptions)} bounty descriptions")
        
        relevance_prompt = f"""
        You are an expert hackathon judge. Analyze if the following bounties are relevant to the user's idea.

        User's Idea: "{user_idea}"

        Similar Bounties Found:
        {chr(10).join(bounty_descriptions)}

        For each bounty, determine:
        1. Is it relevant to the user's idea?
        2. How strong is the relevance (0-10)?
        3. What specific aspects make it relevant or irrelevant?

        Respond with a JSON object:
        {{
            "relevant_bounties": [
                {{
                    "bounty_index": 0,
                    "is_relevant": true/false,
                    "relevance_score": 0-10,
                    "relevance_reason": "Why it's relevant or not"
                }}
            ],
            "overall_relevance": "Overall assessment of bounty relevance",
            "has_any_relevant": true/false
        }}
        """
        
        self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Relevance analysis prompt prepared", {
            "prompt_length": len(relevance_prompt),
            "bounties_in_prompt": len(bounty_descriptions)
        })
        
        try:
            # Update progress
            self._update_step_progress(state, "validate_relevance", {
                "status": "calling_openai",
                "prompt_length": len(relevance_prompt)
            })
            
            logger.info("🤖 Calling OpenAI API for relevance validation...")
            self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Initiating OpenAI API call for relevance analysis")
            
            # Make API call with timing
            api_start = time.time()
            response = self.llm.invoke([HumanMessage(content=relevance_prompt)])
            api_duration = time.time() - api_start
            
            # Log API call details
            self._log_api_call(
                "OpenAI_Relevance_Validation",
                {"prompt": relevance_prompt, "model": "gpt-4o-mini"},
                {"response": response.content},
                api_duration
            )
            
            # Parse response with robust error handling
            self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Parsing OpenAI response")
            result = self._parse_openai_response(response.content, "STEP3_VALIDATE_RELEVANCE")
            
            # Log detailed relevance analysis result
            self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Relevance analysis complete", {
                "total_bounties_analyzed": len(result.get("relevant_bounties", [])),
                "has_any_relevant": result.get("has_any_relevant", False),
                "overall_relevance_length": len(result.get("overall_relevance", ""))
            })
            
            # Update API call count
            state["api_call_count"] = state.get("api_call_count", 0) + 1
            
            # Filter relevant bounties with detailed logging
            relevant_bounties = []
            self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Filtering relevant bounties based on AI analysis")
            
            for bounty_analysis in result["relevant_bounties"]:
                bounty_index = bounty_analysis["bounty_index"]
                is_relevant = bounty_analysis["is_relevant"]
                relevance_score = bounty_analysis["relevance_score"]
                
                self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"Analyzing bounty {bounty_index+1}", {
                    "is_relevant": is_relevant,
                    "relevance_score": relevance_score,
                    "meets_threshold": is_relevant and relevance_score >= 5
                })
                
                if is_relevant and relevance_score >= 5:
                    if bounty_index < len(similar_bounties):
                        relevant_bounty = similar_bounties[bounty_index].copy()
                        relevant_bounty["relevance_score"] = bounty_analysis["relevance_score"]
                        relevant_bounty["relevance_reason"] = bounty_analysis["relevance_reason"]
                        relevant_bounties.append(relevant_bounty)
                        
                        self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"Bounty {bounty_index+1} marked as RELEVANT", {
                            "relevance_score": relevance_score,
                            "reason_preview": bounty_analysis["relevance_reason"][:100] + "..."
                        })
                        
                        logger.info(f"  ✅ Bounty {bounty_index+1}: Relevant (score: {relevance_score})")
                    else:
                        logger.warning(f"  ⚠️ Bounty index {bounty_index} out of range")
                        self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"Bounty index {bounty_index} out of range", {
                            "bounty_index": bounty_index,
                            "similar_bounties_count": len(similar_bounties)
                        })
                else:
                    self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"Bounty {bounty_index+1} marked as NOT RELEVANT", {
                        "is_relevant": is_relevant,
                        "relevance_score": relevance_score,
                        "reason": bounty_analysis["relevance_reason"][:100] + "..."
                    })
                    logger.info(f"  ❌ Bounty {bounty_index+1}: Not relevant (score: {relevance_score})")
            
            state["relevant_bounties"] = relevant_bounties
            state["relevance_analysis"] = result["overall_relevance"]
            
            # Log summary of relevance analysis
            self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Relevance analysis summary", {
                "total_analyzed": len(similar_bounties),
                "relevant_found": len(relevant_bounties),
                "relevance_rate": len(relevant_bounties) / len(similar_bounties) if similar_bounties else 0,
                "has_any_relevant": result.get("has_any_relevant", False)
            })
            
            logger.info(f"📊 Found {len(relevant_bounties)} relevant bounties out of {len(similar_bounties)} total")
            
            if not result["has_any_relevant"] or len(relevant_bounties) == 0:
                logger.warning("❌ No relevant bounties found")
                self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "No relevant bounties found", {
                    "has_any_relevant": result.get("has_any_relevant", False),
                    "relevant_bounties_count": len(relevant_bounties),
                    "overall_relevance": result.get("overall_relevance", "")
                })
                
                # Update progress
                self._update_step_progress(state, "validate_relevance", {
                    "status": "no_relevant_found",
                    "relevant_count": 0,
                    "overall_relevance": result.get("overall_relevance", "")
                })
                
                st.error(f"❌ **No Relevant Bounties Found**\n\n{result['overall_relevance']}\n\nUnfortunately, your idea doesn't relate to any of the available bounties. Try refining your idea or exploring different problem areas.")
                state["error_message"] = f"❌ **No Relevant Bounties Found**\n\n{result['overall_relevance']}\n\nUnfortunately, your idea doesn't relate to any of the available bounties. Try refining your idea or exploring different problem areas."
                state["final_result"] = "no_relevant_bounties"
            else:
                logger.info("✅ Found relevant bounties! Proceeding to metrics calculation...")
                self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", "Relevant bounties found, proceeding to next step", {
                    "relevant_count": len(relevant_bounties),
                    "relevance_scores": [b["relevance_score"] for b in relevant_bounties]
                })
                
                # Update progress
                self._update_step_progress(state, "validate_relevance", {
                    "status": "completed",
                    "relevant_count": len(relevant_bounties),
                    "relevance_scores": [b["relevance_score"] for b in relevant_bounties]
                })
                
                st.success(f"✅ Found {len(relevant_bounties)} relevant bounties! Proceeding to metrics calculation...")
            
        except Exception as e:
            logger.error(f"❌ Relevance analysis error: {str(e)}")
            self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"ERROR occurred: {str(e)}", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            # Update progress
            self._update_step_progress(state, "validate_relevance", {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            st.error(f"❌ **Relevance Analysis Error**\n\nError analyzing bounty relevance: {str(e)}")
            state["error_message"] = f"❌ **Relevance Analysis Error**\n\nError analyzing bounty relevance: {str(e)}"
            state["final_result"] = "error"
        
        # Log step completion
        step_duration = time.time() - step_start_time
        self._log_verbose_step("STEP3_VALIDATE_RELEVANCE", f"Step completed in {step_duration:.2f}s", {
            "duration": step_duration,
            "relevant_bounties_found": len(state.get("relevant_bounties", [])),
            "final_status": "completed" if not state.get("error_message") else "error"
        })
        
        # Update step timing
        if "step_timings" not in state:
            state["step_timings"] = {}
        state["step_timings"]["validate_relevance"] = step_duration
        
        return state
    
    def step4_calculate_metrics(self, state: IdeaEvaluationState) -> IdeaEvaluationState:
        """
        Step 3: Calculate metrics and provide recommendations.
        """
        step_start_time = time.time()
        
        # Update state tracking
        state["current_step"] = "calculate_metrics"
        
        if not state["is_valid_idea"] or not state["relevant_bounties"]:
            logger.info("⏭️ Skipping metrics calculation - no valid idea or relevant bounties")
            self._log_verbose_step("STEP4_CALCULATE_METRICS", "Skipping - no valid idea or relevant bounties", {
                "is_valid_idea": state.get("is_valid_idea", False),
                "relevant_bounties_count": len(state.get("relevant_bounties", []))
            })
            return state
        
        user_idea = state["user_idea"]
        relevant_bounties = state["relevant_bounties"]
        logger.info(f"🔍 STEP 4: Starting metrics calculation for {len(relevant_bounties)} relevant bounties")
        st.info("🔍 **Step 4**: Calculating comprehensive metrics and generating recommendations...")
        
        # Verbose logging
        self._log_verbose_step("STEP4_CALCULATE_METRICS", f"Starting metrics calculation for {len(relevant_bounties)} relevant bounties")
        self._log_verbose_step("STEP4_CALCULATE_METRICS", f"User idea: '{user_idea[:100]}...'")
        
        # Update progress
        self._update_step_progress(state, "calculate_metrics", {
            "status": "starting",
            "relevant_bounties_count": len(relevant_bounties),
            "idea_length": len(user_idea)
        })
        
        # Prepare bounty context for metrics calculation with verbose logging
        bounty_context = []
        self._log_verbose_step("STEP4_CALCULATE_METRICS", "Preparing bounty context for metrics calculation")
        
        for i, bounty in enumerate(relevant_bounties[:3]):  # Use top 3 most relevant
            bounty_text = f"Bounty {i+1}: {bounty['content']}"
            bounty_context.append(bounty_text)
            self._log_verbose_step("STEP4_CALCULATE_METRICS", f"Bounty {i+1} context prepared", {
                "bounty_index": i,
                "relevance_score": bounty.get("relevance_score", 0),
                "content_length": len(bounty['content']),
                "context_length": len(bounty_text)
            })
        
        self._log_verbose_step("STEP4_CALCULATE_METRICS", f"Prepared {len(bounty_context)} bounty contexts")
        
        metrics_prompt = f"""
        You are an expert hackathon judge. Calculate comprehensive metrics for this hackathon idea based on the relevant bounties.

        User's Idea: "{user_idea}"

        Relevant Bounties:
        {chr(10).join(bounty_context)}

        Calculate the following metrics (0-10 scale):
        1. Problem Significance: How important is the problem being solved?
        2. Novelty/Uniqueness: How innovative is the solution?
        3. Technical Feasibility: How realistic is the implementation?
        4. Market Potential: How viable is the business model?
        5. Crypto-Nativeness: How well does it leverage blockchain/crypto?
        6. User Value: How much value does it provide to users?
        7. Scalability: How well can it scale?
        8. Team Readiness: How ready is the team to execute?

        Also provide:
        - Overall Score (0-100)
        - Strengths (top 3)
        - Weaknesses (top 3)
        - Recommendations (3-5 actionable items)

        Respond with a JSON object:
        {{
            "metrics": {{
                "problem_significance": 0-10,
                "novelty_uniqueness": 0-10,
                "technical_feasibility": 0-10,
                "market_potential": 0-10,
                "crypto_nativeness": 0-10,
                "user_value": 0-10,
                "scalability": 0-10,
                "team_readiness": 0-10
            }},
            "overall_score": 0-100,
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2", "weakness3"],
            "recommendations": ["rec1", "rec2", "rec3", "rec4", "rec5"],
            "summary": "Overall assessment and next steps"
        }}
        """
        
        self._log_verbose_step("STEP4_CALCULATE_METRICS", "Metrics calculation prompt prepared", {
            "prompt_length": len(metrics_prompt),
            "bounties_in_prompt": len(bounty_context)
        })
        
        try:
            # Update progress
            self._update_step_progress(state, "calculate_metrics", {
                "status": "calling_openai",
                "prompt_length": len(metrics_prompt)
            })
            
            logger.info("🤖 Calling OpenAI API for metrics calculation...")
            self._log_verbose_step("STEP4_CALCULATE_METRICS", "Initiating OpenAI API call for metrics calculation")
            
            # Make API call with timing
            api_start = time.time()
            response = self.llm.invoke([HumanMessage(content=metrics_prompt)])
            api_duration = time.time() - api_start
            
            # Log API call details
            self._log_api_call(
                "OpenAI_Metrics_Calculation",
                {"prompt": metrics_prompt, "model": "gpt-4o-mini"},
                {"response": response.content},
                api_duration
            )
            
            # Parse response with robust error handling
            self._log_verbose_step("STEP4_CALCULATE_METRICS", "Parsing OpenAI response")
            result = self._parse_openai_response(response.content, "STEP4_CALCULATE_METRICS")
            
            # Log detailed metrics calculation result
            self._log_verbose_step("STEP4_CALCULATE_METRICS", "Metrics calculation complete", {
                "overall_score": result.get("overall_score", 0),
                "metrics_count": len(result.get("metrics", {})),
                "strengths_count": len(result.get("strengths", [])),
                "weaknesses_count": len(result.get("weaknesses", [])),
                "recommendations_count": len(result.get("recommendations", [])),
                "summary_length": len(result.get("summary", ""))
            })
            
            # Log individual metrics
            if "metrics" in result:
                self._log_verbose_step("STEP4_CALCULATE_METRICS", "Individual metrics calculated", result["metrics"])
            
            # Update API call count
            state["api_call_count"] = state.get("api_call_count", 0) + 1
            
            # Update state
            state["metrics"] = result
            state["recommendations"] = result["recommendations"]
            
            logger.info(f"📊 Metrics calculation result: {result}")
            
            # Update progress
            self._update_step_progress(state, "calculate_metrics", {
                "status": "formatting_result",
                "overall_score": result.get("overall_score", 0),
                "metrics_calculated": len(result.get("metrics", {}))
            })
            
            logger.info("📝 Formatting final result...")
            self._log_verbose_step("STEP4_CALCULATE_METRICS", "Formatting final evaluation result")
            
            # Format final result
            final_result = self._format_final_result(result, relevant_bounties)
            state["final_result"] = final_result
            
            self._log_verbose_step("STEP4_CALCULATE_METRICS", "Final result formatted", {
                "result_length": len(final_result),
                "overall_score": result.get("overall_score", 0)
            })
            
            logger.info("✅ LangGraph evaluation completed successfully!")
            self._log_verbose_step("STEP4_CALCULATE_METRICS", "LangGraph evaluation completed successfully", {
                "total_steps": 4,
                "api_calls_made": state.get("api_call_count", 0),
                "overall_score": result.get("overall_score", 0)
            })
            
            # Update progress
            self._update_step_progress(state, "calculate_metrics", {
                "status": "completed",
                "overall_score": result.get("overall_score", 0),
                "final_result_length": len(final_result)
            })
            
            st.success("✅ **Evaluation Complete!** All steps finished successfully.")
            
        except Exception as e:
            logger.error(f"❌ Metrics calculation error: {str(e)}")
            self._log_verbose_step("STEP4_CALCULATE_METRICS", f"ERROR occurred: {str(e)}", {
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            # Update progress
            self._update_step_progress(state, "calculate_metrics", {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            st.error(f"❌ **Metrics Calculation Error**\n\nError calculating metrics: {str(e)}")
            state["error_message"] = f"❌ **Metrics Calculation Error**\n\nError calculating metrics: {str(e)}"
            state["final_result"] = "error"
        
        # Log step completion
        step_duration = time.time() - step_start_time
        self._log_verbose_step("STEP4_CALCULATE_METRICS", f"Step completed in {step_duration:.2f}s", {
            "duration": step_duration,
            "overall_score": result.get("overall_score", 0) if "result" in locals() else "N/A",
            "final_status": "completed" if not state.get("error_message") else "error"
        })
        
        # Update step timing
        if "step_timings" not in state:
            state["step_timings"] = {}
        state["step_timings"]["calculate_metrics"] = step_duration
        
        return state
    
    def _format_final_result(self, metrics: Dict[str, Any], relevant_bounties: List[Dict[str, Any]]) -> str:
        """Format the final evaluation result."""
        result = f"""
## 🎯 **Idea Evaluation Results**

### 📊 **Overall Score: {metrics['overall_score']}/100**

### 📈 **Detailed Metrics**
"""
        
        # Add metrics
        for metric, score in metrics["metrics"].items():
            emoji = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
            formatted_name = metric.replace("_", " ").title()
            result += f"- **{emoji} {formatted_name}**: {score}/10\n"
        
        # Add strengths
        result += f"\n### ✅ **Strengths**\n"
        for strength in metrics["strengths"]:
            result += f"- {strength}\n"
        
        # Add weaknesses
        result += f"\n### ⚠️ **Areas for Improvement**\n"
        for weakness in metrics["weaknesses"]:
            result += f"- {weakness}\n"
        
        # Add recommendations
        result += f"\n### 💡 **Recommendations**\n"
        for i, rec in enumerate(metrics["recommendations"], 1):
            result += f"{i}. {rec}\n"
        
        # Add relevant bounties
        result += f"\n### 🎯 **Relevant Bounties Found**\n"
        for i, bounty in enumerate(relevant_bounties[:3], 1):
            result += f"**{i}. {bounty['metadata'].get('title', 'Untitled Bounty')}**\n"
            result += f"   - Relevance: {bounty['relevance_score']}/10\n"
            result += f"   - Why relevant: {bounty['relevance_reason']}\n\n"
        
        # Add summary
        result += f"\n### 📝 **Summary**\n{metrics['summary']}\n"
        
        return result
    
    def create_evaluation_flow(self) -> StateGraph:
        """Create the LangGraph evaluation flow."""
        workflow = StateGraph(IdeaEvaluationState)
        
        # Add nodes
        workflow.add_node("validate_idea", self.step1_validate_idea)
        workflow.add_node("find_similar_bounties", self.step2_find_similar_bounties)
        workflow.add_node("validate_relevance", self.step3_validate_relevance)
        workflow.add_node("calculate_metrics", self.step4_calculate_metrics)
        
        # Add edges
        workflow.add_edge("validate_idea", "find_similar_bounties")
        workflow.add_edge("find_similar_bounties", "validate_relevance")
        workflow.add_edge("validate_relevance", "calculate_metrics")
        workflow.add_edge("calculate_metrics", END)
        
        # Set entry point
        workflow.set_entry_point("validate_idea")
        
        return workflow
    
    def evaluate_idea(self, user_idea: str, selected_companies: list = None, selected_bounties: list = None, selected_events: list = None) -> str:
        """
        Main method to evaluate a hackathon idea using the LangGraph flow, focused on selected bounties.
        """
        evaluation_start_time = time.time()
        
        logger.info(f"🚀 Starting LangGraph evaluation for idea: '{user_idea[:100]}...'")
        logger.info(f"🎯 Focused on selected companies: {selected_companies}")
        logger.info(f"🎯 Focused on selected bounties: {len(selected_bounties) if selected_bounties else 0}")
        st.info("🚀 **Starting LangGraph Evaluation** - Multi-step AI analysis in progress...")
        
        # Verbose logging for evaluation start
        self._log_verbose_step("EVALUATION_START", f"Starting LangGraph evaluation for idea: '{user_idea[:100]}...'", {
            "idea_length": len(user_idea),
            "word_count": len(user_idea.split()),
            "selected_companies": selected_companies,
            "selected_bounties_count": len(selected_bounties) if selected_bounties else 0,
            "selected_events": selected_events,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create the workflow with verbose logging
        logger.info("🔧 Creating LangGraph workflow...")
        self._log_verbose_step("EVALUATION_START", "Creating LangGraph workflow")
        
        workflow_start = time.time()
        workflow = self.create_evaluation_flow()
        app = workflow.compile()
        workflow_duration = time.time() - workflow_start
        
        logger.info("✅ LangGraph workflow created and compiled")
        self._log_verbose_step("EVALUATION_START", f"LangGraph workflow created and compiled in {workflow_duration:.2f}s", {
            "workflow_creation_time": workflow_duration,
            "nodes_count": 4,
            "edges_count": 4
        })
        
        # Initial state with verbose tracking and selected context
        initial_state = {
            "user_idea": user_idea,
            "selected_companies": selected_companies or [],
            "selected_bounties": selected_bounties or [],
            "selected_events": selected_events or [],
            "is_valid_idea": False,
            "validation_reason": "",
            "similar_bounties": [],
            "relevant_bounties": [],
            "relevance_analysis": "",
            "metrics": {},
            "recommendations": [],
            "final_result": "",
            "error_message": None,
            # Verbose tracking fields
            "step_timings": {},
            "api_call_count": 0,
            "total_tokens_used": 0,
            "current_step": "initialization",
            "step_progress": {}
        }
        
        self._log_verbose_step("EVALUATION_START", "Initial state prepared", {
            "state_fields": list(initial_state.keys()),
            "idea_length": len(user_idea)
        })
        
        try:
            # Run the workflow with verbose logging
            logger.info("🔄 Executing LangGraph workflow...")
            self._log_verbose_step("EVALUATION_EXECUTION", "Starting LangGraph workflow execution")
            
            execution_start = time.time()
            result = app.invoke(initial_state)
            execution_duration = time.time() - execution_start
            
            logger.info("✅ LangGraph workflow execution completed")
            self._log_verbose_step("EVALUATION_EXECUTION", f"LangGraph workflow execution completed in {execution_duration:.2f}s", {
                "execution_time": execution_duration,
                "api_calls_made": result.get("api_call_count", 0),
                "steps_completed": len(result.get("step_timings", {})),
                "final_step": result.get("current_step", "unknown")
            })
            
            # Log comprehensive evaluation summary
            total_evaluation_time = time.time() - evaluation_start_time
            self._log_verbose_step("EVALUATION_SUMMARY", "Comprehensive evaluation summary", {
                "total_evaluation_time": total_evaluation_time,
                "workflow_creation_time": workflow_duration,
                "execution_time": execution_duration,
                "api_calls_made": result.get("api_call_count", 0),
                "steps_completed": len(result.get("step_timings", {})),
                "step_timings": result.get("step_timings", {}),
                "is_valid_idea": result.get("is_valid_idea", False),
                "similar_bounties_found": len(result.get("similar_bounties", [])),
                "relevant_bounties_found": len(result.get("relevant_bounties", [])),
                "overall_score": result.get("metrics", {}).get("overall_score", "N/A"),
                "has_error": bool(result.get("error_message")),
                "final_status": "success" if not result.get("error_message") else "error"
            })
            
            # Return the final result or error message
            if result.get("error_message"):
                logger.warning(f"⚠️ Evaluation completed with error: {result['error_message']}")
                self._log_verbose_step("EVALUATION_RESULT", "Evaluation completed with error", {
                    "error_message": result["error_message"],
                    "total_time": total_evaluation_time
                })
                return result["error_message"]
            else:
                logger.info("🎉 Evaluation completed successfully!")
                self._log_verbose_step("EVALUATION_RESULT", "Evaluation completed successfully", {
                    "overall_score": result.get("metrics", {}).get("overall_score", "N/A"),
                    "relevant_bounties": len(result.get("relevant_bounties", [])),
                    "total_time": total_evaluation_time
                })
                return result["final_result"]
                
        except Exception as e:
            logger.error(f"❌ LangGraph execution error: {str(e)}")
            self._log_verbose_step("EVALUATION_ERROR", f"LangGraph execution error: {str(e)}", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "total_time": time.time() - evaluation_start_time
            })
            
            st.error(f"❌ **LangGraph Execution Error**\n\nAn error occurred during evaluation: {str(e)}")
            return f"❌ **Evaluation Error**\n\nAn error occurred during evaluation: {str(e)}"
    
    def get_collection_count(self) -> int:
        """Get the number of bounties in the vector store."""
        try:
            stats = self.vector_store.get_stats()
            return stats.get("total_documents", 0)
        except Exception as e:
            st.error(f"Error getting vector store count: {str(e)}")
            return 0
    
    def get_verbose_logs(self) -> str:
        """Get a formatted summary of verbose logs for display."""
        # This would typically read from the log file or return cached logs
        # For now, return a placeholder that indicates verbose logging is active
        return """
## 🔍 **Verbose Logging Active**

Your LangGraph evaluation is now running with **FULL VERBOSE LOGGING** enabled! 

### What's being logged:
- ✅ **Step-by-step progress** with timestamps
- ✅ **API call details** (requests, responses, timing)
- ✅ **Database operations** (queries, results, performance)
- ✅ **State transitions** and intermediate results
- ✅ **Performance metrics** for each operation
- ✅ **Error details** with full context
- ✅ **Comprehensive evaluation summary**

### Log locations:
- **Console output**: Real-time verbose logs in terminal
- **Log file**: `langgraph_evaluator.log` (detailed logs)
- **Streamlit UI**: Progress updates and status messages

### Verbose features enabled:
- 🔍 **Microsecond timestamps** for all operations
- 📊 **Performance timing** for each step
- 🌐 **API call logging** with request/response details
- 📈 **Progress tracking** with detailed status updates
- 🗄️ **Database operation logging** with query details
- ⚡ **Real-time state monitoring** throughout the flow

The evaluation will now provide complete visibility into every operation!
        """
    
    def print_verbose_summary(self, state: Dict[str, Any]) -> None:
        """Print a comprehensive verbose summary of the evaluation."""
        print("\n" + "="*80)
        print("🔍 LANGGRAPH VERBOSE EVALUATION SUMMARY")
        print("="*80)
        
        # Basic info
        print(f"📝 User Idea: {state.get('user_idea', 'N/A')[:100]}...")
        print(f"⏱️  Total Steps: {len(state.get('step_timings', {}))}")
        print(f"🌐 API Calls Made: {state.get('api_call_count', 0)}")
        print(f"✅ Idea Valid: {state.get('is_valid_idea', False)}")
        
        # Step timings
        if state.get('step_timings'):
            print(f"\n⏱️  STEP TIMINGS:")
            for step, duration in state['step_timings'].items():
                print(f"   {step}: {duration:.2f}s")
        
        # Bounty results
        print(f"\n🎯 BOUNTY RESULTS:")
        print(f"   Similar bounties found: {len(state.get('similar_bounties', []))}")
        print(f"   Relevant bounties found: {len(state.get('relevant_bounties', []))}")
        
        # Metrics
        if state.get('metrics'):
            metrics = state['metrics']
            print(f"\n📊 EVALUATION METRICS:")
            print(f"   Overall Score: {metrics.get('overall_score', 'N/A')}/100")
            if 'metrics' in metrics:
                for metric, score in metrics['metrics'].items():
                    print(f"   {metric.replace('_', ' ').title()}: {score}/10")
        
        # Progress tracking
        if state.get('step_progress'):
            print(f"\n📈 PROGRESS TRACKING:")
            for step, progress in state['step_progress'].items():
                status = progress.get('progress', {}).get('status', 'unknown')
                print(f"   {step}: {status}")
        
        print("="*80)
        print("✅ VERBOSE EVALUATION COMPLETE")
        print("="*80 + "\n")
