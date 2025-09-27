"""
Simplified LangGraph-based intelligent hackathon idea evaluator.
"""

import os
import time
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import json
from faiss_vector_store import FAISSVectorStore
import streamlit as st
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IdeaEvaluationState(TypedDict):
    user_idea: str
    selected_companies: List[str]
    selected_bounties: List[str]
    selected_event: Optional[str]
    similar_bounties: List[Dict[str, Any]]
    bounty_descriptions: str
    final_result: str
    error_message: Optional[str]

class LangGraphIdeaEvaluator:

    def __init__(self, anthropic_api_key: str = None):
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
        
        if not self.anthropic_api_key or self.anthropic_api_key == 'your_anthropic_api_key_here':
            raise ValueError("Anthropic API key not provided. Please set ANTHROPIC_API_KEY environment variable.")
        
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            api_key=self.anthropic_api_key
        )
        
        self.vector_store = FAISSVectorStore(
            index_path="./faiss_index",
            embedding_model='all-MiniLM-L6-v2'
        )
        
        logger.info("✅ LangGraph evaluator initialized successfully")

    def step1_find_similar_bounties(self, state: IdeaEvaluationState) -> IdeaEvaluationState:
        """
        Step 1: Find similar bounties using vector similarity search, focused on selected bounties.
        """
        step_start_time = time.time()
        user_idea = state["user_idea"]
        selected_companies = state.get("selected_companies", [])
        selected_bounties = state.get("selected_bounties", [])
        selected_events = state.get("selected_events", [])
        
        logger.info(f"🔍 STEP 1: Finding similar bounties for: '{user_idea[:100]}...'")
        logger.info(f"🎯 Focusing on selected companies: {selected_companies}")
        logger.info(f"🎯 Focusing on selected bounties: {len(selected_bounties)}")
        
        try:
            # First try filtered search within selected criteria
            results = self.vector_store.search_filtered(
                query=user_idea,
                k=50,
                score_threshold=0.1,
                event_keys=selected_events,
                companies=selected_companies,
                bounty_ids=selected_bounties
            )
            
            logger.info(f"🎯 Found {len(results)} bounties from selected criteria")
            
            # If no results from filtered search, try unfiltered search as fallback
            if not results:
                logger.warning("⚠️ No bounties found from selected criteria, trying unfiltered search")
                results = self.vector_store.search(user_idea, k=50, score_threshold=0.0)
                logger.info(f"🔄 Fallback search found {len(results)} bounties")
            
            # If still no results, try with even lower threshold
            if not results:
                logger.warning("⚠️ No bounties found with normal threshold, trying very low threshold")
                results = self.vector_store.search(user_idea, k=50, score_threshold=0.0)
                logger.info(f"🔄 Low threshold search found {len(results)} bounties")
            
            # Final fallback - if still no results, return error
            if not results:
                state["error_message"] = "No similar bounties found. Please try a different idea or check if data is loaded."
                logger.warning("No similar bounties found even with fallback searches")
                return state
            
            # Filter results based on similarity scores
            high_similarity = [r for r in results if r['similarity_score'] > 0.7]
            medium_similarity = [r for r in results if 0.4 <= r['similarity_score'] <= 0.7]
            low_similarity = [r for r in results if r['similarity_score'] < 0.4]
            
            # Ensure we always get at least 3 results
            selected_results = []
            
            if high_similarity:
                # Take top 3-5 high similarity results
                selected_results = high_similarity[:min(5, len(high_similarity))]
                logger.info(f"Selected {len(selected_results)} high similarity results")
                
                # If we need more results, add from medium similarity
                if len(selected_results) < 3 and medium_similarity:
                    needed = 3 - len(selected_results)
                    additional = medium_similarity[:needed]
                    selected_results.extend(additional)
                    logger.info(f"Added {len(additional)} medium similarity results to reach minimum")
                    
            elif medium_similarity:
                # Take top 5-7 medium similarity results
                selected_results = medium_similarity[:min(7, len(medium_similarity))]
                logger.info(f"Selected {len(selected_results)} medium similarity results")
                
                # If we need more results, add from low similarity
                if len(selected_results) < 3 and low_similarity:
                    needed = 3 - len(selected_results)
                    additional = low_similarity[:needed]
                    selected_results.extend(additional)
                    logger.info(f"Added {len(additional)} low similarity results to reach minimum")
                    
            else:
                # Take top 8-10 low similarity results
                selected_results = low_similarity[:min(10, len(low_similarity))]
                logger.info(f"Selected {len(selected_results)} low similarity results")
            
            # Final fallback - if we still don't have enough, take the best available
            if len(selected_results) < 3 and len(results) >= 3:
                selected_results = results[:3]
                logger.info(f"Using top 3 results as final fallback")
            
            # Ensure we have at least 3 results
            if len(selected_results) < 3:
                logger.warning(f"⚠️ Only found {len(selected_results)} results, less than minimum of 3")
            
            state["similar_bounties"] = selected_results
            
            # Create bounty descriptions text for the LLM
            bounty_descriptions = []
            for i, result in enumerate(selected_results, 1):
                metadata = result['metadata']
                content = result['content']
                similarity = result['similarity_score']
                
                bounty_desc = f"""
Bounty {i} (Similarity: {similarity:.2f}):
Company: {metadata.get('company', 'Unknown')}
Title: {metadata.get('title', 'No Title')}
Description: {content}
Prizes: {metadata.get('prizes', 'Not specified')}
"""
                bounty_descriptions.append(bounty_desc.strip())
            
            state["bounty_descriptions"] = "\n\n".join(bounty_descriptions)
            
            
            logger.info(f"Found {len(selected_results)} similar bounties")
            
            # Show the actual bounties found
            if selected_results:
                st.success(f"✅ Found {len(selected_results)} similar bounties:")
                for i, result in enumerate(selected_results[:3], 1):  # Show top 3
                    metadata = result['metadata']
                    title = metadata.get('title', 'Untitled')
                    company = metadata.get('company', 'Unknown')
                    similarity = result['similarity_score']
                    st.write(f"**{i}.** {title} ({company}) - Similarity: {similarity:.2f}")
            else:
                st.warning("No similar bounties found")
            
        except Exception as e:
            error_msg = f"Error finding similar bounties: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ **Error**: {error_msg}")
            state["error_message"] = error_msg
        
        step_time = time.time() - step_start_time
        logger.info(f"⏱️ STEP 1 TIMING: {step_time:.2f}s")
        
        return state

    def step2_llm_evaluation(self, state: IdeaEvaluationState) -> IdeaEvaluationState:
        """
        Step 2: LLM evaluation using the provided prompt template.
        """
        step_start_time = time.time()
        user_idea = state["user_idea"]
        bounty_descriptions = state["bounty_descriptions"]
        
        logger.info("🧠 STEP 2: Running LLM evaluation...")
        st.info("🧠 **AI is running intelligent multi-step evaluation...**")
        
        try:
            # Get selected companies and bounties for context
            selected_companies = state.get("selected_companies", [])
            selected_bounties = state.get("selected_bounties", [])
            selected_event = state.get("selected_event", "Unknown Event")
            
            # Use the provided prompt template with selected context
            evaluation_prompt = f"""
You are an expert evaluator for web3/blockchain hackathon projects. Your role is to provide critical, constructive, and realistic feedback to help participants understand how their ideas measure against bounty requirements and market standards.

**EVALUATION CONTEXT:**
- **Event**: {selected_event}
- **Selected Companies**: {', '.join(selected_companies) if selected_companies else 'All companies'}
- **Selected Bounties**: {len(selected_bounties)} bounties selected
- **Focus**: Evaluate specifically against the user's selected companies and bounties

Here is the hackathon project idea you need to evaluate:

<hackathon_idea>
{user_idea}
</hackathon_idea>

Here are the bounty descriptions from the user's selected companies for this evaluation:

<bounty_description>
{bounty_descriptions}
</bounty_description>

## Your Evaluation Process

Follow this three-step validation process:

**Step 1: Initial Validation**
Determine if the submission is a legitimate hackathon project idea. If the text contains spam, random characters, is completely off-topic, or does not describe a project concept, respond with: "This submission does not appear to be a valid hackathon project idea. Please provide a clear description of your project concept, technical approach, and intended outcomes."

**Step 2: Bounty Alignment Assessment** 
Analyze whether the idea genuinely aligns with the bounty requirements. Use your judgment to assess fit - even surface-level similarities require evaluation of core concept alignment. If the idea doesn't fit the bounty:
- Acknowledge any potential merit in the idea
- Clearly explain why it may not be eligible for this specific bounty  
- Suggest 2-3 specific pivots or enhancements that could make it bounty-eligible

**Step 3: Comprehensive Evaluation**
If the idea passes the first two steps, evaluate it thoroughly across all 12 metrics below.

## Evaluation Metrics

Score the project on these 12 metrics (0-10 each):

1. **Problem Significance** - How important and well-defined is the problem being solved?
2. **Novelty/Uniqueness** - How innovative is the solution compared to existing alternatives?
3. **Technical Feasibility** - How realistic is implementation given typical hackathon constraints?
4. **Market Potential** - How viable is the business model and go-to-market strategy?
5. **Crypto-Nativeness** - How effectively does it leverage blockchain/crypto technologies?
6. **User Value** - How much tangible value does it provide to end users?
7. **Scalability** - How well can the solution handle growth in users, transactions, or complexity?
8. **Team Readiness** - Based on the idea presentation, how prepared does the team seem to execute?
9. **Implementation Quality** - How well-thought-out are the technical architecture and development approach?
10. **Community Impact** - How significantly could this project benefit the broader crypto/web3 ecosystem?
11. **Sustainability/Tokenomics** - How sound are the economic incentives and long-term viability mechanisms?
12. **Presentation Clarity** - How clearly and convincingly is the idea communicated?

## Instructions

Provide your response in this exact format:

## EVALUATION SUMMARY
• **Bounty Alignment:** [Eligible/Not Eligible/Partially Eligible]
• **Brief Assessment:** [2-3 sentences summarizing the idea's viability, including a creative simile]

## DETAILED METRIC SCORES

| Metric | Score | Key Reasoning |
|--------|-------|---------------|
| Problem Significance | X/10 | [Brief reasoning] |
| Novelty/Uniqueness | X/10 | [Brief reasoning] |
| Technical Feasibility | X/10 | [Brief reasoning] |
| Market Potential | X/10 | [Brief reasoning] |
| Crypto-Nativeness | X/10 | [Brief reasoning] |
| User Value | X/10 | [Brief reasoning] |
| Scalability | X/10 | [Brief reasoning] |
| Team Readiness | X/10 | [Brief reasoning] |
| Implementation Quality | X/10 | [Brief reasoning] |
| Community Impact | X/10 | [Brief reasoning] |
| Sustainability/Tokenomics | X/10 | [Brief reasoning] |
| Presentation Clarity | X/10 | [Brief reasoning] |

**Overall Score: X/120**

## FEEDBACK SUMMARY

**Key Strengths:**
• [Strength with evidence and simile]
• [Strength with evidence and simile]
• [Additional strengths as appropriate]

**Critical Weaknesses:**
• [Weakness with evidence and simile]  
• [Weakness with evidence and simile]
• [Additional weaknesses as appropriate]

**Actionable Recommendations:**
• [Specific, implementable suggestion]
• [Specific, implementable suggestion] 
• [Specific, implementable suggestion]
• [Additional recommendations as needed]

**Reality Check Summary:**
[A concise paragraph providing honest assessment of the project's chances, using vivid similes to illustrate both opportunities and challenges while considering hackathon timelines and resource constraints]

Example format reference:
- Use bullet points for clarity and brevity
- Include creative similes throughout (e.g., "This project is like a Swiss Army knife - versatile but lacks focus")
- Focus on specific evidence from the submission
- Consider realistic hackathon implementation constraints
- Distinguish between genuine innovation and buzzword usage
"""
            
            # Call Anthropic with streaming
            logger.info("Calling Anthropic Claude for evaluation...")
            
            # Create a placeholder for streaming response
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in self.llm.stream([HumanMessage(content=evaluation_prompt)]):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response)
            
            evaluation_result = full_response.strip()
            
            # Check if response is empty or too short
            if not evaluation_result or len(evaluation_result.strip()) < 50:
                logger.warning("⚠️ LLM response is empty or too short, using fallback")
                evaluation_result = f"""
## 🎯 **Idea Evaluation Results**

### ⚠️ **Evaluation Issue**
The AI evaluation completed but returned an incomplete response. This might be due to:
- API rate limiting
- Response processing issues
- Network connectivity problems

### 📊 **Your Idea**
{user_idea}

### 🔍 **Similar Bounties Found**
{len(state.get('similar_bounties', []))} similar bounties were identified and analyzed.

### 💡 **Next Steps**
Please try running the evaluation again. If the issue persists, check your API key and network connection.

---
*This is a fallback response due to incomplete AI evaluation.*
"""
            
            state["final_result"] = evaluation_result
            
            logger.info("✅ Anthropic evaluation completed successfully")
            
        except Exception as e:
            error_msg = f"Error in LLM evaluation: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ **Evaluation Error**: {error_msg}")
            state["error_message"] = error_msg
            state["final_result"] = f"❌ **Evaluation Error**: {error_msg}"
        
        step_time = time.time() - step_start_time
        logger.info(f"⏱️ STEP 2 TIMING: {step_time:.2f}s")
        
        return state

    def create_evaluation_flow(self) -> StateGraph:
        """Create the simplified 2-step LangGraph evaluation flow."""
        workflow = StateGraph(IdeaEvaluationState)
        
        # Add nodes
        workflow.add_node("find_similar_bounties", self.step1_find_similar_bounties)
        workflow.add_node("llm_evaluation", self.step2_llm_evaluation)
        
        # Add conditional edge - only proceed to LLM evaluation if no error
        def should_evaluate(state: IdeaEvaluationState) -> str:
            if state.get("error_message"):
                return "end"
            return "llm_evaluation"
        
        workflow.add_conditional_edges(
            "find_similar_bounties",
            should_evaluate,
            {
                "llm_evaluation": "llm_evaluation",
                "end": END
            }
        )
        workflow.add_edge("llm_evaluation", END)
        
        # Set entry point
        workflow.set_entry_point("find_similar_bounties")
        
        return workflow

    def evaluate_idea(self, user_idea: str, selected_companies: list = None, selected_bounties: list = None, selected_events: list = None) -> str:
        """
        Main evaluation function with focus on selected bounties.
        """
        logger.info(f"🚀 Starting evaluation for idea: '{user_idea[:100]}...'")
        logger.info(f"🎯 Focused on selected companies: {selected_companies}")
        logger.info(f"🎯 Focused on selected bounties: {len(selected_bounties) if selected_bounties else 0}")
        
        # Initialize state
        initial_state = {
            "user_idea": user_idea,
            "selected_companies": selected_companies or [],
            "selected_bounties": selected_bounties or [],
            "selected_events": selected_events or [],
            "similar_bounties": [],
            "bounty_descriptions": "",
            "final_result": "",
            "error_message": None
        }
        
        try:
            # Create and run the workflow
            workflow = self.create_evaluation_flow()
            app = workflow.compile()
            
            # Run the evaluation
            final_state = app.invoke(initial_state)
            
            
            # Return the final result
            if final_state.get("error_message"):
                return f"❌ **Error**: {final_state['error_message']}"
            
            return final_state.get("final_result", "No evaluation result generated.")
            
        except Exception as e:
            error_msg = f"Error running evaluation workflow: {str(e)}"
            logger.error(error_msg)
            return f"❌ **Workflow Error**: {error_msg}"

    def get_collection_count(self) -> int:
        """Get the number of documents in the FAISS store."""
        try:
            stats = self.vector_store.get_stats()
            return stats['total_documents']
        except Exception as e:
            logger.error(f"Error getting collection count: {str(e)}")
            return 0
