#!/usr/bin/env python3
"""
Test script for RAG evaluator functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_evaluator import RAGEvaluator
import json

def test_rag_evaluator():
    """Test the RAG evaluator with sample data"""
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Test input validation
    print("Testing input validation...")
    
    # Test valid input
    is_valid, message = evaluator.validate_input("I'm building a decentralized voting system using zero-knowledge proofs")
    print(f"Valid input test: {is_valid} - {message}")
    
    # Test invalid input (question)
    is_valid, message = evaluator.validate_input("What is a hackathon?")
    print(f"Question input test: {is_valid} - {message}")
    
    # Test invalid input (too short)
    is_valid, message = evaluator.validate_input("Voting app")
    print(f"Short input test: {is_valid} - {message}")
    
    # Test invalid input (chit-chat)
    is_valid, message = evaluator.validate_input("Hello there!")
    print(f"Chat input test: {is_valid} - {message}")
    
    print("\nInput validation tests completed!")
    
    # Test evaluation criteria
    print("\nTesting evaluation criteria...")
    for criterion, config in evaluator.evaluation_criteria.items():
        print(f"- {criterion}: {config['description']}")
    
    print("\nRAG evaluator test completed successfully!")

if __name__ == "__main__":
    test_rag_evaluator()
