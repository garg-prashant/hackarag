#!/usr/bin/env python3
"""
Test script for the Hackathon Idea Evaluator app
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import HackathonEvaluator

def test_evaluator():
    """Test the HackathonEvaluator class"""
    print("🧪 Testing HackathonEvaluator...")
    
    evaluator = HackathonEvaluator()
    
    # Test URL
    test_url = "https://example.com/test-bounty"
    
    # Test URL hash generation
    url_hash = evaluator.get_url_hash(test_url)
    print(f"✅ URL hash generated: {url_hash}")
    
    # Test company name extraction
    company_name = evaluator.extract_company_name(test_url)
    print(f"✅ Company name extracted: {company_name}")
    
    # Test data directory creation
    if os.path.exists(evaluator.data_dir):
        print(f"✅ Data directory exists: {evaluator.data_dir}")
    else:
        print(f"❌ Data directory not found: {evaluator.data_dir}")
    
    # Test bounty data structure
    test_bounty_data = {
        'url': test_url,
        'title': 'Test Bounty',
        'scraped_at': datetime.now().isoformat(),
        'content': 'This is a test bounty for demonstration purposes.',
        'links': ['https://example.com/prize', 'https://example.com/rules'],
        'requirements': ['Must be innovative', 'Should use modern tech'],
        'prizes': ['$1000', '$500'],
        'deadline': '2024-12-31',
        'company': 'Test Company'
    }
    
    # Test saving bounty data
    file_path = evaluator.save_bounty_data(test_bounty_data)
    print(f"✅ Bounty data saved to: {file_path}")
    
    # Test loading bounty data
    loaded_data = evaluator.load_bounty_data(test_url)
    if loaded_data:
        print("✅ Bounty data loaded successfully")
        print(f"   - Title: {loaded_data.get('title')}")
        print(f"   - Company: {loaded_data.get('company')}")
        print(f"   - Prizes: {loaded_data.get('prizes')}")
    else:
        print("❌ Failed to load bounty data")
    
    # Test chat history
    test_message = "I have an idea for a blockchain voting system"
    test_response = "That sounds interesting! Let me evaluate it against the bounty requirements."
    
    evaluator.save_chat_history(test_url, test_message, test_response)
    print("✅ Chat history saved")
    
    chat_history = evaluator.load_chat_history(test_url)
    if chat_history:
        print(f"✅ Chat history loaded: {len(chat_history)} messages")
    else:
        print("❌ Failed to load chat history")
    
    # Test getting all bounties
    all_bounties = evaluator.get_all_bounties()
    print(f"✅ Found {len(all_bounties)} bounties in database")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    test_evaluator()
