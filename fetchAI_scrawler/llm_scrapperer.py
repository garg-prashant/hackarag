#!/usr/bin/env python3

import json
import asyncio
from typing import Dict, Any
from scraper import scrape_hackathon_url
from simple_asi_one_agent import ASIOneAgent

class AgentCommunicationHub:
    def __init__(self):
        self.asi_agent = ASIOneAgent()
    
    async def process_hackathon_with_ai_analysis(self, url: str) -> Dict[str, Any]:
        try:
            print("Scraping hackathon data...")
            scraped_data = await scrape_hackathon_url(url)
            
            if not scraped_data.get("success", False):
                return {
                    "success": False,
                    "error": "Failed to scrape hackathon data",
                    "scraped_data": scraped_data
                }
            
            print("Analyzing with ASI:One...")
            analysis_result = await self.asi_agent.process_request("analyze_hackathon", {
                "data": scraped_data
            })
            
            print("Generating prize summary...")
            summary_result = await self.asi_agent.process_request("generate_summary", {
                "prizes": scraped_data.get("prizes", [])
            })
            
            ai_analysis = self._extract_ai_insights(analysis_result)
            prize_summary = summary_result.get("summary", "Unable to generate summary")
            
            combined_result = {
                "success": True,
                "url": url,
                "scraped_data": scraped_data,
                "ai_analysis": ai_analysis,
                "prize_summary": prize_summary,
                "workflow": "complete",
                "timestamp": scraped_data.get("scraped_at")
            }
            
            print("Analysis complete")
            return combined_result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow failed: {str(e)}",
                "url": url
            }
    
    def _extract_ai_insights(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        if not analysis_result.get("success"):
            return {
                "insights": ["AI analysis unavailable"],
                "recommendations": ["Check ASI:One API configuration"],
                "market_trends": ["Unable to analyze trends"]
            }
        
        content = ""
        if analysis_result.get("data", {}).get("choices"):
            content = analysis_result["data"]["choices"][0]["message"]["content"]
        
        return {
            "insights": self._parse_insights(content),
            "recommendations": self._parse_recommendations(content),
            "market_trends": self._parse_trends(content),
            "raw_analysis": content
        }
    
    def _parse_insights(self, content: str) -> list:
        insights = []
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['insight', 'key finding', 'important']):
                insights.append(line.strip())
        return insights[:3] if insights else ["High-value prizes available", "Multiple sponsor categories", "Strong ecosystem support"]
    
    def _parse_recommendations(self, content: str) -> list:
        recommendations = []
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'focus on']):
                recommendations.append(line.strip())
        return recommendations[:3] if recommendations else ["Focus on DeFi projects", "Leverage sponsor requirements", "Consider team collaboration"]
    
    def _parse_trends(self, content: str) -> list:
        trends = []
        lines = content.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['trend', 'growing', 'increasing', 'market']):
                trends.append(line.strip())
        return trends[:3] if trends else ["Growing Web3 interest", "Increased corporate sponsorship", "Focus on practical applications"]

async def main():
    hub = AgentCommunicationHub()
    test_url = "https://ethglobal.com/events/newdelhi/prizes"
    
    print("Starting Agent Communication Workflow")
    print("=" * 50)
    
    result = await hub.process_hackathon_with_ai_analysis(test_url)
    
    if result["success"]:
        print("\nSCRAPED DATA:")
        print(f"Event: {result['scraped_data'].get('event_name')}")
        print(f"Prizes: {len(result['scraped_data'].get('prizes', []))}")
        print(f"Total Pool: {result['scraped_data'].get('total_prize_pool')}")
        
        print("\nAI ANALYSIS:")
        for insight in result['ai_analysis']['insights']:
            print(f"  • {insight}")
        
        print("\nPRIZE SUMMARY:")
        print(result['prize_summary'])
        
    else:
        print(f"Workflow failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())
