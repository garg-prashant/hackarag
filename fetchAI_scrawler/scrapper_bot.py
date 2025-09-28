#!/usr/bin/env python3
"""
Enhanced Web Scraper for Hackathon Prize Data
Highly accurate scraper specifically designed for ETHGlobal and similar hackathon sites
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import aiohttp
import ssl


class HackathonScraper:
    """High accuracy web scraper for hackathon prize information"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        self.session = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        self.session = aiohttp.ClientSession(
            connector=connector,
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_page(self, url: str) -> str:
        """Fetches the HTML content of a given URL with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)

    def extract_event_name(self, soup: BeautifulSoup, url: str) -> str:
        """Extract event name with multiple fallback methods"""
        methods = [
            lambda: soup.find('title').get_text().strip() if soup.find('title') else None,
            lambda: soup.find('h1').get_text().strip() if soup.find('h1') else None,
            lambda: soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else None,
            lambda: self._extract_event_from_text(soup.get_text()),
            lambda: self._extract_event_from_url(url)
        ]
        
        for method in methods:
            try:
                result = method()
                if result and len(result.strip()) > 3:
                    return result.strip()
            except:
                continue
        
        return 'Unknown Event'

    def _extract_event_from_text(self, text: str) -> Optional[str]:
        """Extract event name from page text"""
        patterns = [
            r'ETHGlobal\s+([^,\n]+)',
            r'([^,\n]*Hackathon[^,\n]*)',
            r'([^,\n]*Event[^,\n]*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1).strip()
        return None

    def _extract_event_from_url(self, url: str) -> str:
        """Extract event name from URL"""
        if 'ethglobal' in url.lower():
            if 'newdelhi' in url.lower():
                return 'ETHGlobal New Delhi'
            elif 'singapore' in url.lower():
                return 'ETHGlobal Singapore'
            elif 'istanbul' in url.lower():
                return 'ETHGlobal Istanbul'
            else:
                return 'ETHGlobal'
        return 'Hackathon'

    def extract_prizes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract prize information with enhanced accuracy"""
        prizes = []
        
        ethglobal_prizes = self._extract_ethglobal_prizes(soup)
        if ethglobal_prizes:
            prizes.extend(ethglobal_prizes)
        
        if not prizes:
            general_prizes = self._extract_general_prizes(soup)
            prizes.extend(general_prizes)
        
        if not prizes:
            fallback_prizes = self._extract_fallback_prizes(soup)
            prizes.extend(fallback_prizes)
        
        prizes = self._clean_and_deduplicate_prizes(prizes)
        
        return prizes

    def _extract_ethglobal_prizes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract prizes specifically from ETHGlobal structure"""
        prizes = []
        
        prize_elements = soup.find_all(['div', 'section', 'article'], 
                                     class_=re.compile(r'prize|bounty|track|company|sponsor', re.I))
        
        amount_elements = soup.find_all(text=re.compile(r'\$[\d,]+'))
        
        for element in prize_elements:
            prize_data = self._extract_prize_from_element(element)
            if prize_data and self._is_valid_prize(prize_data):
                prizes.append(prize_data)
        
        if not prizes and amount_elements:
            prizes = self._extract_prizes_from_amounts(soup, amount_elements)
        
        return prizes

    def _extract_prize_from_element(self, element) -> Optional[Dict[str, Any]]:
        """Extract prize data from a single element"""
        try:
            company_name = self._extract_company_name_enhanced(element)
            if not company_name:
                return None
            
            prize_amount = self._extract_prize_amount_enhanced(element)
            title = self._extract_prize_title_enhanced(element)
            description = self._extract_prize_description_enhanced(element)
            requirements = self._extract_requirements_enhanced(element)
            category = self._determine_category_enhanced(element, description)
            website, twitter = self._extract_links_enhanced(element)
            
            return {
                'company': company_name,
                'title': title or f"{company_name} Prize",
                'description': description or 'No description available',
                'prizes': prize_amount or 'Unknown',
                'category': category,
                'requirements': requirements,
                'website': website,
                'twitter': twitter
            }
        except Exception as e:
            print(f"Error extracting prize from element: {e}")
            return None

    def _extract_company_name_enhanced(self, element) -> Optional[str]:
        """Enhanced company name extraction"""
        selectors = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            '[class*="company"]', '[class*="sponsor"]', '[class*="title"]',
            '[class*="name"]', '[class*="brand"]', '[class*="logo"]'
        ]
        
        for selector in selectors:
            elements = element.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                if text and len(text) > 2 and not text.isdigit():
                    text = re.sub(r'\s+', ' ', text).strip()
                    if not any(skip in text.lower() for skip in ['prize', 'bounty', 'track', 'winner', 'place']):
                        return text
        
        img_elements = element.find_all('img')
        for img in img_elements:
            alt_text = img.get('alt', '') or img.get('title', '')
            if alt_text and len(alt_text) > 2:
                return alt_text.strip()
        
        parent = element.parent
        if parent:
            return self._extract_company_name_enhanced(parent)
        
        return None

    def _extract_prize_amount_enhanced(self, element) -> Optional[str]:
        """Enhanced prize amount extraction"""
        text = element.get_text()
        
        patterns = [
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:USD|dollars?))?',
            r'[\d,]+(?:\.\d{2})?\s*(?:USD|dollars?)',
            r'(\d+)\s*(?:ETH|ethereum)',
            r'(\d+)\s*(?:BTC|bitcoin)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.I)
            if matches:
                amounts = []
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    amount_str = match.replace('$', '').replace(',', '')
                    try:
                        amounts.append((float(amount_str), match))
                    except:
                        pass
                
                if amounts:
                    return max(amounts, key=lambda x: x[0])[1]
        
        return None

    def _extract_prize_title_enhanced(self, element) -> Optional[str]:
        """Enhanced prize title extraction"""
        title_selectors = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            '[class*="title"]', '[class*="name"]', '[class*="prize"]',
            '[class*="bounty"]', '[class*="track"]'
        ]
        
        for selector in title_selectors:
            title_elem = element.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title and len(title) > 3:
                    title = re.sub(r'\s+', ' ', title).strip()
                    return title
        
        return None

    def _extract_prize_description_enhanced(self, element) -> Optional[str]:
        """Enhanced description extraction"""
        desc_selectors = [
            'p', '[class*="description"]', '[class*="about"]',
            '[class*="content"]', '[class*="text"]', '[class*="details"]'
        ]
        
        descriptions = []
        
        for selector in desc_selectors:
            desc_elems = element.select(selector)
            for desc_elem in desc_elems:
                text = desc_elem.get_text(strip=True)
                if text and len(text) > 20 and not text.isdigit():
                    text = re.sub(r'\s+', ' ', text)
                    if text not in descriptions:
                        descriptions.append(text)
        
        if descriptions:
            return max(descriptions, key=len)
        
        return None

    def _extract_requirements_enhanced(self, element) -> List[str]:
        """Enhanced requirements extraction"""
        requirements = []
        
        req_patterns = [
            r'(?:must|should|require|need|integrate|use|build|deploy|submit)[^.!?]*[.!?]',
            r'(?:qualification|requirement|criteria)[^.!?]*[.!?]'
        ]
        
        text = element.get_text()
        for pattern in req_patterns:
            matches = re.findall(pattern, text, re.I)
            for match in matches:
                req = match.strip()
                if len(req) > 10:
                    requirements.append(req)
        
        list_items = element.find_all(['li', 'p'])
        for item in list_items:
            text = item.get_text(strip=True)
            if text and len(text) > 10 and not text.isdigit():
                if any(keyword in text.lower() for keyword in ['must', 'should', 'require', 'need', 'integrate', 'use', 'build', 'deploy']):
                    requirements.append(text)
        
        return list(set(requirements))

    def _determine_category_enhanced(self, element, description: str) -> str:
        """Enhanced category determination"""
        text = (element.get_text() + ' ' + (description or '')).lower()
        
        categories = {
            'DeFi': ['defi', 'decentralized finance', 'swap', 'liquidity', 'yield', 'uniswap', 'dex', 'lending', 'borrowing'],
            'NFT': ['nft', 'non-fungible', 'token', 'collectible', 'art', 'gaming'],
            'DAO': ['dao', 'governance', 'voting', 'decentralized autonomous'],
            'Identity': ['identity', 'world id', 'verification', 'proof of personhood', 'kyc', 'aml'],
            'Privacy': ['privacy', 'zero-knowledge', 'zk', 'fhenix', 'confidential', 'private'],
            'Infrastructure': ['infrastructure', 'node', 'validator', 'avail', 'storage', 'network'],
            'Gaming': ['game', 'gaming', 'play', 'metaverse', 'virtual'],
            'Social': ['social', 'community', 'circles', 'social media', 'chat'],
            'Cross-Chain': ['cross-chain', 'bridge', 'layerzero', 'hyperlane', 'multichain'],
            'Oracles': ['oracle', 'chainlink', 'pyth', 'price feed', 'data'],
            'Payments': ['payment', 'circle', 'stablecoin', 'usdc', 'usdt', 'pyusd'],
            'AI/ML': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'agent', 'agentic'],
            'Real Estate': ['real estate', 'property', 'tokenization', 'rwa', 'regulated asset'],
            'Storage': ['storage', 'filecoin', 'ipfs', 'walrus', 'decentralized storage']
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'General'

    def _extract_links_enhanced(self, element) -> Tuple[Optional[str], Optional[str]]:
        """Enhanced link extraction"""
        website = None
        twitter = None
        
        for link in element.find_all('a', href=True):
            href = link['href']
            
            if href.startswith('/'):
                href = urljoin('https://ethglobal.com', href)
            
            if 'twitter.com' in href or 'x.com' in href:
                twitter = href
            elif href.startswith('http') and not any(domain in href for domain in ['twitter', 'github', 'discord', 'telegram', 'linkedin']):
                website = href
        
        return website, twitter

    def _extract_prizes_from_amounts(self, soup: BeautifulSoup, amount_elements) -> List[Dict[str, Any]]:
        """Extract prizes from elements containing amounts"""
        prizes = []
        
        for amount_text in amount_elements:
            parent = amount_text.parent
            while parent and parent.name not in ['div', 'section', 'article']:
                parent = parent.parent
            
            if parent:
                prize_data = self._extract_prize_from_element(parent)
                if prize_data:
                    prizes.append(prize_data)
        
        return prizes

    def _extract_general_prizes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract prizes using general methods"""
        prizes = []
        
       potential_prize_elements = soup.find_all(['div', 'section', 'article', 'li'])
        for element in potential_prize_elements:
            text = element.get_text()
            if re.search(r'\$[\d,]+', text) and len(text) > 50:
                prize_data = self._extract_prize_from_element(element)
                if prize_data and self._is_valid_prize(prize_data):
                    prizes.append(prize_data)
        
        return prizes

    def _extract_fallback_prizes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Fallback method using text patterns"""
        text = soup.get_text()
        prizes = []
        
        prize_patterns = [
            r'(\$[\d,]+(?:\.\d{2})?)\s*([^.!?]*(?:prize|bounty|track)[^.!?]*)',
            r'([^.!?]*(?:prize|bounty|track)[^.!?]*)\s*(\$[\d,]+(?:\.\d{2})?)'
        ]
        
        for pattern in prize_patterns:
            matches = re.findall(pattern, text, re.I)
            for i, match in enumerate(matches):
                if len(match) == 2:
                    amount, description = match
                    if amount and description:
                        prizes.append({
                            'company': f"Sponsor {i+1}",
                            'title': f"Prize {i+1}",
                            'description': description.strip(),
                            'prizes': amount,
                            'category': 'General',
                            'requirements': [],
                            'website': None,
                            'twitter': None
                        })
        
        return prizes

    def _is_valid_prize(self, prize_data: Dict[str, Any]) -> bool:
        """Validate if prize data is meaningful"""
        if not prize_data:
            return False
        
        if not prize_data.get('company') or prize_data['company'] in ['Company', 'Unknown Company']:
            return False
        
        if not prize_data.get('description') and not prize_data.get('title'):
            return False
        
        description = prize_data.get('description', '')
        if len(description) < 10:
            return False
        
        return True

    def _clean_and_deduplicate_prizes(self, prizes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and remove duplicate prizes"""
        cleaned_prizes = []
        seen_companies = set()
        
        for prize in prizes:
            company = prize.get('company', '')
            description = prize.get('description', '')
            
            key = f"{company}_{description[:50]}"
            
            if key not in seen_companies and self._is_valid_prize(prize):
                cleaned_prizes.append(prize)
                seen_companies.add(key)
        
        return cleaned_prizes

    def calculate_total_prize_pool(self, prizes: List[Dict[str, Any]]) -> str:
        """Calculate total prize pool from prizes"""
        total = 0
        for prize in prizes:
            amount_str = prize.get('prizes', '0').replace('$', '').replace(',', '')
            try:
                total += float(amount_str)
            except:
                pass
        return f"${total:,.2f}"

    def extract_companies(self, prizes: List[Dict[str, Any]]) -> List[str]:
        """Extract unique companies from prizes"""
        companies = set()
        for prize in prizes:
            company = prize.get('company')
            if company and company not in ['Company', 'Unknown Company']:
                companies.add(company)
        return list(companies)

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Main scraping method with enhanced accuracy"""
        try:
            print(f"Scraping URL: {url}")
            html = await self.fetch_page(url)
            soup = BeautifulSoup(html, 'html.parser')
            
            event_name = self.extract_event_name(soup, url)
            print(f"Event: {event_name}")
            
            # Extract prizes
            prizes = self.extract_prizes(soup)
            print(f"Found {len(prizes)} prizes")
            
            # Calculate totals
            total_prize_pool = self.calculate_total_prize_pool(prizes)
            companies = self.extract_companies(prizes)
            
            # Extract requirements
            all_requirements = []
            for prize in prizes:
                all_requirements.extend(prize.get('requirements', []))
            
            # Create result
            result = {
                'url': url,
                'event_name': event_name,
                'total_prize_pool': total_prize_pool,
                'prizes': prizes,
                'companies': companies,
                'requirements': all_requirements,
                'scraped_at': datetime.now().isoformat(),
                'source': 'hackathon_scraper'
            }
            
            print("Scraping complete!")
            print(f"   Companies: {len(companies)}")
            print(f"   Total Prizes: {len(prizes)}")
            print(f"   Total Prize Pool: {total_prize_pool}")
            
            return result
            
        except Exception as e:
            print(f"Scraping failed: {e}")
            raise Exception(f"Failed to scrape URL {url}: {str(e)}")


async def scrape_hackathon_url(url: str) -> Dict[str, Any]:
    """Main function to scrape hackathon data with high accuracy"""
    async with HackathonScraper() as scraper:
        return await scraper.scrape_url(url)


def save_scraped_data(data: Dict[str, Any], filename: str = None) -> str:
    """Save scraped data to JSON file"""
    if not filename:
        event_name = data.get('event_name', 'hackathon').replace(' ', '_').lower()
        date_str = datetime.now().strftime('%Y_%m_%d')
        filename = f"scraped_data/{event_name}_{date_str}.json"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data saved to: {filename}")
    return filename


async def main():
    """Test the scraper with New Delhi 2025 URL"""
    url = "https://ethglobal.com/events/newdelhi/prizes"
    print("HACKATHON SCRAPER TEST")
    print("=" * 40)
    print(f"URL: {url}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        data = await scrape_hackathon_url(url)
        filename = save_scraped_data(data)
        
        print("SCRAPING RESULTS:")
        print("=" * 30)
        
        for i, prize in enumerate(data['prizes'][:10], 1):
            print(f"\n{i}. {prize['title']} - {prize['prizes']}")
            print(f"   Company: {prize['company']}")
            print(f"   Category: {prize['category']}")
            print(f"   Description: {prize['description'][:100]}...")
            if prize['requirements']:
                print(f"   Requirements: {len(prize['requirements'])} items")
        
        if len(data['prizes']) > 10:
            print(f"\n... and {len(data['prizes']) - 10} more prizes")
        
        print("Scraping complete!")
        print(f"   Successfully scraped {len(data['prizes'])} prizes")
        print(f"   Found {len(data['companies'])} companies")
        print(f"   Total prize pool: {data['total_prize_pool']}")
        
        return data
        
    except Exception as e:
        print(f"Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())
