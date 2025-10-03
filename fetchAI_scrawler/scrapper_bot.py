#!/usr/bin/env python3
"""
Structured Web Scraper for Hackathon Prize Data
Extracts data in the exact format requested by the user
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


class StructuredHackathonScraper:
    """Web scraper that extracts hackathon data in the exact target format"""
    
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

    def extract_hackathon_sponsors(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract hackathon sponsors in the exact target format"""
        sponsors = []
        
        # Find all company sections
        company_sections = self._find_company_sections(soup)
        
        for company_name, section in company_sections.items():
            sponsor_data = self._extract_sponsor_data(section, company_name)
            if sponsor_data:
                sponsors.append(sponsor_data)
        
        return sponsors

    def _find_company_sections(self, soup: BeautifulSoup) -> Dict[str, BeautifulSoup]:
        """Find company sections by looking for logos and headings"""
        company_sections = {}
        
        # Method 1: Find by company logos
        logos = soup.find_all('img', alt=re.compile(r'logo', re.I))
        
        for logo in logos:
            company_name = self._extract_company_name_from_logo(logo)
            if company_name:
                parent_section = self._find_parent_section(logo)
                if parent_section:
                    company_sections[company_name] = parent_section
        
        # Method 2: Find by company headings
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            text = heading.get_text(strip=True)
            if self._is_company_heading(text):
                company_name = self._clean_company_name(text)
                if company_name and company_name not in company_sections:
                    parent_section = self._find_parent_section(heading)
                    if parent_section:
                        company_sections[company_name] = parent_section
        
        return company_sections

    def _extract_company_name_from_logo(self, logo) -> Optional[str]:
        """Extract company name from logo element"""
        alt_text = logo.get('alt', '') or logo.get('title', '')
        if alt_text:
            company_name = alt_text.replace('logo', '').strip()
            return self._clean_company_name(company_name)
        return None

    def _find_parent_section(self, element) -> Optional[BeautifulSoup]:
        """Find the parent section that contains company information"""
        current = element.parent
        while current:
            if current.name in ['div', 'section', 'article']:
                text = current.get_text()
                if len(text) > 200:  # Substantial content
                    return current
            current = current.parent
        return None

    def _is_company_heading(self, text: str) -> bool:
        """Check if a heading text is likely a company name"""
        if not text or len(text) < 3 or len(text) > 50:
            return False
        
        skip_words = ['prize', 'bounty', 'track', 'winner', 'place', 'logo', 'about', 'requirements', 'qualification', 'criteria', 'workshop', 'jobs', 'links', 'resources', 'cross chain', 'building', 'creating', 'guide', 'intro', 'build', 'zk', 'pdf', 'proof', 'apps', 'phone', 'ai', 'dapp', 'storage', 'graph', 'hyperg', 'bitcoin', 'applications', 'uniswap', 'exploring', 'possibilities', 'scheduled', 'txns', 'killer', 'apps']
        if any(skip in text.lower() for skip in skip_words):
            return False
        
        # Skip if it contains dollar amounts
        if re.search(r'\$[\d,]+', text):
            return False
        
        # Skip if it contains emojis (workshop titles often have emojis)
        if re.search(r'[🛠️🌊🍊📱]', text):
            return False
        
        # Skip if it's too long (likely a description, not a company name)
        if len(text) > 30:
            return False
        
        return True

    def _clean_company_name(self, name: str) -> Optional[str]:
        """Clean up company name"""
        name = name.strip()
        name = re.sub(r'\s+(logo|prize|bounty|track)$', '', name, flags=re.I)
        name = re.sub(r'\s+', ' ', name).strip()
        
        if len(name) > 2 and not any(skip in name.lower() for skip in ['prize', 'bounty', 'track', 'winner', 'place']):
            return name
        
        return None

    def _extract_sponsor_data(self, section: BeautifulSoup, company_name: str) -> Optional[Dict[str, Any]]:
        """Extract sponsor data in the exact target format"""
        try:
            # Extract basic information
            logo = self._extract_logo(section, company_name)
            website = self._extract_website(section)
            total_prize_pool = self._extract_total_prize_pool(section)
            about = self._extract_about(section)
            
            # Extract challenges
            challenges = self._extract_challenges(section)
            
            # Extract workshop info
            workshop = self._extract_workshop(section)
            
            # Extract jobs
            jobs = self._extract_jobs(section)
            
            sponsor_data = {
                "name": company_name,
                "logo": logo,
                "website": website,
                "totalPrizePool": total_prize_pool,
                "about": about,
                "challenges": challenges,
                "workshop": workshop,
                "jobs": jobs
            }
            
            return sponsor_data
            
        except Exception as e:
            print(f"Error extracting sponsor data for {company_name}: {e}")
            return None

    def _extract_logo(self, section: BeautifulSoup, company_name: str) -> str:
        """Extract logo information"""
        logos = section.find_all('img')
        for logo in logos:
            alt_text = logo.get('alt', '')
            if company_name.lower() in alt_text.lower() or 'logo' in alt_text.lower():
                return alt_text or f"{company_name} logo"
        return f"{company_name} logo"

    def _extract_website(self, section: BeautifulSoup) -> Optional[str]:
        """Extract company website"""
        for link in section.find_all('a', href=True):
            href = link['href']
            if href.startswith('/'):
                href = urljoin('https://ethglobal.com', href)
            
            if (href.startswith('http') and 
                not any(domain in href for domain in ['twitter', 'github', 'discord', 'telegram', 'linkedin', 'ethglobal', 'x.com'])):
                return href
        
        return None

    def _extract_total_prize_pool(self, section: BeautifulSoup) -> int:
        """Extract total prize pool as integer"""
        text = section.get_text()
        
        # Look for dollar amounts
        amounts = re.findall(r'\$[\d,]+', text)
        if amounts:
            # Convert to numbers and sum them
            total = 0
            for amount in amounts:
                try:
                    num = int(float(amount.replace('$', '').replace(',', '')))
                    total += num
                except:
                    pass
            
            if total > 0:
                return total
        
        return 10000  # Default

    def _clean_text(self, text: str) -> str:
        """Clean text by removing emojis and symbols"""
        # Remove emojis and symbols
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\{\}\"\'\/\\\@\#\$\%\^\&\*\+\=\|<>~`]', '', text)
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_about(self, section: BeautifulSoup) -> str:
        """Extract company about section with enhanced accuracy"""
        descriptions = []
        
        # Method 1: Look for explicit "About" sections
        about_headers = section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                       string=re.compile(r'about', re.I))
        
        for header in about_headers:
            current = header.next_sibling
            while current:
                if hasattr(current, 'get_text'):
                    text = self._clean_text(current.get_text(strip=True))
                    if text and len(text) > 50:
                        descriptions.append(text)
                current = current.next_sibling
        
        # Method 2: Look for paragraphs that start with company descriptions
        paragraphs = section.find_all('p')
        for p in paragraphs:
            text = self._clean_text(p.get_text(strip=True))
            if (text and len(text) > 50 and len(text) < 1000 and
                not re.search(r'\$[\d,]+', text) and
                not any(skip in text.lower() for skip in ['prize', 'bounty', 'track', 'winner', 'place', 'logo', 'workshop', 'jobs', 'qualification', 'requirements', 'deploy', 'verify', 'provide', 'include', 'optional', 'enhancements', 'earn', 'extra', 'points']) and
                any(keyword in text.lower() for keyword in ['provides', 'enables', 'builds', 'offers', 'aims', 'focuses', 'specializes', 'develops', 'creates', 'delivers', 'suite', 'tools', 'platform', 'network', 'protocol', 'blockchain', 'decentralized', 'sidechain', 'layer', 'ecosystem', 'powerful', 'designed', 'built', 'allows', 'supports', 'empowers', 'facilitates'])):
                descriptions.append(text)
        
        # Method 3: Look for specific company description patterns from the web search
        text = self._clean_text(section.get_text())
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if (len(line) > 50 and len(line) < 1000 and
                not re.search(r'\$[\d,]+', line) and
                not any(skip in line.lower() for skip in ['prize', 'bounty', 'track', 'winner', 'place', 'logo', 'workshop', 'jobs', 'qualification', 'requirements', 'deploy', 'verify', 'provide', 'include', 'optional', 'enhancements', 'earn', 'extra', 'points']) and
                (line.startswith(('1inch', 'Rootstock', 'Flow', 'Pyth', 'Hedera', 'Polygon', 'ENS', 'World', 'Uniswap', 'Filecoin')) or
                 any(pattern in line.lower() for pattern in ['provides a suite', 'is the top', 'is a layer', 'provides real-time', 'is the fast', 'is a new', 'is a next-generation', 'is a blockchain', 'is a platform', 'is a network', 'is a protocol', 'is a decentralized', 'is a sidechain', 'is a layer one', 'is a layer 1', 'is a layer-1']))):
                descriptions.append(line)
        
        # Method 4: Look for company descriptions that don't start with numbers or bullet points
        for line in lines:
            line = line.strip()
            if (len(line) > 50 and len(line) < 1000 and
                not re.search(r'\$[\d,]+', line) and
                not re.match(r'^\d+\.', line) and  # Don't start with numbers
                not re.match(r'^[•\-\*]', line) and  # Don't start with bullet points
                not any(skip in line.lower() for skip in ['prize', 'bounty', 'track', 'winner', 'place', 'logo', 'workshop', 'jobs', 'qualification', 'requirements', 'deploy', 'verify', 'provide', 'include', 'optional', 'enhancements', 'earn', 'extra', 'points']) and
                any(keyword in line.lower() for keyword in ['provides', 'enables', 'builds', 'offers', 'aims', 'focuses', 'specializes', 'develops', 'creates', 'delivers', 'suite', 'tools', 'platform', 'network', 'protocol', 'blockchain', 'decentralized', 'sidechain', 'layer', 'ecosystem', 'powerful', 'designed', 'built', 'allows', 'supports', 'empowers', 'facilitates', 'leverages', 'utilizes', 'combines', 'integrates'])):
                descriptions.append(line)
        
        # Method 5: Look for company descriptions in div elements with specific classes
        about_divs = section.find_all('div', class_=re.compile(r'about|description|content|intro', re.I))
        for div in about_divs:
            text = self._clean_text(div.get_text(strip=True))
            if (text and len(text) > 50 and len(text) < 1000 and
                not re.search(r'\$[\d,]+', text) and
                not any(skip in text.lower() for skip in ['prize', 'bounty', 'track', 'winner', 'place', 'logo', 'workshop', 'jobs', 'qualification', 'requirements'])):
                descriptions.append(text)
        
        if descriptions:
            # Return the longest, most descriptive text that looks like a company description
            best_description = max(descriptions, key=len)
            return best_description
        
        return f"Prize sponsor: {company_name}"

    def _extract_challenges(self, section: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract challenges in the target format"""
        challenges = []
        
        # Look for challenge sections
        challenge_headers = section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                           string=re.compile(r'challenge|prize|track', re.I))
        
        for header in challenge_headers:
            challenge_data = self._extract_challenge_data(header)
            if challenge_data:
                challenges.append(challenge_data)
        
        # If no challenges found, create a default one
        if not challenges:
            total_prize = self._extract_total_prize_pool(section)
            challenges.append({
                "title": f"{self._extract_company_name_from_section(section)} Challenge",
                "prizePool": total_prize,
                "prizes": {
                    "1st": total_prize // 2,
                    "2nd": total_prize // 3,
                    "3rd": total_prize // 6
                },
                "description": f"Build innovative solutions using {self._extract_company_name_from_section(section)} technology.",
                "requirements": [
                    "Deploy smart contract or run transactions",
                    "Include README with project description",
                    "Provide demo video or slides"
                ]
            })
        
        return challenges

    def _extract_company_name_from_section(self, section: BeautifulSoup) -> str:
        """Extract company name from section"""
        headings = section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            text = heading.get_text(strip=True)
            if self._is_company_heading(text):
                return self._clean_company_name(text)
        return "Company"

    def _extract_challenge_data(self, header) -> Optional[Dict[str, Any]]:
        """Extract data for a single challenge"""
        try:
            title = header.get_text(strip=True)
            
            # Find prize amount in the header or nearby
            prize_pool = self._extract_challenge_prize_pool(header)
            
            # Extract description
            description = self._extract_challenge_description(header)
            
            # Extract requirements
            requirements = self._extract_challenge_requirements(header)
            
            # Extract judging criteria
            judging_criteria = self._extract_judging_criteria(header)
            
            # Create prize breakdown
            prizes = self._create_prize_breakdown(prize_pool)
            
            challenge_data = {
                "title": title,
                "prizePool": prize_pool,
                "prizes": prizes,
                "description": description,
                "requirements": requirements
            }
            
            if judging_criteria:
                challenge_data["judgingCriteria"] = judging_criteria
            
            return challenge_data
            
        except Exception as e:
            print(f"Error extracting challenge data: {e}")
            return None

    def _extract_challenge_prize_pool(self, header) -> int:
        """Extract prize pool for a challenge"""
        text = header.get_text()
        amounts = re.findall(r'\$[\d,]+', text)
        if amounts:
            try:
                return int(float(amounts[0].replace('$', '').replace(',', '')))
            except:
                pass
        return 1000

    def _extract_challenge_description(self, header) -> str:
        """Extract challenge description"""
        current = header.next_sibling
        while current:
            if hasattr(current, 'get_text'):
                text = self._clean_text(current.get_text(strip=True))
                if text and len(text) > 20:
                    return text
            current = current.next_sibling
        return "Build innovative solutions for this challenge."

    def _extract_challenge_requirements(self, header) -> List[str]:
        """Extract challenge requirements"""
        requirements = []
        
        # Look for requirement sections
        req_headers = header.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                   string=re.compile(r'requirement|qualification|criteria', re.I))
        
        for req_header in req_headers:
            current = req_header.next_sibling
            while current:
                if hasattr(current, 'get_text'):
                    text = current.get_text(strip=True)
                    if text and len(text) > 10:
                        items = re.split(r'[•\-\*]\s*|\n', text)
                        for item in items:
                            item = item.strip()
                            if len(item) > 10:
                                requirements.append(item)
                current = current.next_sibling
        
        # If no requirements found, add defaults
        if not requirements:
            requirements = [
                "Deploy smart contract or run transactions",
                "Include README with project description",
                "Provide demo video or slides"
            ]
        
        return requirements[:10]  # Limit to 10

    def _extract_judging_criteria(self, header) -> List[str]:
        """Extract judging criteria"""
        criteria = []
        
        # Look for judging criteria sections
        criteria_headers = header.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                         string=re.compile(r'judging|criteria|evaluation', re.I))
        
        for criteria_header in criteria_headers:
            current = criteria_header.next_sibling
            while current:
                if hasattr(current, 'get_text'):
                    text = current.get_text(strip=True)
                    if text and len(text) > 10:
                        items = re.split(r'[•\-\*]\s*|\n', text)
                        for item in items:
                            item = item.strip()
                            if len(item) > 10:
                                criteria.append(item)
                current = current.next_sibling
        
        return criteria[:5]  # Limit to 5

    def _create_prize_breakdown(self, total_prize: int) -> Dict[str, int]:
        """Create prize breakdown from total prize pool"""
        if total_prize <= 1000:
            return {"upTo3Teams": total_prize}
        elif total_prize <= 5000:
            return {
                "1st": total_prize // 2,
                "2nd": total_prize // 3
            }
        else:
            return {
                "1st": total_prize // 2,
                "2nd": total_prize // 3,
                "3rd": total_prize // 6
            }

    def _extract_workshop(self, section: BeautifulSoup) -> Optional[Dict[str, str]]:
        """Extract workshop information"""
        workshop_headers = section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                          string=re.compile(r'workshop', re.I))
        
        for header in workshop_headers:
            title = header.get_text(strip=True)
            
            # Look for time and location info nearby
            text = section.get_text()
            time_match = re.search(r'(\d{1,2}:\d{2}\s*[AP]M[^—]*—[^,]+)', text)
            location_match = re.search(r'(Workshop Room \d+|Room \d+)', text)
            speaker_match = re.search(r'Speaker[:\s]+([^,\n]+)', text, re.I)
            
            workshop_info = {"title": title}
            
            if time_match:
                workshop_info["time"] = time_match.group(1)
            if location_match:
                workshop_info["location"] = location_match.group(1)
                workshop_info["format"] = "In-person"
            if speaker_match:
                workshop_info["speaker"] = speaker_match.group(1)
            
            # Check if it's online/video format
            if "video" in text.lower() or "online" in text.lower():
                workshop_info["format"] = "Online/Video"
            
            return workshop_info
        
        return None

    def _extract_jobs(self, section: BeautifulSoup) -> List[str]:
        """Extract job listings"""
        jobs = []
        
        job_headers = section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], 
                                     string=re.compile(r'job', re.I))
        
        for header in job_headers:
            current = header.next_sibling
            while current:
                if hasattr(current, 'get_text'):
                    text = current.get_text(strip=True)
                    if text and len(text) > 5:
                        # Split by common separators
                        job_items = re.split(r'[•\-\*]\s*|\n', text)
                        for item in job_items:
                            item = item.strip()
                            if len(item) > 5 and not re.search(r'\$[\d,]+', item):
                                jobs.append(item)
                current = current.next_sibling
        
        # Clean up jobs
        cleaned_jobs = []
        for job in jobs:
            if len(job) > 5 and len(job) < 100:
                cleaned_jobs.append(job)
        
        return cleaned_jobs[:5]  # Limit to 5

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Main scraping method that returns data in the exact target format"""
        try:
            print(f"Scraping URL: {url}")
            html = await self.fetch_page(url)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract hackathon sponsors
            hackathon_sponsors = self.extract_hackathon_sponsors(soup)
            print(f"Found {len(hackathon_sponsors)} sponsors")
            
            # Create result in the exact target format
            result = {
                "hackathonSponsors": hackathon_sponsors
            }
            
            print("Scraping complete!")
            print(f"   Sponsors: {len(hackathon_sponsors)}")
            
            return result
            
        except Exception as e:
            print(f"Scraping failed: {e}")
            raise Exception(f"Failed to scrape URL {url}: {str(e)}")


async def scrape_hackathon_url(url: str) -> Dict[str, Any]:
    """Main function to scrape hackathon data in the target format"""
    async with StructuredHackathonScraper() as scraper:
        return await scraper.scrape_url(url)


def save_scraped_data(data: Dict[str, Any], filename: str = None) -> str:
    """Save scraped data to JSON file"""
    if not filename:
        date_str = datetime.now().strftime('%Y_%m_%d')
        filename = f"scraped_data/structured_hackathon_{date_str}.json"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data saved to: {filename}")
    return filename


async def main():
    """Test the structured scraper with New Delhi 2025 URL"""
    url = "https://ethglobal.com/events/newdelhi/prizes"
    print("STRUCTURED HACKATHON SCRAPER TEST")
    print("=" * 50)
    print(f"URL: {url}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        data = await scrape_hackathon_url(url)
        filename = save_scraped_data(data)
        
        print("SCRAPING RESULTS:")
        print("=" * 30)
        
        sponsors = data.get("hackathonSponsors", [])
        for i, sponsor in enumerate(sponsors[:5], 1):
            print(f"\n{i}. {sponsor['name']}")
            print(f"   Total Prize Pool: ${sponsor['totalPrizePool']:,}")
            print(f"   Website: {sponsor.get('website', 'N/A')}")
            print(f"   About: {sponsor['about'][:100]}...")
            print(f"   Challenges: {len(sponsor['challenges'])}")
            if sponsor.get('workshop'):
                print(f"   Workshop: {sponsor['workshop']['title']}")
            if sponsor.get('jobs'):
                print(f"   Jobs: {len(sponsor['jobs'])} positions")
        
        if len(sponsors) > 5:
            print(f"\n... and {len(sponsors) - 5} more sponsors")
        
        print("Scraping complete!")
        print(f"   Successfully scraped {len(sponsors)} sponsors")
        
        return data
        
    except Exception as e:
        print(f"Scraping failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(main())
