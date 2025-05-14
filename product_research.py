"""
Product Research Module for FallnAI DropShipper
----------------------------------------------
Handles automated product research, trend analysis, and selection
of profitable products for dropshipping.
"""

import logging
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Set, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import random
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import re
from functools import lru_cache

from config import Config, ProductResearchConfig
from utils.ai_utils import generate_text, analyze_market_potential, generate_product_description
from utils.web import fetch_url, scrape_product_data

logger = logging.getLogger("FallnAI.ProductResearch")

@dataclass
class ProductNiche:
    """Represents a product niche with market data."""
    name: str
    category: str
    search_volume: int
    competition_level: float  # 0.0 to 1.0
    trend_direction: str  # "up", "down", "stable"
    seasonal: bool
    estimated_margin: float
    related_keywords: List[str]
    timestamp: str = ""
    
    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductNiche':
        """Create instance from dictionary."""
        return cls(**data)
    
    @property
    def score(self) -> float:
        """Calculate niche score based on various factors."""
        # Higher search volume is better, but diminishing returns after a point
        search_score = min(self.search_volume / 10000, 1.0) * 0.3
        
        # Lower competition is better
        competition_score = (1 - self.competition_level) * 0.3
        
        # Upward trends preferred
        trend_score = {"up": 0.3, "stable": 0.2, "down": 0.1}.get(self.trend_direction, 0.1)
        
        # Higher margins are better
        margin_score = min(self.estimated_margin / 0.5, 1.0) * 0.3
        
        # Non-seasonal products slightly preferred for consistent sales
        seasonal_penalty = 0.05 if self.seasonal else 0
        
        return search_score + competition_score + trend_score + margin_score - seasonal_penalty


@dataclass
class Product:
    """Represents a specific product for dropshipping."""
    id: str  # Unique identifier
    name: str
    description: str
    niche: str
    category: str
    estimated_cost: float
    recommended_price: float
    image_urls: List[str]
    features: List[str]
    specifications: Dict[str, str]
    keywords: List[str]
    weight: float  # in kg
    dimensions: Dict[str, float]  # length, width, height in cm
    shipping_restrictions: List[str]
    variants: List[Dict[str, Any]]
    source_urls: List[str]
    timestamp: str = ""
    
    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        
        # Generate ID if not provided
        if not self.id or self.id == "auto":
            # Create a slug-like ID from the name
            self.id = re.sub(r'[^a-z0-9]', '-', self.name.lower())
            self.id = re.sub(r'-+', '-', self.id).strip('-')
            # Add a timestamp hash for uniqueness
            time_hash = hex(int(time.time()))[-6:]
            self.id = f"{self.id[:20]}-{time_hash}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        """Create instance from dictionary."""
        return cls(**data)
    
    @property
    def profit_margin(self) -> float:
        """Calculate profit margin percentage."""
        if self.estimated_cost <= 0 or self.recommended_price <= 0:
            return 0.0
        return (self.recommended_price - self.estimated_cost) / self.recommended_price
    
    @property
    def profit_amount(self) -> float:
        """Calculate profit amount per unit."""
        return self.recommended_price - self.estimated_cost
    
    def save(self, data_dir: str = "data/products") -> None:
        """Save product data to file."""
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"{self.id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.debug(f"Product {self.id} saved to {file_path}")


class ProductResearcher:
    """
    Handles all aspects of product research and selection.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the product researcher with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.product_config = config.product_research
        self.data_dir = os.path.join(config.data_directory, "products")
        self.niche_data_file = os.path.join(self.data_dir, "niches.json")
        self.last_research_time = None
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load previously researched niches
        self._load_niches()
        
        logger.info("ProductResearcher initialized")
    
    def _load_niches(self) -> None:
        """Load previously researched niches from file."""
        self.niches: List[ProductNiche] = []
        
        if os.path.exists(self.niche_data_file):
            try:
                with open(self.niche_data_file, 'r') as f:
                    niche_dicts = json.load(f)
                
                self.niches = [ProductNiche.from_dict(niche_dict) for niche_dict in niche_dicts]
                logger.info(f"Loaded {len(self.niches)} product niches from file")
            except Exception as e:
                logger.error(f"Failed to load niches: {e}")
                self.niches = []
    
    def _save_niches(self) -> None:
        """Save researched niches to file."""
        try:
            niche_dicts = [niche.to_dict() for niche in self.niches]
            
            with open(self.niche_data_file, 'w') as f:
                json.dump(niche_dicts, f, indent=2)
            
            logger.info(f"Saved {len(self.niches)} product niches to file")
        except Exception as e:
            logger.error(f"Failed to save niches: {e}")
    
    def should_find_new_products(self) -> bool:
        """
        Determine if it's time to find new products based on the configured interval.
        
        Returns:
            bool: True if new product research should be performed
        """
        if self.last_research_time is None:
            return True
        
        interval_days = self.product_config.product_search_interval_days
        time_since_last = datetime.now() - self.last_research_time
        
        return time_since_last.days >= interval_days
    
    async def _fetch_trend_data(self, keyword: str, source: str) -> Dict[str, Any]:
        """
        Fetch trend data for a keyword from the specified source.
        
        Args:
            keyword: Search keyword
            source: Trend data source (e.g., "google_trends")
            
        Returns:
            Dict: Trend data
        """
        # Implement different fetching methods based on source
        if source == "google_trends":
            # Simulated Google Trends data for demonstration
            # In a real application, you'd use a proper API or scraping
            search_volume = random.randint(1000, 50000)
            trend = random.choice(["up", "down", "stable"])
            competition = random.uniform(0.2, 0.8)
            
            return {
                "search_volume": search_volume,
                "trend": trend,
                "competition": competition,
                "seasonal": random.random() > 0.7
            }
        
        elif source in ["aliexpress", "amazon"]:
            # Simulated marketplace trend data
            # In a real application, you'd implement proper API calls or scraping
            return {
                "search_volume": random.randint(500, 20000),
                "trend": random.choice(["up", "down", "stable"]),
                "competition": random.uniform(0.3, 0.9),
                "seasonal": random.random() > 0.8,
                "estimated_margin": random.uniform(0.15, 0.45)
            }
        
        # Default response for unsupported sources
        return {
            "search_volume": 0,
            "trend": "stable",
            "competition": 0.5,
            "seasonal": False,
            "estimated_margin": 0.2
        }
    
    async def _research_niche(self, category: str, keyword: str) -> Optional[ProductNiche]:
        """
        Research a specific product niche using various data sources.
        
        Args:
            category: Product category
            keyword: Niche keyword to research
            
        Returns:
            Optional[ProductNiche]: Researched niche data or None if not viable
        """
        try:
            # Gather data from multiple sources
            tasks = []
            for source in self.product_config.trend_data_sources:
                tasks.append(self._fetch_trend_data(keyword, source))
            
            # Wait for all data to be collected
            results = await asyncio.gather(*tasks)
            
            # Aggregate and average the data
            search_volume = int(sum(r.get("search_volume", 0) for r in results) / len(results))
            
            # Determine trend direction by voting
            trend_votes = [r.get("trend", "stable") for r in results]
            trend_direction = max(set(trend_votes), key=trend_votes.count)
            
            # Average competition level
            competition_level = sum(r.get("competition", 0.5) for r in results) / len(results)
            
            # Determine seasonality if any source indicates it
            seasonal = any(r.get("seasonal", False) for r in results)
            
            # Average estimated margin
            estimated_margin = sum(r.get("estimated_margin", 0.2) for r in results) / len(results)
            
            # Generate related keywords using AI
            prompt = f"Generate 5 related product keywords for '{keyword}' in the '{category}' category."
            related_keywords_text = await generate_text(prompt, self.config.openai_api_key)
            related_keywords = [k.strip() for k in related_keywords_text.split(',')][:5]
            
            # Create the niche object
            niche = ProductNiche(
                name=keyword,
                category=category,
                search_volume=search_volume,
                competition_level=competition_level,
                trend_direction=trend_direction,
                seasonal=seasonal,
                estimated_margin=estimated_margin,
                related_keywords=related_keywords
            )
            
            logger.debug(f"Researched niche: {niche.name}, Score: {niche.score:.2f}")
            return niche
        
        except Exception as e:
            logger.error(f"Error researching niche '{keyword}': {e}")
            return None
    
    async def _generate_product_details(self, niche: ProductNiche) -> List[Product]:
        """
        Generate specific product ideas within a niche.
        
        Args:
            niche: Product niche to explore
            
        Returns:
            List[Product]: List of generated product ideas
        """
        products = []
        
        try:
            # Generate product ideas using AI
            prompt = f"""
            Generate {min(3, self.product_config.max_products_per_niche)} specific product ideas for the '{niche.name}' niche in the '{niche.category}' category.
            For each product, provide:
            1. Name
            2. Brief description (2-3 sentences)
            3. Estimated cost price (between ${self.product_config.min_product_price/2:.2f} and ${self.product_config.max_product_price/2:.2f})
            4. Recommended retail price
            5. 3-5 key features
            6. Estimated weight in kg
            7. Estimated dimensions (length, width, height in cm)
            """
            
            product_ideas_text = await generate_text(prompt, self.config.openai_api_key)
            
            # Parse the product ideas text and create Product objects
            # This is a simplified implementation; in a real system,
            # you'd want more sophisticated parsing
            product_sections = re.split(r'Product \d+:', product_ideas_text)
            product_sections = [p for p in product_sections if p.strip()]
            
            for section in product_sections:
                try:
                    # Extract product details using regex patterns
                    name_match = re.search(r'Name:?\s*(.+?)(?:\n|$)', section)
                    name = name_match.group(1).strip() if name_match else f"Unknown {niche.name} Product"
                    
                    desc_match = re.search(r'Description:?\s*(.+?)(?=\n\d\.|\n[A-Za-z]|$)', section, re.DOTALL)
                    description = desc_match.group(1).strip() if desc_match else f"A quality {niche.name} product."
                    
                    cost_match = re.search(r'(?:Cost|Estimated cost):?\s*\$?(\d+\.?\d*)', section)
                    estimated_cost = float(cost_match.group(1)) if cost_match else random.uniform(self.product_config.min_product_price/2, self.product_config.max_product_price/2)
                    
                    price_match = re.search(r'(?:Price|Recommended|Retail price):?\s*\$?(\d+\.?\d*)', section)
                    recommended_price = float(price_match.group(1)) if price_match else estimated_cost * (1 + random.uniform(0.3, 0.7))
                    
                    # Extract features
                    features_text = re.search(r'Features:?(.*?)(?=\d\.|\n[A-Za-z]|$)', section, re.DOTALL)
                    features = []
                    if features_text:
                        feature_lines = re.findall(r'[-•*]?\s*(.+?)(?:\n|$)', features_text.group(1))
                        features = [f.strip() for f in feature_lines if f.strip()]
                    
                    if not features:
                        features = [f"Quality {niche.name} product", "Durable construction", "Modern design"]
                    
                    # Extract weight
                    weight_match = re.search(r'Weight:?\s*(\d+\.?\d*)\s*kg', section)
                    weight = float(weight_match.group(1)) if weight_match else random.uniform(0.1, 5.0)
                    
                    # Extract dimensions
                    dimensions = {}
                    dim_match = re.search(r'Dimensions:?\s*(.+?)(?:\n|$)', section)
                    if dim_match:
                        dim_text = dim_match.group(1)
                        dim_values = re.findall(r'(\d+\.?\d*)', dim_text)
                        if len(dim_values) >= 3:
                            dimensions = {
                                "length": float(dim_values[0]),
                                "width": float(dim_values[1]),
                                "height": float(dim_values[2])
                            }
                    
                    if not dimensions:
                        dimensions = {
                            "length": random.uniform(10, 50),
                            "width": random.uniform(10, 50),
                            "height": random.uniform(5, 30)
                        }
                    
                    # Generate additional product details
                    product = Product(
                        id="auto",  # Will be auto-generated in __post_init__
                        name=name,
                        description=description,
                        niche=niche.name,
                        category=niche.category,
                        estimated_cost=estimated_cost,
                        recommended_price=recommended_price,
                        image_urls=[],  # Will be populated later
                        features=features,
                        specifications={
                            "Material": "Various",
                            "Color": "Multiple options",
                            "Brand": "FallnAI"
                        },
                        keywords=[niche.name] + niche.related_keywords[:3],
                        weight=weight,
                        dimensions=dimensions,
                        shipping_restrictions=[],
                        variants=[
                            {"name": "Standard", "price_modifier": 0.0}
                        ],
                        source_urls=[]
                    )
                    
                    products.append(product)
                    
                except Exception as e:
                    logger.error(f"Error parsing product section: {e}")
                    continue
            
            return products
        
        except Exception as e:
            logger.error(f"Error generating products for niche '{niche.name}': {e}")
            return []
    
    def search_category(self, category: str) -> List[ProductNiche]:
        """
        Search for product niches within a specific category.
        
        Args:
            category: Product category to search within
            
        Returns:
            List[ProductNiche]: List of discovered product niches
        """
        found_niches = []
        
        try:
            # Run asynchronous research tasks
            async def _search_category_async() -> List[ProductNiche]:
                # Generate keywords for the category using AI
                prompt = f"Generate 5 trending product niches in the '{category}' category for dropshipping."
                keywords_text = await generate_text(prompt, self.config.openai_api_key)
                keywords = [k.strip() for k in keywords_text.split(',')]
                
                # Research each keyword asynchronously
                tasks = []
                for keyword in keywords:
                    tasks.append(self._research_niche(category, keyword))
                
                # Wait for all research tasks to complete
                niches = await asyncio.gather(*tasks)
                
                # Filter out None values (failed researches)
                return [n for n in niches if n is not None]
            
            # Run the async function using asyncio
            found_niches = asyncio.run(_search_category_async())
            
            # Update the list of niches and save
            for niche in found_niches:
                # Check if the niche already exists
                existing = next((n for n in self.niches if n.name == niche.name), None)
                if existing:
                    # Update existing niche
                    existing_idx = self.niches.index(existing)
                    self.niches[existing_idx] = niche
                else:
                    # Add new niche
                    self.niches.append(niche)
            
            # Save updated niches
            self._save_niches()
            
            return found_niches
        
        except Exception as e:
            logger.error(f"Error searching category '{category}': {e}")
            return []
    
    def find_profitable_niches(self) -> List[ProductNiche]:
        """
