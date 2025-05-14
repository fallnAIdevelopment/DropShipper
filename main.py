#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

# Local modules
from config import Config, load_config
from product_research import ProductResearcher
from supplier_integration import SupplierManager
from ecommerce_platform import StoreManager
from listing_management import ListingManager
from order_processing import OrderProcessor
from customer_service import CustomerServiceBot
from analytics import AnalyticsEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("falln_ai.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FallnAI")

class FallnAI:
    """
    Main application class that orchestrates the entire autonomous dropshipping system.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the FallnAI Dropshipping system.
        
        Args:
            config_path: Path to the configuration file
        """
        logger.info("Initializing FallnAI DropShipper...")
        self.config = load_config(config_path)
        
        # Initialize components
        self.product_researcher = ProductResearcher(self.config)
        self.supplier_manager = SupplierManager(self.config)
        self.store_manager = StoreManager(self.config)
        self.listing_manager = ListingManager(self.config)
        self.order_processor = OrderProcessor(self.config)
        self.customer_service = CustomerServiceBot(self.config)
        self.analytics = AnalyticsEngine(self.config)
        
        # Create data directories if they don't exist
        self._setup_directories()
        
        logger.info("FallnAI DropShipper initialized successfully")
    
    def _setup_directories(self) -> None:
        """Create necessary directories for data storage."""
        dirs = [
            "data",
            "data/products",
            "data/suppliers", 
            "data/orders",
            "data/customers",
            "data/analytics",
            "data/listings",
            "logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Directory structure created")
    
    def setup(self) -> bool:
        """
        Set up the dropshipping business from scratch.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            logger.info("Starting business setup...")
            
            # 1. Set up e-commerce store
            logger.info("Setting up e-commerce store...")
            store_setup_success = self.store_manager.setup_store()
            if not store_setup_success:
                logger.error("Failed to set up e-commerce store")
                return False
            
            # 2. Research products
            logger.info("Researching profitable products...")
            product_niches = self.product_researcher.find_profitable_niches()
            if not product_niches:
                logger.error("Failed to find profitable product niches")
                return False
            
            selected_products = self.product_researcher.select_products(product_niches)
            if not selected_products:
                logger.error("Failed to select products")
                return False
            
            # 3. Find suppliers
            logger.info("Finding reliable suppliers...")
            suppliers = self.supplier_manager.find_suppliers(selected_products)
            if not suppliers:
                logger.error("Failed to find suppliers")
                return False
            
            # 4. Create product listings
            logger.info("Creating product listings...")
            for product in selected_products:
                self.listing_manager.create_listing(product)
            
            logger.info("Business setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False
    
    def run(self) -> None:
        """
        Run the main business operation loop.
        """
        logger.info("Starting FallnAI DropShipper operation...")
        
        try:
            while True:
                # Process new orders
                new_orders = self.order_processor.fetch_new_orders()
                if new_orders:
                    logger.info(f"Processing {len(new_orders)} new orders...")
                    for order in new_orders:
                        self.order_processor.process_order(order)
                
                # Handle customer inquiries
                self.customer_service.handle_inquiries()
                
                # Update inventory and prices
                self.supplier_manager.update_inventory()
                self.listing_manager.update_listings()
                
                # Analyze performance and optimize
                if self.analytics.should_run_analysis():
                    analysis_results = self.analytics.run_analysis()
                    self.optimize_business(analysis_results)
                
                # Research new products periodically
                if self.product_researcher.should_find_new_products():
                    new_niches = self.product_researcher.find_profitable_niches()
                    if new_niches:
                        new_products = self.product_researcher.select_products(new_niches)
                        if new_products:
                            for product in new_products:
                                suppliers = self.supplier_manager.find_suppliers([product])
                                if suppliers:
                                    self.listing_manager.create_listing(product)
                
                # Sleep to avoid hammering APIs
                time.sleep(self.config.operation_interval)
                
        except KeyboardInterrupt:
            logger.info("Gracefully shutting down FallnAI DropShipper...")
        except Exception as e:
            logger.error(f"Operation error: {str(e)}")
            logger.info("Attempting to restart operation...")
            self.run()  # Restart the operation loop
    
    def optimize_business(self, analysis_results: Dict[str, Any]) -> None:
        """
        Optimize business operations based on analytics results.
        
        Args:
            analysis_results: Dictionary of analysis metrics and recommendations
        """
        logger.info("Optimizing business operations...")
        
        # Adjust pricing
        if "pricing_recommendations" in analysis_results:
            self.listing_manager.adjust_prices(analysis_results["pricing_recommendations"])
        
        # Adjust marketing
        if "marketing_recommendations" in analysis_results:
            self.store_manager.adjust_marketing(analysis_results["marketing_recommendations"])
        
        # Product portfolio optimization
        if "product_recommendations" in analysis_results:
            # Remove underperforming products
            for product_id in analysis_results["product_recommendations"].get("remove", []):
                self.listing_manager.remove_listing(product_id)
            
            # Add new products in promising categories
            for category in analysis_results["product_recommendations"].get("add_categories", []):
                self.product_researcher.search_category(category)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FallnAI DropShipper - Autonomous Dropshipping Solution")
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--setup-only", action="store_true", help="Only run the initial setup and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize the system
    falln_ai = FallnAI(args.config)
    
    # Check if store is already set up
    if not os.path.exists("data/store_info.json"):
        setup_success = falln_ai.setup()
        if not setup_success:
            logger.error("Initial setup failed. Exiting.")
            sys.exit(1)
        
        if args.setup_only:
            logger.info("Setup completed successfully. Exiting as requested.")
            sys.exit(0)
    
    # Run the main business operation loop
    falln_ai.run()
