"""
Configuration module for FallnAI DropShipper
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger("FallnAI.Config")

@dataclass
class EcommerceConfig:
    """E-commerce platform configuration"""
    platform: str = "shopify"  # shopify, woocommerce, etc.
    api_key: str = ""
    api_secret: str = ""
    store_name: str = "fallnai-dropshipper"
    store_url: str = ""
    store_email: str = "contact@fallnaidropshipper.com"
    store_currency: str = "USD"
    payment_gateways: List[str] = field(default_factory=lambda: ["stripe", "paypal"])
    shipping_zones: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Generate store URL if not provided"""
        if not self.store_url and self.platform == "shopify":
            self.store_url = f"https://{self.store_name}.myshopify.com"

@dataclass
class ProductResearchConfig:
    """Product research configuration"""
    target_profit_margin: float = 0.30  # 30% minimum profit margin
    trend_data_sources: List[str] = field(default_factory=lambda: ["google_trends", "aliexpress", "amazon"])
    max_product_price: float = 100.0
    min_product_price: float = 10.0
    niche_categories: List[str] = field(default_factory=lambda: [
        "home_improvement", "pet_supplies", "kitchen_gadgets", 
        "beauty", "fitness", "tech_accessories"
    ])
    product_search_interval_days: int = 7
    max_products_per_niche: int = 5
    competitor_analysis_enabled: bool = True

@dataclass
class SupplierConfig:
    """Supplier integration configuration"""
    preferred_suppliers: List[str] = field(default_factory=lambda: ["aliexpress", "alibaba", "dhgate"])
    supplier_apis: Dict[str, Dict[str, str]] = field(default_factory=dict)
    supplier_criteria: Dict[str, float] = field(default_factory=lambda: {
        "min_rating": 4.5,
        "min_order_count": 500,
        "max_shipping_days": 20
    })
    inventory_check_interval_hours: int = 4

@dataclass
class OrderProcessingConfig:
    """Order processing configuration"""
    auto_fulfill_orders: bool = True
    order_check_interval_minutes: int = 15
    send_customer_notifications: bool = True
    payment_verification_required: bool = True
    fraud_check_enabled: bool = True
    order_tracking_enabled: bool = True

@dataclass
class CustomerServiceConfig:
    """Customer service automation configuration"""
    enabled: bool = True
    check_interval_minutes: int = 10
    auto_response_templates: Dict[str, str] = field(default_factory=dict)
    escalation_keywords: List[str] = field(default_factory=lambda: [
        "refund", "damaged", "complaint", "supervisor", "manager"
    ])
    auto_refund_threshold: float = 20.0  # Auto-refund orders below this amount if complaint seems valid

@dataclass
class AnalyticsConfig:
    """Analytics and optimization configuration"""
    analysis_interval_hours: int = 24
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "conversion_rate", "cart_abandonment", "profit_margin", "customer_acquisition_cost"
    ])
    optimization_strategies: List[str] = field(default_factory=lambda: [
        "price_adjustment", "inventory_optimization", "marketing_budget_allocation"
    ])
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_profit_margin": 0.15,
        "max_customer_acquisition_cost": 15.0,
        "min_conversion_rate": 0.02
    })

@dataclass
class Config:
    """Main configuration for FallnAI DropShipper"""
    # General configuration
    business_name: str = "FallnAI DropShipper"
    business_email: str = "contact@fallnaidropshipper.com"
    operation_interval: int = 60  # seconds
    data_directory: str = "data"
    openai_api_key: str = ""
    proxy_settings: Optional[Dict[str, str]] = None
    
    # Component-specific configurations
    ecommerce: EcommerceConfig = field(default_factory=EcommerceConfig)
    product_research: ProductResearchConfig = field(default_factory=ProductResearchConfig)
    supplier: SupplierConfig = field(default_factory=SupplierConfig)
    order_processing: OrderProcessingConfig = field(default_factory=OrderProcessingConfig)
    customer_service: CustomerServiceConfig = field(default_factory=CustomerServiceConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    
    def save(self, path: str = "config.json") -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            path: Path to save the configuration file
        """
        try:
            with open(path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

def load_config(path: str = "config.json") -> Config:
    """
    Load configuration from a JSON file. If the file doesn't exist,
    create a default configuration and save it.
    
    Args:
        path: Path to the configuration file
    
    Returns:
        Config: Configuration object
    """
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            # Create a default config
            config = Config()
            
            # Update only the fields present in the file
            update_config_from_dict(config, config_dict)
            
            logger.info(f"Configuration loaded from {path}")
            return config
        else:
            # Create and save default configuration
            config = Config()
            config.save(path)
            logger.info(f"Default configuration created at {path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.info("Using default configuration")
        return Config()

def update_config_from_dict(config: object, config_dict: Dict[str, Any]) -> None:
    """
    Recursively update configuration from a dictionary.
    
    Args:
        config: Configuration object to update
        config_dict: Dictionary containing configuration values
    """
    for key, value in config_dict.items():
        if hasattr(config, key):
            if isinstance(value, dict) and not isinstance(getattr(config, key), (str, int, float, bool, list)):
                # Recursively update nested config objects
                update_config_from_dict(getattr(config, key), value)
            else:
                setattr(config, key, value)
