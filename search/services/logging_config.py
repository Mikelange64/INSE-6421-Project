"""
Logging configuration for the AI Paper Aggregator
Provides consistent logging across all modules with console and file output
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging():
    """
    Configure logging for the entire application
    Creates a logs/ directory and sets up both console and file handlers
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(levelname)s] %(name)s - %(message)s'
    )
    
    # Console handler (INFO level - less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (DEBUG level - detailed)
    file_handler = RotatingFileHandler(
        log_dir / 'debug.log',
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name):
    """
    Get a logger instance for a specific module
    
    Usage:
        from search.services.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    
    Args:
        name: Usually __name__ of the calling module
    
    Returns:
        logging.Logger instance
    """
    # Ensure logging is set up
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


# Setup logging when module is imported
setup_logging()

