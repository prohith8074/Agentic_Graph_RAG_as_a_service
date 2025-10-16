"""
Utility functions and helpers for the Lyzr Challenge RAG system.
Includes logging setup, data validation, and common operations.
"""

import logging
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from config.settings import settings

def setup_logging():
    """Set up logging configuration for the application."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Configure logging with UTF-8 encoding for Windows compatibility
    import sys

    # Create a custom handler for console output that handles Unicode properly
    class UnicodeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
            except UnicodeEncodeError:
                # Fallback: encode the message and try again
                msg = self.format(record)
                try:
                    # Try UTF-8 encoding
                    self.stream.write(msg + self.terminator)
                    self.stream.flush()
                except UnicodeEncodeError:
                    # Last resort: replace problematic characters
                    safe_msg = msg.encode('utf-8', errors='replace').decode('utf-8')
                    self.stream.write(safe_msg + self.terminator)
                    self.stream.flush()

    # Set console encoding to UTF-8 for Windows
    if sys.platform == 'win32':
        try:
            # Force UTF-8 output for console
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7 doesn't have reconfigure
            pass

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(f"logs/lyzr_challenge_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
            UnicodeStreamHandler(sys.stdout)
        ],
        encoding='utf-8'  # Ensure UTF-8 encoding for console output
    )

    # Suppress some noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")

def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file

    Returns:
        True if file exists, False otherwise
    """
    if os.path.exists(file_path):
        return True
    else:
        logger = logging.getLogger(__name__)
        logger.error(f"File not found: {file_path}")
        return False

def save_json_to_file(data: Any, file_path: str) -> bool:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger = logging.getLogger(__name__)
        logger.info(f"Data saved to {file_path}")
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False

def load_json_from_file(file_path: str) -> Optional[Any]:
    """
    Load data from JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger = logging.getLogger(__name__)
        logger.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None

def format_query_result(result: Dict[str, Any]) -> str:
    """
    Format query result for display.

    Args:
        result: Query result dictionary

    Returns:
        Formatted string representation
    """
    output = []
    output.append(f"Query: {result.get('query', 'Unknown')}")

    if 'routing' in result:
        routing = result['routing']
        output.append(f"Routing: {routing.get('method', 'Unknown')} (confidence: {routing.get('confidence', 0):.2f})")
        output.append(f"Reasoning: {routing.get('reasoning', 'Unknown')}")

    output.append(f"Answer: {result.get('answer', 'No answer')}")

    if result.get('chunks_retrieved', 0) > 0:
        output.append(f"Chunks retrieved: {result['chunks_retrieved']}")

    return "\n".join(output)

def calculate_text_stats(text: str) -> Dict[str, Any]:
    """
    Calculate basic statistics for text content.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'line_count': 0,
            'avg_word_length': 0
        }

    chars = len(text)
    words = len(text.split())
    lines = len(text.split('\n'))

    avg_word_length = sum(len(word) for word in text.split()) / words if words > 0 else 0

    return {
        'char_count': chars,
        'word_count': words,
        'line_count': lines,
        'avg_word_length': round(avg_word_length, 2)
    }

def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix

def safe_get_nested_value(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely get nested dictionary value.

    Args:
        data: Dictionary to search
        keys: List of keys to traverse
        default: Default value if not found

    Returns:
        Value or default
    """
    current = data
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """
    Create a simple progress bar string.

    Args:
        current: Current progress value
        total: Total value
        width: Width of progress bar

    Returns:
        Progress bar string
    """
    if total == 0:
        return "[████████████████████] 100%"

    progress = min(current / total, 1.0)
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    percentage = int(progress * 100)

    return f"[{bar}] {percentage}%"

def log_performance(func_name: str, start_time: float, end_time: float, **kwargs):
    """
    Log performance information for operations.

    Args:
        func_name: Name of the function/operation
        start_time: Start time
        end_time: End time
        **kwargs: Additional context information
    """
    duration = end_time - start_time
    logger = logging.getLogger(__name__)

    context_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"Performance - {func_name}: {duration:.2f}s ({context_str})")

def get_system_info() -> Dict[str, Any]:
    """
    Get system and configuration information.

    Returns:
        Dictionary with system info
    """
    return {
        'python_version': os.sys.version,
        'working_directory': os.getcwd(),
        'config': {
            'llm_model': settings.GROQ_MODEL,
            'cohere_model': settings.COHERE_MODEL,
            'context_window_size': settings.CONTEXT_WINDOW_SIZE,
            'rate_limit_delay': settings.RATE_LIMIT_DELAY
        },
        'timestamp': datetime.now().isoformat()
    }