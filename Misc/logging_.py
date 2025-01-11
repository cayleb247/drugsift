import logging

# Configure logging
logging.basicConfig(
    filename='app.log',  # Log file name
    level=logging.INFO,  # Minimum logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)
