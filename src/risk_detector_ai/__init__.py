import os
from flask import Flask
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_app(test_config=None):
    """Create and configure Flask application (delegates to app.py)"""
    from .app import create_app as create_app_impl
    return create_app_impl()
