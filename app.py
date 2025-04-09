from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import logging
import socket
from postgrest.exceptions import APIError
from flask_cors import CORS
import traceback
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Validate environment variables
if not all([SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY]):
    logger.error("Missing required environment variables. Please check your .env file.")
    for var in ['SUPABASE_URL', 'SUPABASE_KEY', 'SUPABASE_SERVICE_KEY']:
        if not os.getenv(var):
            logger.error(f"Missing {var}")
    raise ValueError("Missing required environment variables")

logger.info(f"Supabase URL: {SUPABASE_URL}")
logger.info("Initializing Supabase client...")

try:
    # Initialize Supabase client with admin privileges for full access
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Successfully initialized Supabase client with admin privileges")
    
    # Test the connection by making a simple query
    test_response = supabase.table("content").select("count").limit(1).execute()
    logger.info("Successfully tested Supabase connection")
    
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

@app.route('/')
def index():
    """Render the main page with headlines."""
    return render_template('index.html')

@app.route('/api/headlines')
def get_headlines():
    """Get headlines for the last 7 days."""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Fetching headlines from {start_date} to {end_date}")
        
        try:
            # Query Supabase for headlines with detailed logging
            logger.info("Executing Supabase query...")
            
            # First, check if the table exists and has data
            test_query = supabase.table("content").select("count").execute()
            logger.info(f"Table test query response: {test_query}")
            
            # Main query
            response = supabase.table("content") \
                             .select("*") \
                             .eq("content_type", "headlines") \
                             .gte("date", start_date.strftime("%Y-%m-%d")) \
                             .lte("date", end_date.strftime("%Y-%m-%d")) \
                             .execute()
            
            logger.info(f"Supabase query response: {response}")
            logger.info(f"Response data: {response.data}")
            
            headlines = []
            for item in response.data:
                try:
                    logger.info(f"Processing item: {item}")
                    headlines.append({
                        'date': item['date'],
                        'url': item['public_url'],
                        'filename': item['filename']
                    })
                except KeyError as ke:
                    logger.error(f"Missing key in item data: {ke}")
                    logger.error(f"Item data: {item}")
            
            logger.info(f"Successfully processed {len(headlines)} headlines")
            return jsonify({
                'success': True,
                'headlines': headlines
            })
            
        except APIError as ae:
            logger.error(f"Supabase API Error: {str(ae)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'error': f"Database error: {str(ae)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error fetching headlines: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/content/<date>/<content_type>/<path:filename>')
def get_content(date, content_type, filename):
    """Get content URL for a specific file."""
    try:
        logger.info(f"Fetching content for {date}/{content_type}/{filename}")
        logger.info(f"Query parameters - date: {date}, content_type: {content_type}, filename: {filename}")
        
        # First, let's check what content we have in the database
        test_query = supabase.table("content").select("content_type, date, filename").execute()
        logger.info(f"All available content: {test_query.data}")
        
        # Extract just the filename from the path
        actual_filename = filename.split('/')[-1]
        logger.info(f"Using filename: {actual_filename}")
        
        # Determine the correct content type
        if filename.endswith('.mp3'):
            actual_content_type = "audio"
        elif "article" in filename:
            actual_content_type = "article"
        else:
            actual_content_type = content_type
        logger.info(f"Using content type: {actual_content_type}")
        
        # Let's see what files we have for this date and content type
        date_content_query = supabase.table("content").select("*").eq("date", date).eq("content_type", actual_content_type).execute()
        logger.info(f"Content for date {date} and type {actual_content_type}: {date_content_query.data}")
        
        # Query Supabase for the specific content
        response = supabase.table("content").select("*").eq("date", date).eq("content_type", actual_content_type).eq("filename", actual_filename).execute()
        
        logger.info(f"Supabase response: {response}")
        logger.info(f"Response data: {response.data}")
        
        if response.data:
            logger.info("Content found successfully")
            url = response.data[0]['public_url']
            logger.info(f"Public URL: {url}")
            
            # If it's an HTML or Markdown file, serve it directly with proper content type
            if actual_filename.endswith('.html') or actual_filename.endswith('.md'):
                try:
                    content = requests.get(url).text
                    logger.info("Successfully fetched content")
                    
                    # Set appropriate content type based on file extension
                    content_type = 'text/html' if actual_filename.endswith('.html') else 'text/markdown'
                    return content, 200, {
                        'Content-Type': f'{content_type}; charset=utf-8',
                        'Content-Disposition': 'inline'
                    }
                except Exception as e:
                    logger.error(f"Error fetching content: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return jsonify({
                        'success': False,
                        'error': f"Error fetching content: {str(e)}"
                    }), 500
            
            # For audio files, return the URL directly
            if actual_filename.endswith('.mp3'):
                return jsonify({
                    'success': True,
                    'url': url
                })
            
            return jsonify({
                'success': True,
                'url': url
            })
        else:
            # Let's try to find any content with this filename, regardless of content_type
            fallback_query = supabase.table("content").select("*").eq("filename", actual_filename).execute()
            logger.info(f"Fallback query results: {fallback_query.data}")
            
            logger.warning(f"Content not found: {date}/{actual_content_type}/{actual_filename}")
            return jsonify({
                'success': False,
                'error': 'Content not found',
                'details': {
                    'date': date,
                    'content_type': actual_content_type,
                    'filename': actual_filename,
                    'available_content': test_query.data,
                    'date_specific_content': date_content_query.data
                }
            }), 404
    except Exception as e:
        logger.error(f"Error fetching content: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def find_available_port(start_port=5000, max_port=5050):
    """Find an available port to run the Flask app."""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise OSError(f"No available ports found between {start_port} and {max_port}")

if __name__ == '__main__':
    try:
        port = find_available_port()
        logger.info(f"Starting Flask application on port {port}")
        app.run(debug=True, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 