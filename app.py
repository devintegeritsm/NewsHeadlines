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

# Validate environment variables
if not all([SUPABASE_URL, SUPABASE_KEY]):
    logger.error("Missing required environment variables. Please check your .env file.")
    for var in ['SUPABASE_URL', 'SUPABASE_KEY']:
        if not os.getenv(var):
            logger.error(f"Missing {var}")
    raise ValueError("Missing required environment variables")

logger.info(f"Supabase URL: {SUPABASE_URL}")
logger.info("Initializing Supabase client...")

try:
    # Initialize Supabase client with admin privileges for full access
    # supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
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
    try:
        # Get dates with available articles for the last 7 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Fetching dates with articles from {start_date} to {end_date}")
        
        # Get unique dates that have articles in Supabase
        response = supabase.table("content") \
                           .select("date") \
                           .eq("content_type", "article") \
                           .gte("date", start_date.strftime("%Y-%m-%d")) \
                           .lte("date", end_date.strftime("%Y-%m-%d")) \
                           .execute()
        
        # Extract unique dates with articles
        unique_dates = set()
        for item in response.data:
            unique_dates.add(item['date'])
        
        logger.info(f"Found {len(unique_dates)} dates with articles")
        
        # Render the template with dates that have articles
        return render_template('index.html', dates_with_articles=list(unique_dates))
        
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return render_template('index.html', error=str(e))

@app.route('/api/headlines')
def get_headlines():
    """Get headlines for the last 7 days from available articles."""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Fetching article-based headlines from {start_date} to {end_date}")
        
        try:
            # Query Supabase for articles with detailed logging
            logger.info("Executing Supabase query for articles...")
            
            # Get unique dates that have articles in Supabase
            date_response = supabase.table("content") \
                                   .select("date") \
                                   .eq("content_type", "article") \
                                   .gte("date", start_date.strftime("%Y-%m-%d")) \
                                   .lte("date", end_date.strftime("%Y-%m-%d")) \
                                   .execute()
            
            # Extract unique dates
            unique_dates = set()
            for item in date_response.data:
                unique_dates.add(item['date'])
            
            logger.info(f"Found articles for {len(unique_dates)} unique dates")
            
            headlines = []
            
            # For each date, get the articles and create a headline entry
            for date in sorted(unique_dates, reverse=True):
                # Get articles for this date
                articles_response = supabase.table("content") \
                                           .select("*") \
                                           .eq("content_type", "article") \
                                           .eq("date", date) \
                                           .limit(1000) \
                                           .execute()
                
                logger.info(f"Found {len(articles_response.data)} articles for date {date}")
                
                if articles_response.data:
                    # Format headline data for this date
                    headline_data = {
                        'date': date,
                        'article_count': len(articles_response.data),
                        'articles': []
                    }
                    
                    # Add article info
                    for article in articles_response.data:
                        try:
                            article_filename = article['filename']
                            article_title = article_filename.replace('-', ' ').replace('.md', '').title()
                            
                            article_info = {
                                'filename': article_filename,
                                'title': article_title,
                                'url': f"/api/content/{date}/article/{article_filename}",
                            }
                            
                            # Add political score if it exists and is not NULL
                            if 'score' in article and article['score'] is not None:
                                article_info['score'] = article['score']
                            
                            # Check if there's a corresponding audio file
                            audio_filename = article_filename.replace('.md', '.wav')
                            audio_response = supabase.table("content") \
                                                   .select("*") \
                                                   .eq("content_type", "audio") \
                                                   .eq("date", date) \
                                                   .eq("filename", audio_filename) \
                                                   .execute()
                            
                            if audio_response.data:
                                article_info['audio_url'] = f"/api/content/{date}/audio/{audio_filename}"
                            
                            headline_data['articles'].append(article_info)
                            
                        except KeyError as ke:
                            logger.error(f"Missing key in article data: {ke}")
                            logger.error(f"Article data: {article}")
                    
                    headlines.append(headline_data)
            
            logger.info(f"Successfully processed headlines for {len(headlines)} dates")
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
        if actual_filename.endswith('.mp3') or actual_filename.endswith('.wav'):
            actual_content_type = "audio"
        elif "article" in filename or actual_filename.endswith('.md'):
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
            if actual_filename.endswith('.mp3') or actual_filename.endswith('.wav'):
                audio_type = 'audio/mpeg' if actual_filename.endswith('.mp3') else 'audio/wav'
                return jsonify({
                    'success': True,
                    'url': url,
                    'content_type': audio_type
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting Flask application on port {port}")
    app.run(host='0.0.0.0', port=port) 