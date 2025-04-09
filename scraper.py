from playwright.sync_api import sync_playwright
import json
import os
import re
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse
from datetime import datetime
import time
import soundfile as sf
import numpy as np
from f5_tts_mlx.generate import generate
import base64
from supabase_api import delete_headlines_from_supabase, get_files_from_supabase, upload_content_to_supabase, ensure_storage_bucket
from bs4 import BeautifulSoup
import supabase
from supabase import create_client

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Configure the custom AI API
CUSTOM_AI_ENDPOINT = "http://127.0.0.1:1234"
MODEL_NAME = "gemma-3-12b-it"
AI_TIMEOUT = 300  # 5 minutes timeout (increased from 3 minutes)
AI_CHUNK_SIZE = 8192  # Buffer size for reading response

# Function to test the API functionality
def test_api_functionality():
    """Test the API functionality by checking if the content bucket exists."""
    print("Testing API functionality...")
    
    try:
        # Check if content bucket exists
        bucket_exists = ensure_storage_bucket()
        if bucket_exists:
            print("API test successful! Content bucket exists and is accessible.")
            return True
        else:
            print("API test failed. Content bucket does not exist or is not accessible.")
            return False
    except Exception as e:
        print(f"API test failed with error: {str(e)}")
        return False

def extract_date_from_url(url):
    """Extract date from URL or return current date if not found."""
    try:
        # Try to extract date from URL (e.g., april-3-2025)
        parts = url.split('/')
        for part in parts:
            if '-' in part and any(month in part.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
                # Parse the date string
                date_parts = part.split('-')
                if len(date_parts) >= 3:
                    month_str = date_parts[0].lower()
                    day = int(date_parts[1])
                    year = int(date_parts[2])
                    
                    # Convert month name to number
                    months = {
                        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
                    }
                    month = months.get(month_str, 1)
                    
                    return datetime(year, month, day)
    except Exception as e:
        print(f"Error extracting date from URL: {e}")
    
    # Return current date if extraction fails
    return datetime.now()

def get_date_based_filename(date, suffix="", extension=".md"):
    """Generate a filename based on the date in YYYY-MM-DD format"""
    if isinstance(date, str):
        # If date is already a string in the correct format, use it directly
        formatted_date = date
    else:
        # Format datetime object
        formatted_date = date.strftime("%Y-%m-%d")
    
    # Clean up the suffix
    if suffix:
        suffix = f"-{suffix.strip('-')}"
    
    return f"{formatted_date}{suffix}{extension}"

def get_article_folder_name(date=None):
    """Generate a folder name based on the article date."""
    if date is None:
        date = datetime.now()
    date_str = date.strftime("%Y-%m-%d")
    return f"{date_str}-article"

def file_exists(filename):
    """Check if a file exists."""
    return os.path.exists(filename)

def create_folder(folder_name):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
        return True
    return False

def extract_headlines(content):
    """
    Extract headlines from the content.
    Headlines in this content are typically:
    1. Short lines (less than 100 characters)
    2. Followed by a longer paragraph
    3. Often contain keywords like ":", "-", or are in title case
    4. Do not end with periods
    """
    print("Starting headline extraction...")
    headlines = []
    lines = content.split('\n')
    
    print(f"Processing {len(lines)} lines of content...")
    
    # Skip the first line which is usually "What matters now"
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        # Print first few characters of the line for debugging
        print(f"Processing line {i}: {line[:50]}...")
            
        # Check if this line is likely a headline
        if len(line) < 100 and not line.endswith('.'):  # Headlines are typically shorter and don't end with periods
            # Check if this line is followed by a longer paragraph
            if i < len(lines) - 1:
                next_line = lines[i+1].strip()
                
                # If the next line is empty, look at the line after that
                if not next_line and i < len(lines) - 2:
                    next_line = lines[i+2].strip()
                
                # If the next non-empty line is a longer paragraph, this is likely a headline
                if next_line and len(next_line) > 100:
                    # Additional checks to confirm it's a headline
                    if (':' in line or  # Common headline punctuation
                        '-' in line or
                        line.istitle() or  # Title case is common in headlines
                        any(word.isupper() for word in line.split()) or  # Contains uppercase words
                        (i > 0 and not lines[i-1].strip())):  # Preceded by a blank line
                        print(f"Found headline: {line}")
                        headlines.append(line)
    
    print(f"Finished processing. Found {len(headlines)} headlines.")
    if headlines:
        print("Headlines found:")
        for headline in headlines:
            print(f"- {headline}")
    else:
        print("No headlines found. Content might need different parsing.")
    
    return headlines

def filter_headlines_with_custom_ai(headlines):
    """
    Use custom AI API to filter out sports and vaccination related headlines.
    """
    print(f"Starting filtering process with {len(headlines)} headlines")
    
    prompt = f"""
    You are a content filter. Your task is to:
    
    1. Review the following list of news headlines
    2. Identify which headlines are related to sports or vaccination
    3. Return ONLY a JSON array of the headlines that are NOT related to sports or vaccination
    
    Here are the headlines to review:
    {json.dumps(headlines, indent=2)}
    
    Please return ONLY the JSON array of filtered headlines. Do not include any explanations or notes.
    """
    
    print(f"Connecting to custom AI API at {CUSTOM_AI_ENDPOINT}...")
    print(f"Using model: {MODEL_NAME}")
    print(f"Timeout set to {AI_TIMEOUT} seconds")
    
    try:
        print("Sending request to custom AI API...")
        start_time = time.time()
        
        # Use a session for better connection handling
        with requests.Session() as session:
            # Set a longer timeout for the connection and read operations
            session.mount('http://', requests.adapters.HTTPAdapter(
                max_retries=3,
                pool_connections=10,
                pool_maxsize=10
            ))
            
            # Send the request with streaming enabled
            response = session.post(
                f"{CUSTOM_AI_ENDPOINT}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": "You are a content filter that specializes in identifying non-sports and non-vaccination related headlines."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000
                },
                timeout=AI_TIMEOUT,
                stream=True  # Enable streaming to handle large responses
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Read the response in chunks
            print("Reading response from custom AI API...")
            response_content = ""
            for chunk in response.iter_content(chunk_size=AI_CHUNK_SIZE, decode_unicode=True):
                if chunk:
                    response_content += chunk
            
            elapsed_time = time.time() - start_time
            print(f"Received complete response from custom AI API in {elapsed_time:.2f} seconds")
            print(f"Response length: {len(response_content)} characters")
            
            # Parse the response
            try:
                result = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"Error parsing initial response: {e}")
                print(f"Response preview: {response_content[:200]}...")
                
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        print("Successfully extracted JSON from response")
                    except json.JSONDecodeError:
                        print("Failed to parse extracted JSON")
                        return headlines
                else:
                    print("Could not find JSON in response")
                    return headlines
        
        if 'choices' in result and len(result['choices']) > 0:
            print("Successfully processed headlines with custom AI")
            filtered_headlines_text = result['choices'][0]['message']['content']
            
            # Try to parse the JSON response
            try:
                # Clean up the response to ensure it's valid JSON
                filtered_headlines_text = filtered_headlines_text.strip()
                
                # Remove markdown code blocks if present
                if filtered_headlines_text.startswith('```json'):
                    filtered_headlines_text = filtered_headlines_text[7:]
                if filtered_headlines_text.endswith('```'):
                    filtered_headlines_text = filtered_headlines_text[:-3]
                filtered_headlines_text = filtered_headlines_text.strip()
                
                # Try to parse the cleaned JSON
                try:
                    filtered_headlines = json.loads(filtered_headlines_text)
                    print(f"Successfully parsed JSON with {len(filtered_headlines)} headlines")
                    print(f"Filtered out {len(headlines) - len(filtered_headlines)} headlines (sports/vaccination related)")
                    return filtered_headlines
                except json.JSONDecodeError as inner_e:
                    print(f"Error parsing cleaned JSON: {inner_e}")
                    print(f"Cleaned response text: {filtered_headlines_text}")
                    
                    # Try an alternative approach - extract array content
                    try:
                        # Look for array content between square brackets
                        array_match = re.search(r'\[(.*)\]', filtered_headlines_text, re.DOTALL)
                        if array_match:
                            array_content = array_match.group(1)
                            # Split by commas and clean each item
                            items = [item.strip().strip('"\'') for item in array_content.split(',')]
                            # Filter out empty items
                            items = [item for item in items if item]
                            print(f"Extracted {len(items)} headlines using regex")
                            print(f"Filtered out {len(headlines) - len(items)} headlines (sports/vaccination related)")
                            return items
                    except Exception as extract_e:
                        print(f"Error extracting array content: {extract_e}")
                    
                    # If all parsing attempts fail, return original headlines
                    return headlines
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Response text: {filtered_headlines_text}")
                return headlines  # Return all headlines if parsing fails
        else:
            print("Error: Unexpected response format from custom AI API")
            print(f"Response: {json.dumps(result, indent=2)}")
            return headlines  # Return all headlines if processing fails
            
    except requests.exceptions.ConnectionError:
        print(f"Connection error: Could not connect to custom AI API at {CUSTOM_AI_ENDPOINT}")
        print("Please ensure the API server is running and accessible")
        return headlines  # Return all headlines if connection fails
    except requests.exceptions.Timeout:
        print(f"Timeout error: Custom AI API did not respond within {AI_TIMEOUT} seconds")
        print("The API might be overloaded or experiencing issues")
        return headlines  # Return all headlines if timeout occurs
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"Status code: {e.response.status_code}")
        print(f"Response: {e.response.text}")
        return headlines  # Return all headlines if HTTP error occurs
    except Exception as e:
        print(f"Unexpected error when calling custom AI API: {type(e).__name__} - {e}")
        return headlines  # Return all headlines if any other error occurs

def format_content_with_headlines(content, title, filtered_headlines):
    """
    Format the content as markdown using the filtered headlines.
    """
    # Clean up the title
    clean_title = title.split('|')[0].strip()
    
    # Start with the title
    markdown = f"# {clean_title}\n\n"
    
    # Split content into paragraphs
    paragraphs = content.split('\n')
    
    # Track if we're in a section that should be included
    include_section = True
    
    # Process each paragraph
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Check if this is a section header
        is_header = paragraph.isupper() or paragraph.endswith(':')
        
        if is_header:
            # Check if this header is in our filtered list
            include_section = paragraph in filtered_headlines
            if include_section:
                markdown += f"## {paragraph}\n\n"
        elif include_section:
            # Include paragraph if we're in a section that should be included
            markdown += f"{paragraph}\n\n"
    
    return markdown

def upload_headlines(content, filename, date_str):
    """Save content to a file and upload to Supabase."""
    try:
        # Save content to a file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Content saved to: {filename}")
        
        # Clean supabase table
        delete_headlines_from_supabase(date_str)
        
        # Upload to Supabase
        try:
            success, url = upload_content_to_supabase(
                date_str=date_str,
                content_type='headlines',
                filename=filename,
                content=content
            )
            
            if success:
                print(f"Successfully uploaded {filename} to Supabase: {url}")
                return True
            else:
                print(f"Failed to upload {filename} to Supabase")
                return False
                
        except Exception as upload_error:
            print(f"Error uploading to Supabase: {upload_error}")
            return False
            
    except Exception as e:
        print(f"Error saving content: {e}")
        return False

def split_content_by_topics(content, headlines, article_folder):
    """Split content into separate files based on headlines."""
    try:
        # Get the date from the article folder name
        date_str = article_folder.split('-article')[0]

        files = get_files_from_supabase(f"{date_str}/article")
        # Split content into sections based on headlines
        sections = []
        current_section = []
        lines = content.split('\n')
        
        for line in lines:
            # Check if line matches any headline
            if any(headline.lower() in line.lower() for headline in headlines):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Create a file for each section
        for i, section in enumerate(sections):
            # Get the first line as the title
            title = section.split('\n')[0].strip()
            # Create a filename from the title
            filename = title.lower()
            # Remove special characters and replace spaces with hyphens
            filename = re.sub(r'[^\w\s-]', '', filename)  # Remove all special characters except hyphens
            filename = re.sub(r'[-\s]+', '-', filename)  # Replace multiple spaces/hyphens with single hyphen
            filename = filename.strip('-')  # Remove leading/trailing hyphens
            filename = f"{filename}.md"
            filepath = os.path.join(article_folder, filename)
            
            # Write the content to the file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(section)

            article_filename = os.path.basename(filepath)
            print(f"Precessing article file {article_filename}...")
            try:
                article_exists = any(f['name'] == article_filename for f in files)
                if article_exists:
                    print(f"Article file already exists in Supabase for {filepath}")
                    continue    
            except Exception as e:
                print(f"Error checking article file on Supabase: {str(e)}")
            
            # Upload the file to Supabase
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                success, url = upload_content_to_supabase(date_str, "article", article_filename, content)
                if success:
                    print(f"Successfully uploaded {filename} to Supabase: {url}")
                else:
                    print(f"Failed to upload {filename} to Supabase")
            except Exception as e:
                print(f"Error uploading {filename} to Supabase: {str(e)}")
        
        return True
    except Exception as e:
        print(f"Error splitting content: {str(e)}")
        return False

def save_topic_files(headlines, folder_name):
    """Save topic files to the specified folder"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    topic_files = []
    for headline in headlines:
        # Create safe filename
        safe_filename = re.sub(r'[^\w\s-]', '', headline.lower())
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename).strip('-')
        filepath = os.path.join(folder_name, f"{safe_filename}.md")
        topic_files.append(filepath)
    
    return topic_files

def clean_html_tags(content):
    """Remove HTML tags from content while preserving text and structure."""
    # First, preserve line breaks by replacing <br>, </p>, and </div> with newlines
    content = re.sub(r'<br[^>]*>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'</p>', '\n\n', content, flags=re.IGNORECASE)
    content = re.sub(r'</div>', '\n', content, flags=re.IGNORECASE)
    
    # Remove script and style tags and their content
    content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Replace common HTML entities
    content = content.replace('&nbsp;', ' ')
    content = content.replace('&amp;', '&')
    content = content.replace('&lt;', '<')
    content = content.replace('&gt;', '>')
    content = content.replace('&quot;', '"')
    content = content.replace('&#39;', "'")
    
    # Remove all remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Clean up whitespace while preserving line breaks
    content = re.sub(r' +', ' ', content)  # Multiple spaces to single space
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Multiple blank lines to double line break
    content = content.strip()
    
    print(f"Cleaned content length: {len(content)}")
    print("First 200 characters of cleaned content:")
    print(content[:200])
    
    return content

def process_news_content(content, article_date):
    """Process the news content and generate all necessary files."""
    try:
        # Clean HTML tags from content
        content = clean_html_tags(content)
        
        # Save original content
        content_file = get_date_based_filename(article_date)
        with open(content_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Content saved to: {content_file}")

        # Extract and save headlines
        print("Extracting headlines from content...")
        headlines = extract_headlines(content)
        print(f"Found {len(headlines)} headlines")
        
        headlines_file = get_date_based_filename(article_date, "headlines")
        with open(headlines_file, "w", encoding="utf-8") as f:
            f.write("\n".join(headlines))
        print(f"Headlines saved to: {headlines_file}")

        # Filter headlines with custom AI
        print("Filtering headlines with custom AI...")
        try:
            filtered_headlines = filter_headlines_with_custom_ai(headlines)
            print(f"Successfully filtered headlines. Kept {len(filtered_headlines)} headlines.")
        except Exception as e:
            print(f"Failed to filter headlines with custom AI: {str(e)}")
            print("Using original headlines as fallback...")
            filtered_headlines = headlines

        # Create article folder and split content
        article_folder = get_date_based_filename(article_date, "article", "")
        os.makedirs(article_folder, exist_ok=True)
        print(f"Created folder: {article_folder}")

        print("Splitting content by topics...")
        try:
            split_content_by_topics(content, filtered_headlines, article_folder)
            print("Content split successfully.")

            # Save topic files for reference
            topic_files = save_topic_files(filtered_headlines, article_folder)
            print(f"Created {len(topic_files)} article files.")
        except Exception as e:
            print(f"Error splitting content: {str(e)}")
            return False

        # Generate audio files
        try:
            generate_audio_for_articles(article_date)
            print("Audio files generated successfully.")
        except Exception as e:
            print(f"Error generating audio files: {str(e)}")

        # Generate HTML page
        try:
            generate_headlines_html(article_date)
            print("HTML page generated successfully.")
        except Exception as e:
            print(f"Error generating HTML page: {str(e)}")

        return True
    except Exception as e:
        print(f"Error processing news content: {str(e)}")
        return False

def scrape_first_news_link(url: str, force_refresh: bool = False):
    """Scrape the first news link found on the page."""
    with sync_playwright() as p:
        try:
            # Launch browser and create page
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            print(f"Attempting to scrape: {url}")
            
            # Navigate to the page
            print("Navigating to page...")
            page.goto(url, wait_until="networkidle")
            print("Page loaded, waiting for additional rendering...")
            
            # Find the first news link
            print("Looking for the first news link...")
            first_link = page.query_selector("a[href*='/briefs/']")
            if first_link:
                href = first_link.get_attribute("href")
                link_text = first_link.inner_text()
                print(f"Found first news link: {href}")
                print(f"Link text: {link_text}")
                
                # Construct absolute URL
                base_url = "https://news.iliane.xyz"
                article_url = base_url + href if href.startswith("/") else href
                
                # Extract date from the link text or href
                date_match = re.search(r'(\w+ \d+, \d{4})', link_text)
                if date_match:
                    article_date = datetime.strptime(date_match.group(1), "%B %d, %Y")
                else:
                    # Fallback to current date
                    article_date = datetime.now()
                
                print(f"\n{article_date.strftime('%B %d, %Y').lower()}")
                
                # Check if content already exists for this date
                content_file = get_date_based_filename(article_date)
                if os.path.exists(content_file) and not force_refresh:
                    print(f"Content already exists for {article_date}")
                    browser.close()
                    return True
                
                # Navigate to the article
                print(f"Navigating to the first news link: {article_url}")
                page.goto(article_url, wait_until="networkidle")
                print("Link page loaded, waiting for additional rendering...")
                
                # Get the page content and title
                content = page.content()
                title = page.title()
                print(f"Got content from linked page (length: {len(content)})")
                print(f"Page title: {title}")
                
                # Close browser before processing content
                browser.close()
                
                # Process the content
                return process_news_content(content, article_date)
            else:
                print("No news links found on the page.")
                browser.close()
                return False

        except Exception as e:
            print(f"Error scraping the website: {str(e)}")
            if 'browser' in locals():
                browser.close()
            return False

def generate_audio_for_articles(date=None):
    """Generate audio files for each article using f5-tts-mlx."""
    if date is None:
        date = datetime.now()
    date_str = date.strftime("%Y-%m-%d")
    article_folder = f"{date_str}-article"
    
    if not os.path.exists(article_folder):
        print(f"Article folder {article_folder} not found.")
        return False
    
    try:
        # Get all markdown files in the article folder
        article_files = [f for f in os.listdir(article_folder) if f.endswith('.md')]
        
        files = get_files_from_supabase(f"{date_str}/audio")

        # Process all articles
        for article_file in article_files:
            article_path = os.path.join(article_folder, article_file)
            audio_file = article_path.replace('.md', '.mp3')
            
                
            audio_filename = os.path.basename(audio_file)
            print(f"Precessing audio file {audio_filename}...")
            # Skip if audio file already exists in Supabase
            try:
                # List files in the audio directory
                file_exists = any(f['name'] == audio_filename for f in files)
                
                if file_exists:
                    print(f"Audio file already exists in Supabase for {article_file}")
                    continue
            except Exception as e:
                # If the directory doesn't exist or there's an error, proceed with generation
                print(f"Error checking audio file on Supabase: {str(e)}")
            
            # Read the article content
            with open(article_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove markdown formatting
            content = re.sub(r'#+ ', '', content)  # Remove headers
            content = re.sub(r'\*\*|\*|__|_', '', content)  # Remove bold/italic
            content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Remove links
            
            # Generate audio file using the f5-tts-mlx library directly
            print(f"Generating audio for {article_file}...")
            try:
                # Generate audio using the library with the output_path parameter
                generate(content, output_path=audio_file)
                print(f"Successfully generated audio for {article_file}")
                
                # Read the generated audio file
                with open(audio_file, 'rb') as f:
                    audio_data = f.read()
                
                # Upload to Supabase
                success, url = upload_content_to_supabase(
                    date_str=date_str,
                    content_type="audio",
                    filename=audio_filename,
                    content=audio_data,
                    is_binary=True
                )
                
                if success:
                    print(f"Successfully uploaded audio for {article_file}")
                    print(f"Audio URL: {url}")
                else:
                    print(f"Failed to upload audio for {article_file}")
                
                
            except Exception as e:
                print(f"Failed to generate audio for {article_file}: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error generating audio files: {e}")
        return False

def generate_headlines_html(date=None):
    """Generate HTML file with headlines and upload to Supabase."""
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y-%m-%d")
    folder_name = get_article_folder_name(date)
    
    # Get all markdown files in the folder
    markdown_files = [f for f in os.listdir(folder_name) if f.endswith('.md')]
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>News Headlines - {date_str}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .headline {{ margin: 10px 0; padding: 10px; border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
    <h1>News Headlines - {date_str}</h1>"""
    
    for md_file in markdown_files:
        # Read the first line of each markdown file as the headline
        with open(os.path.join(folder_name, md_file), 'r') as f:
            headline = f.readline().strip('# \n')
        
        html_content += f"""
    <div class="headline">
        <h2>{headline}</h2>
        <a href="/api/content/{date_str}/article/{md_file}">Read more</a>
        <a href="/api/content/{date_str}/audio/{md_file.replace('.md', '.mp3')}">Listen to audio</a>
    </div>"""
    
    html_content += """
</body>
</html>"""
    
    # Save and upload the HTML file
    html_filename = f"{date_str}-headlines.html"
    success = upload_headlines(html_content, html_filename, date_str)
    
    if success:
        print(f"Successfully generated and uploaded headlines HTML for {date_str}")
    else:
        print(f"Failed to generate or upload headlines HTML for {date_str}")

def main():
    """Main function to run the scraper."""
    try:
        print("Starting script...")
        
        # Test API functionality
        print("Testing API functionality...")
        api_test_success = test_api_functionality()
        if not api_test_success:
            print("API test failed. Exiting script.")
            return
        
        # Scrape the first news link with force_refresh=True
        result = scrape_first_news_link("https://news.iliane.xyz/briefs", force_refresh=True)
        if result:
            print("Script finished successfully.")
        else:
            print("Failed to scrape or process the data.")
        
        print("Script finished.")
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
