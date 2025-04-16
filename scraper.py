import math
from playwright.sync_api import sync_playwright
import json
import os
import re
from dotenv import load_dotenv
import requests
from datetime import datetime
import time
import soundfile as sf
import numpy as np
# from f5_tts_mlx.generate import generate
import base64
from supabase_api import get_files_from_supabase, update_score_in_supabase, upload_content_to_supabase, ensure_storage_bucket
from supabase import create_client
from kokoro_api import generate_audio
from unidecode import unidecode
import argparse


# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Configure the custom AI API
ENABLE_FILTERING_WITH_CUSTOM_AI = False
ENABLE_UPLOAD_TO_BACKEND = True
ENABLE_SCORING_WITH_CUSTOM_AI = True
UPDATE_SCORES_WITH_CUSTOM_AI = False
CUSTOM_AI_ENDPOINT = "http://127.0.0.1:1234"
# MODEL_NAME = "deepseek-r1-distill-qwen-7b"
MODEL_NAME = "claude-3.7-sonnet-reasoning-gemma3-12b"
AI_TIMEOUT = 300  # 5 minutes timeout (increased from 3 minutes)
AI_CHUNK_SIZE = 8192  # Buffer size for reading response

# TTS_PROVIDER = "f5"
# TTS_OUTPUT_FORMAT = ".mp3"

TTS_PROVIDER = "kokoro"
TTS_OUTPUT_FORMAT = ".wav"

# Function to check API availability and get available models
def check_api_availability():
    """Check if the API is available and get available models."""
    print(f"Checking LocalAPI availability at {CUSTOM_AI_ENDPOINT}...")
    
    try:
        response = requests.get(f"{CUSTOM_AI_ENDPOINT}/v1/models", timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            if 'data' in models_data and len(models_data['data']) > 0:
                # Get the first model from the list
                first_model = models_data['data'][0]['id']
                print(f"LocalAPI is available. First available model: {first_model}")
                return True, first_model
            else:
                print("LocalAPI is available but no models found.")
                return True, None
        else:
            print(f"LocalAPI check failed with status code: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"Local API availability check failed with error: {str(e)}")
        return False, None

# Function to test the API functionality
def test_api_functionality():
    """Test the API functionality by checking if it's available and content bucket exists."""
    
    if ENABLE_FILTERING_WITH_CUSTOM_AI or ENABLE_SCORING_WITH_CUSTOM_AI:
        # Check API availability and get models
        api_available, first_model = check_api_availability()
        if not api_available:
            return False
    
        # Update MODEL_NAME with the first available model if found
        # if first_model:
        #     global MODEL_NAME
        #     MODEL_NAME = first_model

        print(f"Local API is available. Using model: {MODEL_NAME}")
    
    if ENABLE_UPLOAD_TO_BACKEND:
        print("Testing remote API is available...")
        try:
            # Check if content bucket exists
            bucket_exists = ensure_storage_bucket()
            if bucket_exists:
                print("Remote API test successful! Content bucket exists and is accessible.")
                return True
            else:
                print("Remote API test failed. Content bucket does not exist or is not accessible.")
                return False
        except Exception as e:
            print(f"API test failed with error: {str(e)}")
            return False
    
    return True

def get_date_based_filename(date_str: str, suffix="", extension=".md"):
    """Generate a filename based on the date in YYYY-MM-DD format"""
    # Clean up the suffix
    if suffix:
        suffix = f"-{suffix.strip('-')}"
    
    return f"{date_str}{suffix}{extension}"

def get_article_folder_name(date_str: str):
    """Generate a folder name based on the article date."""
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
            
        #Check if this line is likely a headline
        if (len(line) <= 108 and not line.endswith('.')
            and not line.endswith(':')
            and line.startswith(HEADLINE_TAG)):  # Headlines are typically shorter and don't end with periods
            # Check if this line is followed by a longer paragraph
            if i < len(lines) - 1:
                next_line = lines[i+1].strip()
                
                if not next_line:  # Preceded by a blank line
                    print(f"Found headline: {line}")
                    headlines.append(line.replace(HEADLINE_TAG, ''))

    print(f"Finished processing. Found {len(headlines)} headlines.")
    if headlines:
        print("Headlines found:")
        for headline in headlines:
            print(f"- {headline}")
    else:
        print("No headlines found. Content might need different parsing.")
    
    return headlines

def split_content_by_topics(content, headlines, article_folder):
    """Split content into separate files based on headlines."""
    try:
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
                if current_section:
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
            filename = unidecode(filename)
            filename = f"{filename}.md"
            filepath = os.path.join(article_folder, filename)
            
            # Write the content to the file
            with open(filepath, 'w', encoding='utf-8') as f:
                section = section.replace(HEADLINE_TAG, '')
                f.write(section)

        return True
    except Exception as e:
        print(f"Error splitting content: {str(e)}")
        return False

HEADLINE_TAG = "*****"
STRONG_TAG = "#####"

def clean_html_tags(content):
    """Remove HTML tags from content while preserving text and structure."""

    # Truncate content after </article> tag
    tag = '</article>'
    tag_index = content.find(tag)

    if tag_index != -1:
        # Calculate the end position in the original content
        end_pos = tag_index + len(tag)
        content = content[:end_pos]

    # First, preserve line breaks by replacing <br>, </p>, and </div> with newlines
    content = re.sub(r'<br[^>]*>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'</p>', '\n\n', content, flags=re.IGNORECASE)
    content = re.sub(r'</div>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<strong>', STRONG_TAG, content, flags=re.IGNORECASE)
    # content = re.sub(r'<li>', '* <li>', content, flags=re.IGNORECASE)
    
    # Remove script and style tags and their content
    content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove h1,h2 tags and their content
    content = re.sub(r'<h1.*?</h1>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<h2.*?</h2>', '', content, flags=re.DOTALL | re.IGNORECASE)

    # Replace common HTML entities
    content = content.replace('&nbsp;', ' ')
    content = content.replace('&amp;', '&')
    content = content.replace('&lt;', '<')
    content = content.replace('&gt;', '>')
    content = content.replace('&quot;', '"')
    content = content.replace('&#39;', "'")

    # Remove all remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    content = content.replace(f'\n{STRONG_TAG}', f'\n{HEADLINE_TAG}')
    content = content.replace(STRONG_TAG, '')

    # Clean up whitespace while preserving line breaks
    content = re.sub(r' +', ' ', content)  # Multiple spaces to single space
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Multiple blank lines to double line break

    content = content.strip()
    
    print(f"Cleaned content length: {len(content)}")
    print("First 200 characters of cleaned content:")
    print(content[:200])
    
    return content

def process_news_content(content, article_date: str):
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

        # Create article folder and split content
        article_folder = get_date_based_filename(article_date, "article", "")
        os.makedirs(article_folder, exist_ok=True)
        print(f"Created folder: {article_folder}")

        print("Splitting content by topics...")
        try:
            split_content_by_topics(content, headlines, article_folder)
            print("Content split successfully.")
        except Exception as e:
            print(f"Error splitting content: {str(e)}")
            return False

        if ENABLE_FILTERING_WITH_CUSTOM_AI:
            # Filter articles with custom AI
            filter_articles_with_custom_ai(article_folder)
            
        if ENABLE_UPLOAD_TO_BACKEND:
            # Upload the filtered articles to Supabase
            upload_articles_to_supabase(article_date, article_folder)

        # Generate audio files for filtered articles
        try:
            generate_audio_for_articles(article_date, article_folder, TTS_PROVIDER)
            print("Audio files generated successfully.")
        except Exception as e:
            print(f"Error generating audio files: {str(e)}")

        return True
    except Exception as e:
        print(f"Error processing news content: {str(e)}")
        return False
    
def get_article_files(article_folder: str, extension: str = ".md"):
    """Get all article files from the article folder."""
    if not os.path.exists(article_folder):
        print(f"Article folder {article_folder} not found.")
        return []
    
    return [f for f in os.listdir(article_folder) if f.endswith(extension)] #[:5]
    

def filter_articles_with_custom_ai(article_folder: str):
    """Process the articles with custom AI."""
    article_files = get_article_files(article_folder)

    filtered_article_files = []

    print(f"Filtering {len(article_files)} articles with custom AI...")
    for article_file in article_files:
        article_path = os.path.join(article_folder, article_file)
        
        # Read article content
        with open(article_path, 'r', encoding='utf-8') as f:
            article_content = f.read()
        
        # Filter article
        should_exclude = filter_article_with_custom_ai(article_content)
        
        if should_exclude:
            print(f"Excluding article: {article_file} (matches exclusion criteria)")
            # Remove the file as it's excluded
            os.remove(article_path)
        else:
            print(f"Keeping article: {article_file}")
            filtered_article_files.append(article_file)

    print(f"Filtered articles. Kept {len(filtered_article_files)} out of {len(article_files)} articles.")


def upload_articles_to_supabase(date_str: str, article_folder):
    """Upload the filtered articles to Supabase."""

    article_files = get_article_files(article_folder)

    print(f"Uploading {len(article_files)} articles to Supabase from {article_folder} folder...")

    files = get_files_from_supabase(f"{date_str}/article")

    for article_file in article_files:
        article_path = os.path.join(article_folder, article_file)

        print(f"Precessing article file {article_file}...")
        try:
            article_exists = any(f['name'] == article_file for f in files)
            if article_exists:
                print(f"Article file already exists in Supabase for {article_file}")
                if UPDATE_SCORES_WITH_CUSTOM_AI:
                    print(f"Updating score for article file {article_file} in Supabase...")
                    score = score_article_with_custom_ai(article_path)
                    if score is not None:
                        update_score_in_supabase(date_str, "article", article_file, score)
                continue    
        except Exception as e:
            print(f"Error checking article file on Supabase: {str(e)}")
            
        # Upload the file to Supabase
        try:
            with open(article_path, 'r', encoding='utf-8') as f:
                content = f.read()

            score = score_article_with_custom_ai(content) if ENABLE_SCORING_WITH_CUSTOM_AI else None
            
            success, url = upload_content_to_supabase(date_str, "article", article_file, content, score=score)
            if success:
                print(f"Successfully uploaded {article_file} to Supabase: {url}")
            else:
                print(f"Failed to upload {article_file} to Supabase")
        except Exception as e:
            print(f"Error uploading {article_file} to Supabase: {str(e)}")


def filter_article_with_custom_ai(article_content):
    """
    Use custom AI API process an article.
    """
    print("Checking article content with custom AI...")

    prompt = f"""
    You are a content filter that checks if articles are related to any kind of sports or pro-vaccination. 
    Answer only YES or NO like the following: ANSWER:YES or ANSWER:NO
    
    Here the content to review:
    {article_content}    
    """

    print(f"Connecting to custom AI API at {CUSTOM_AI_ENDPOINT}...")
    print(f"Using model: {MODEL_NAME}")
    
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
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Lower temperature for more deterministic answers
                    "max_tokens": -1
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
            print(f"Received response from custom AI API in {elapsed_time:.2f} seconds")
            
            # Parse the response
            result = json.loads(response_content)
            
            if 'choices' in result and len(result['choices']) > 0:
                answer_raw = result['choices'][0]['message']['content'].strip()
                print(f"AI raw response: {answer_raw}")
                answer = answer_raw.split("ANSWER:")[1].strip()
                print(f"AI response: {answer}")
                
                # Return True if the answer is YES (exclude the article)
                return "YES" in answer
            else:
                print("Error: Unexpected response format from custom AI API")
                print(f"Response: {json.dumps(result, indent=2)}")
                return False  # Keep the article if there's an error
                
    except Exception as e:
        print(f"Error when calling custom AI API: {type(e).__name__} - {e}")
        return False  # Keep the article if there's an error


def score_article_with_custom_ai(article_content):
    """
    Use custom AI API process an article.
    """
    print("Checking article content with custom AI...")

    prompt = f"""
    Please assign a political bias score to the content below. The score must range from -1.0 (indicating strongly left-leaning sources/framing) to +1.0 (indicating strongly right-leaning sources/framing), with 0.0 representing neutrality or balance.
    Replay with the score number, like the following: SCORE:-0.3 or SCORE:0.0 or SCORE:0.5 and do not append any comment after.
    
    Here the content to review:
    {article_content}    
    """
    
    print(f"Connecting to custom AI API at {CUSTOM_AI_ENDPOINT}...")
    print(f"Using model: {MODEL_NAME}")
    
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
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,  # Lower temperature for more deterministic answers
                    "max_tokens": -1
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
            print(f"Received response from custom AI API in {elapsed_time:.2f} seconds")
            
            # Parse the response
            result = json.loads(response_content)
            
            if 'choices' in result and len(result['choices']) > 0:
                answer_raw = result['choices'][0]['message']['content'].strip()
                print(f"AI raw response: {answer_raw}")
                answer = answer_raw.split("SCORE:")[1].strip()
                answer = answer.split("\n")[0].strip()
                print(f"AI response: {answer}")
                
                return float(answer)
            else:
                print("Error: Unexpected response format from custom AI API")
                print(f"Response: {json.dumps(result, indent=2)}")
                return None
                
    except Exception as e:
        print(f"Error when calling custom AI API: {type(e).__name__} - {e}")
        return None

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
                    parsed_date = datetime.strptime(date_match.group(1), "%B %d, %Y")
                else:
                    # Fallback to current date
                    print(f"Cannot parse a link date")
                    return False
                
                # Format datetime object
                article_date = parsed_date.strftime("%Y-%m-%d")
                print(f"Parsed a link date {article_date}")
                
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

def generate_audio_for_articles(date_str: str, article_folder: str, provider: str):
    """Generate audio files for each article using tts."""
    
    try:
        # Get all markdown files in the article folder
        article_files = get_article_files(article_folder)
        
        files = get_files_from_supabase(f"{date_str}/audio") if ENABLE_UPLOAD_TO_BACKEND else []

        # Process all articles
        for article_file in article_files:
        # for article_file in article_files:
            article_path = os.path.join(article_folder, article_file)
            audio_file = article_path.replace('.md', TTS_OUTPUT_FORMAT)
                
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
                if provider == "f5":
                    # Generate audio using the library with the output_path parameter
                    # generate(content, output_path=audio_file)
                    raise Exception("{provider} is not installed")
                else:
                    # Generate audio using the kokoro API
                    generate_audio(content, audio_file)
                print(f"Successfully generated audio for {article_file}")
                
                if ENABLE_UPLOAD_TO_BACKEND:
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

def main():
    """Main function to run the scraper."""

    parser = argparse.ArgumentParser(description="A simple example script using argparse.")
    parser.add_argument("--ai-filtering", action="store_true", help="Enable AI filtering")
    parser.add_argument("--no-supabase", action="store_true", help="Disable upload to supabase")
    parser.add_argument("--no-scoring", action="store_true", help="Disable AI scoring")
    parser.add_argument("--update-scores", action="store_true", help="Update AI scores")
    args = parser.parse_args()

    global ENABLE_FILTERING_WITH_CUSTOM_AI
    global ENABLE_UPLOAD_TO_BACKEND
    global ENABLE_SCORING_WITH_CUSTOM_AI
    global UPDATE_SCORES_WITH_CUSTOM_AI
    if args.ai_filtering:
        ENABLE_FILTERING_WITH_CUSTOM_AI = True
        print("AI filtering enabled")
    if args.no_supabase:
        ENABLE_UPLOAD_TO_BACKEND = False
        print("Upload to supabase disabled")
    if args.no_scoring:
        ENABLE_SCORING_WITH_CUSTOM_AI = False
        print("AI scoring disabled")
    if args.update_scores:
        UPDATE_SCORES_WITH_CUSTOM_AI = True
        print("Updating AI scores")

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
