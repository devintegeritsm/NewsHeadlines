import requests
import json
from datetime import datetime, timedelta
import base64

# API Configuration
WEBSITE_API_URL = "https://ee698794-a909-4a3b-b7ff-a1e78102b549-00-1h9txqc27ajmx.kirk.replit.dev"
WEBSITE_API_USERNAME = "dev"
WEBSITE_API_PASSWORD = "aaaaaa"
WEBSITE_API_TOKEN = None

def authenticate_with_website_api():
    """Authenticate with the website API and get a token."""
    global WEBSITE_API_TOKEN
    
    url = f"{WEBSITE_API_URL}/api/auth/login"
    payload = {
        "username": WEBSITE_API_USERNAME,
        "password": WEBSITE_API_PASSWORD
    }
    
    try:
        print(f"Attempting to authenticate with {url}")
        response = requests.post(url, json=payload)
        print(f"Authentication response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Authentication failed with status {response.status_code}")
            print(f"Response content: {response.text}")
            return False
            
        result = response.json()
        WEBSITE_API_TOKEN = result.get("token")
        print(f"Successfully authenticated with website API")
        print(f"Token: {WEBSITE_API_TOKEN[:10]}... (truncated)")
        return True
    except Exception as e:
        print(f"Error authenticating with website API: {e}")
        return False

def upload_content_to_website(date_str, content_type, filename, content, is_binary=False):
    """
    Upload content to the website API.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        content_type: Type of content (article, headlines, audio)
        filename: Name of the file
        content: File content (binary for audio, text for HTML/markdown)
        is_binary: Whether the content is binary (for audio files)
    
    Returns:
        Success status and URL if successful
    """
    global WEBSITE_API_TOKEN
    
    # Ensure we have a valid token
    if not WEBSITE_API_TOKEN and not authenticate_with_website_api():
        return False, None
    
    # API endpoint
    url = f"{WEBSITE_API_URL}/api/content/upload"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {WEBSITE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Prepare content
    if is_binary:
        # For binary files like audio, encode as base64
        content_encoded = base64.b64encode(content).decode('utf-8')
    else:
        # For text files, just use the content as is
        content_encoded = content
    
    # Prepare payload
    payload = {
        "date": date_str,
        "content_type": content_type,
        "filename": filename,
        "content": content_encoded,
        "metadata": {
            "source": "newsapp",
            "generated_at": datetime.now().isoformat()
        }
    }
    
    # Send request
    try:
        print(f"Attempting to upload content to {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        response = requests.post(url, json=payload, headers=headers)
        print(f"Upload response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Upload failed with status {response.status_code}")
            print(f"Response content: {response.text}")
            return False, None
            
        response.raise_for_status()
        result = response.json()
        return True, result.get("url")
    except Exception as e:
        print(f"Error uploading content: {e}")
        # If token expired, try to re-authenticate and retry once
        if "401" in str(e) or "403" in str(e):
            print("Token may have expired, attempting to re-authenticate...")
            if authenticate_with_website_api():
                headers["Authorization"] = f"Bearer {WEBSITE_API_TOKEN}"
                try:
                    print("Retrying upload with new token...")
                    response = requests.post(url, json=payload, headers=headers)
                    print(f"Retry upload response status: {response.status_code}")
                    
                    if response.status_code != 200:
                        print(f"Retry upload failed with status {response.status_code}")
                        print(f"Response content: {response.text}")
                        return False, None
                        
                    response.raise_for_status()
                    result = response.json()
                    return True, result.get("url")
                except Exception as retry_error:
                    print(f"Error on retry upload: {retry_error}")
        return False, None

def test_api_functionality():
    """Test the API functionality with realistic content from yesterday."""
    print("Testing API functionality...")
    
    # First authenticate
    if not authenticate_with_website_api():
        print("Authentication failed. Cannot proceed with API test.")
        return False
    
    # Get yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Testing with date: {yesterday}")
    
    # Test cases with realistic content
    test_cases = [
        {
            "content_type": "headlines",
            "filename": f"{yesterday}-headlines.html",
            "content": f"""<!DOCTYPE html>
<html>
<head>
    <title>News Headlines - {yesterday}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .headline {{ margin: 10px 0; padding: 10px; border-bottom: 1px solid #eee; }}
    </style>
</head>
<body>
    <h1>News Headlines - {yesterday}</h1>
    <div class="headline">
        <h2>Test Article: Major Breakthrough in AI Technology</h2>
        <p>Preview: Scientists announce revolutionary advances in artificial intelligence...</p>
        <a href="{yesterday}/article/test-article.md">Read more</a>
        <a href="{yesterday}/audio/test-article.mp3">Listen to audio</a>
    </div>
</body>
</html>""",
            "is_binary": False
        },
        {
            "content_type": "article",
            "filename": "test-article.md",
            "content": f"""# Major Breakthrough in AI Technology

*Published on {yesterday}*

Scientists have announced a revolutionary breakthrough in artificial intelligence technology that promises to transform the way we interact with machines. The new system, developed through a collaborative effort between leading research institutions, demonstrates unprecedented capabilities in natural language understanding and problem-solving.

## Key Developments

- Advanced neural network architecture
- Improved learning efficiency
- Enhanced decision-making capabilities
- Real-world applications in healthcare and education

## Impact and Future Implications

This breakthrough represents a significant step forward in the field of artificial intelligence. Experts predict that this technology will lead to more sophisticated AI systems that can better understand and assist humans in their daily tasks.

The research team plans to publish their findings in a leading scientific journal next month, and they are already working on practical applications of their technology.""",
            "is_binary": False
        },
        {
            "content_type": "audio",
            "filename": "test-article.mp3",
            "content": b"TEST_AUDIO_CONTENT",  # In real usage, this would be the actual MP3 content
            "is_binary": True
        }
    ]
    
    # Run all test cases
    results = []
    for test_case in test_cases:
        print(f"\nTesting {test_case['content_type']} upload...")
        success, url = upload_content_to_website(
            date_str=yesterday,
            content_type=test_case['content_type'],
            filename=test_case['filename'],
            content=test_case['content'],
            is_binary=test_case['is_binary']
        )
        results.append(success)
        print(f"{test_case['content_type']} upload {'succeeded' if success else 'failed'}")
        if success and url:
            print(f"URL: {url}")
    
    # Overall test result
    all_succeeded = all(results)
    print(f"\nOverall API test {'succeeded' if all_succeeded else 'failed'}.")
    print(f"Successful uploads: {sum(results)}/{len(results)}")
    return all_succeeded

if __name__ == "__main__":
    print("Starting API test...")
    test_api_functionality()
    print("API test completed.") 