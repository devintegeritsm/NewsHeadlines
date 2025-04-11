import os
from datetime import datetime, timedelta
from supabase import create_client, Client
import base64
from dotenv import load_dotenv
from typing import Union, Tuple, Optional

# Load environment variables from .env file
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Initialize Supabase clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def ensure_storage_bucket():
    """Ensure the content storage bucket exists."""
    try:
        # First, try to list all buckets to check if 'content' exists
        buckets = supabase_admin.storage.list_buckets()
        content_bucket_exists = any(bucket.name == "content" for bucket in buckets)
        
        if not content_bucket_exists:
            print("Creating storage bucket 'content'...")
            # Create the bucket using admin client
            supabase_admin.storage.create_bucket(
                id="content",
                name="content",
                options={
                    "public": True,
                    "file_size_limit": 52428800,  # 50MB
                    "allowed_mime_types": ["text/*", "audio/*"]
                }
            )
            print("Storage bucket 'content' created successfully.")
            return True
        else:
            print("Storage bucket 'content' exists.")
            return True
            
    except Exception as e:
        print(f"Error managing storage bucket: {e}")
        # Try to create the bucket directly if listing fails
        try:
            print("Attempting to create bucket directly...")
            supabase_admin.storage.create_bucket(
                id="content",
                name="content",
                options={
                    "public": True,
                    "file_size_limit": 52428800,
                    "allowed_mime_types": ["text/*", "audio/*"]
                }
            )
            print("Storage bucket 'content' created successfully.")
            return True
        except Exception as create_error:
            print(f"Failed to create bucket: {create_error}")
            return False

def ensure_content_table():
    """Ensure the content table exists with the correct schema."""
    try:
        # Check if the table exists by trying to select from it
        supabase_admin.table("content").select("*").limit(1).execute()
        print("Content table exists.")
    except Exception as e:
        if "relation \"content\" does not exist" in str(e):
            print("Creating content table...")
            # Create the table using raw SQL
            supabase_admin.rpc('create_content_table').execute()
            print("Content table created successfully.")
        else:
            raise e

def upload_content_to_supabase(date_str: str, content_type: str, filename: str, content: Union[str, bytes], is_binary: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Upload content to Supabase storage and create a record in the content table.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        content_type: Type of content (article, headlines, audio)
        filename: Name of the file
        content: File content (binary for audio, text for HTML/markdown)
        is_binary: Whether the content is binary (for audio files)
    
    Returns:
        Success status and URL if successful
    """

    try:
        # Ensure the storage bucket and table exist
        ensure_storage_bucket()
        ensure_content_table()
        
        # Prepare the storage path
        storage_path = f"{date_str}/{content_type}/{filename}"
        
        # Handle binary content (audio files)
        if is_binary:
            # For binary files, we need to encode as base64
            if isinstance(content, str):
                content = content.encode('utf-8')
            content_encoded = base64.b64encode(content).decode('utf-8')
        else:
            # For text files, ensure it's a string
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            content_encoded = content
        
        # Create a temporary file with the content
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            if is_binary:
                temp_file.write(base64.b64decode(content_encoded))
            else:
                temp_file.write(content_encoded.encode('utf-8'))
            temp_file_path = temp_file.name

        upload_content_type = "text/plain"
        if filename.endswith(".html"):
            upload_content_type = "text/html"
        elif filename.endswith(".md"):
            upload_content_type = "text/markdown"
        elif filename.endswith(".mp3"):
            upload_content_type = "audio/mpeg"
        elif filename.endswith(".wav"):
            upload_content_type = "audio/wav"
        
        try:
            # Upload to Supabase Storage using admin client
            with open(temp_file_path, 'rb') as file:
                storage_response = supabase_admin.storage.from_("content").upload(
                    path=storage_path,
                    file=file,
                    file_options={"x-upsert": "true", "content-type": upload_content_type}
                )
            
            # Get the public URL using admin client
            public_url = supabase_admin.storage.from_("content").get_public_url(storage_path)
            
            # Create a record in the content table using the admin client
            content_record = {
                "date": date_str,
                "content_type": content_type,
                "filename": filename,
                "storage_path": storage_path,
                "public_url": public_url,
                "metadata": {
                    "source": "newsapp",
                    "generated_at": datetime.now().isoformat()
                }
            }
            
            # Insert into content table using admin client
            supabase_admin.table("content").insert(content_record).execute()
            
            return True, public_url
            
        finally:
            # Clean up the temporary file
            import os
            os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"Error uploading to Supabase: {e}")
        return False, None

def test_supabase_functionality():
    """Test the Supabase functionality with realistic content from yesterday."""
    print("Testing Supabase functionality...")
    
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
        success, url = upload_content_to_supabase(
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
    print(f"\nOverall Supabase test {'succeeded' if all_succeeded else 'failed'}.")
    print(f"Successful uploads: {sum(results)}/{len(results)}")
    return all_succeeded

def upload_specific_audio_file():
    """Upload a specific MP3 file to Supabase for testing."""
    try:
        # Get yesterday's date
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Uploading audio file for date: {yesterday}")
        
        # Path to the MP3 file
        mp3_path = "2025-04-07-article/DOJ_Lawyer_Suspended_After_Questioning_Controversial_Deportation.mp3"
        
        # Read the MP3 file
        with open(mp3_path, 'rb') as file:
            mp3_content = file.read()
        
        # Upload to Supabase
        success, url = upload_content_to_supabase(
            date_str="2025-04-07",  # Using the date from the file path
            content_type="audio",
            filename="DOJ_Lawyer_Suspended_After_Questioning_Controversial_Deportation.mp3",
            content=mp3_content,
            is_binary=True
        )
        
        if success:
            print(f"Successfully uploaded MP3 file. URL: {url}")
        else:
            print("Failed to upload MP3 file")
        
        return success, url
        
    except Exception as e:
        print(f"Error uploading MP3 file: {e}")
        return False, None
    
def list_files_in_supabase():
    """List all files in the content bucket on Supabase."""
    try:
        # Get all files from the content bucket
        files = supabase_admin.storage.from_("content").list()
        # files = supabase.storage.from_("content").list()
        print(f"Found {len(files)} files on Supabase")
        for file in files:
            print(file)
    except Exception as e:
        print(f"Error listing files on Supabase: {str(e)}")

def get_files_from_supabase(path: str):
    """Get all files from a specific path in the content bucket on Supabase."""
    try:
        files = supabase_admin.storage.from_("content").list(path)
        print(f"Found {len(files)} files in {path} on Supabase")
        return files
    except Exception as e:
        print(f"Error listing article files on Supabase: {str(e)}")
        return []
    
def get_headlines_from_supabase():
    """Get all headlines from the content table on Supabase."""
    try:
        test_query = supabase.table("content") \
                            .select("count") \
                            .eq("content_type", "headlines") \
                            .execute()
        print(f"Test query: {test_query}") 

        headlines = supabase.table("content") \
                             .select("*") \
                             .eq("content_type", "headlines") \
                             .execute()
        
        print(f"Found {headlines} on Supabase")
        return headlines
    except Exception as e:
        print(f"Error getting headlines from Supabase: {str(e)}")
        return []
    
def delete_headlines_from_supabase(date_str: str):
    """Delete all headlines from the content table on Supabase for a specific date."""
    try:
        supabase_admin.table("content") \
            .delete() \
            .eq("content_type", "headlines") \
            .eq("date", date_str) \
            .execute()
        print(f"Headlines have been deleted for date: {date_str}")
    except Exception as e:
        print(f"Error deleting headlines from Supabase: {str(e)}")
    
    
if __name__ == "__main__":
    print("Starting Supabase test...")
    # test_supabase_functionality()
    # upload_specific_audio_file()
    # list_files_in_supabase()
    # get_headlines_from_supabase()
    delete_headlines_from_supabase("2025-04-09")
    print("Supabase test completed.") 