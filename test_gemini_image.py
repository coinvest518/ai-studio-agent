"""
Test file to verify Gemini image generation works independently.
This will help isolate and test the image generation functionality.
"""
import os
import logging
from composio import Composio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gemini_image_generation():
    """Test Gemini image generation in isolation."""
    
    # Initialize Composio client
    composio_client = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))
    
    # Test prompt
    test_prompt = "A modern, professional business office with AI technology, clean and futuristic design, suitable for social media marketing"
    
    logger.info("Testing Gemini image generation...")
    logger.info(f"Test prompt: {test_prompt}")
    logger.info(f"Composio API Key: {'*' * 20 + os.getenv('COMPOSIO_API_KEY', '')[-8:]}")
    logger.info(f"Google AI API Key: {'*' * 20 + os.getenv('GOOGLE_AI_API_KEY', '')[-8:]}")
    
    try:
        # Test Gemini image generation
        image_response = composio_client.tools.execute(
            "GEMINI_GENERATE_IMAGE",
            {
                "prompt": test_prompt,
                "model": "gemini-2.5-flash-image-preview", 
                "temperature": 0.7,
                "system_instruction": "Generate a professional, modern business image with clean, futuristic styling."
            }
        )
        
        logger.info("=" * 50)
        logger.info("GEMINI RESPONSE:")
        logger.info(f"Full response: {image_response}")
        logger.info("=" * 50)
        
        if image_response.get("successful", False):
            logger.info("✅ Image generation successful!")
            
            # Extract image data
            image_data = image_response.get("data", {})
            logger.info(f"Image data keys: {list(image_data.keys())}")
            
            # Check for content array (MCP format)
            content_array = image_data.get("content", [])
            logger.info(f"Content array length: {len(content_array)}")
            
            image_url = None
            
            # Look for image in content array
            for i, content_block in enumerate(content_array):
                logger.info(f"Content block {i}: {content_block}")
                if isinstance(content_block, dict):
                    if content_block.get("type") == "image":
                        if "source" in content_block and "media_type" in content_block["source"]:
                            # Direct S3 URL in source
                            source_data = content_block["source"].get("data", "")
                            if source_data.startswith("https://"):
                                image_url = source_data
                                logger.info(f"✅ Found S3 URL in source: {image_url}")
                                break
                    elif content_block.get("type") == "text":
                        # Check if text content contains S3 URL
                        text_content = content_block.get("text", "")
                        if text_content.startswith("https://") and "r2.dev" in text_content:
                            image_url = text_content
                            logger.info(f"✅ Found S3 URL in text: {image_url}")
                            break
                    elif "image_url" in content_block:
                        # Alternative format
                        image_url = content_block["image_url"]
                        logger.info(f"✅ Found image_url: {image_url}")
                        break
            
            if image_url:
                logger.info(f"🎉 SUCCESS! Generated image URL: {image_url}")
                return image_url
            else:
                logger.error("❌ No image URL found in response")
                return None
        else:
            error_msg = image_response.get("error", "Unknown error")
            logger.error(f"❌ Image generation failed: {error_msg}")
            return None
            
    except Exception as e:
        logger.exception(f"❌ Exception during image generation: {e}")
        return None

def main():
    """Main test function."""
    logger.info("🧪 Starting Gemini Image Generation Test")
    logger.info("-" * 50)
    
    # Check environment variables
    required_vars = ["COMPOSIO_API_KEY", "GOOGLE_AI_API_KEY", "COMPOSIO_TOOLKIT_VERSION_GEMINI"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            logger.info(f"✅ {var}: {'*' * 10 + value[-4:]}")
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    # Test image generation
    result = test_gemini_image_generation()
    
    if result:
        logger.info("🎉 TEST PASSED: Image generation working!")
        return True
    else:
        logger.error("❌ TEST FAILED: Image generation not working")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)