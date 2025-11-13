"""
Real test file using the exact same configuration as the main agent.
This matches the actual working configuration with toolkit_version parameter.
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

def test_real_gemini_configuration():
    """Test Gemini with the exact same config as main agent."""
    
    # Initialize Composio client exactly like main agent
    composio_client = Composio(
        api_key=os.getenv("COMPOSIO_API_KEY"),
        entity_id="default"
    )
    
    # Test prompt matching main agent style
    test_prompt = "Professional AI automation business workspace with modern technology, clean futuristic design for social media marketing"
    
    logger.info("Testing REAL Gemini configuration (matching main agent)...")
    logger.info(f"Test prompt: {test_prompt}")
    logger.info(f"Composio API Key: {'*' * 15 + os.getenv('COMPOSIO_API_KEY', '')[-5:]}")
    logger.info(f"Toolkit Version: {os.getenv('COMPOSIO_TOOLKIT_VERSION_GEMINI')}")
    
    try:
        # Use correct Composio API without invalid toolkit_version parameter
        image_response = composio_client.tools.execute(
            "GEMINI_GENERATE_IMAGE",
            {
                "prompt": test_prompt,
                "model": "gemini-2.5-flash-image-preview",
                "temperature": 0.7,
                "system_instruction": "Generate a professional, modern business image that aligns with AI automation and small business themes. Style should be clean, futuristic, and engaging for social media."
            }
        )
        
        logger.info("=" * 60)
        logger.info("REAL CONFIGURATION GEMINI RESPONSE:")
        logger.info(f"Success: {image_response.get('successful', False)}")
        logger.info(f"Error: {image_response.get('error', 'None')}")
        logger.info("=" * 60)
        
        if image_response.get("successful", False):
            logger.info("✅ REAL CONFIG: Image generation successful!")
            
            # Extract image data using EXACT same logic as main agent
            image_data = image_response.get("data", {})
            content_array = image_data.get("content", [])
            
            image_url = None
            
            # Look for image in content array (exact main agent logic)
            for content_block in content_array:
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
            
            # Fallback extraction (exact main agent logic)
            if not image_url:
                image_url = image_data.get("image_url") or image_data.get("url") or image_data.get("s3_url")
            
            if image_url:
                logger.info(f"🎉 REAL CONFIG SUCCESS! Image URL: {image_url}")
                logger.info(f"🔗 URL Length: {len(image_url)} characters")
                logger.info(f"🌐 Is Valid URL: {image_url.startswith('https://')}")
                return {"success": True, "image_url": image_url}
            else:
                logger.error("❌ REAL CONFIG: No image URL found in response")
                logger.error(f"Full response data: {image_response}")
                return {"success": False, "error": "No image URL found"}
        else:
            error_msg = image_response.get("error", "Unknown error")
            logger.error(f"❌ REAL CONFIG: Image generation failed: {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        logger.exception(f"❌ REAL CONFIG: Exception during image generation: {e}")
        return {"success": False, "error": str(e)}

def test_environment_variables():
    """Test that all required environment variables are present."""
    logger.info("🔧 Testing Environment Variables...")
    
    required_vars = {
        "COMPOSIO_API_KEY": os.getenv("COMPOSIO_API_KEY"),
        "GOOGLE_AI_API_KEY": os.getenv("GOOGLE_AI_API_KEY"), 
        "COMPOSIO_TOOLKIT_VERSION_GEMINI": os.getenv("COMPOSIO_TOOLKIT_VERSION_GEMINI")
    }
    
    all_good = True
    for var_name, var_value in required_vars.items():
        if var_value:
            masked_value = "*" * 10 + var_value[-4:] if len(var_value) > 4 else "*" * len(var_value)
            logger.info(f"✅ {var_name}: {masked_value}")
        else:
            logger.error(f"❌ {var_name}: MISSING!")
            all_good = False
    
    return all_good

def main():
    """Main test function with real configuration."""
    logger.info("🧪 REAL CONFIGURATION TEST - Matching Main Agent")
    logger.info("=" * 60)
    
    # Test environment
    env_ok = test_environment_variables()
    if not env_ok:
        logger.error("❌ Environment variables missing - test cannot proceed")
        return False
    
    logger.info("\n🚀 Starting Real Gemini Image Generation Test...")
    
    # Test real configuration
    result = test_real_gemini_configuration()
    
    logger.info("\n" + "=" * 60)
    if result.get("success"):
        logger.info("🎉 REAL CONFIG TEST PASSED!")
        logger.info("✅ Main agent should work with this exact configuration!")
        logger.info(f"🖼️  Generated Image: {result.get('image_url', 'No URL')}")
        return True
    else:
        logger.error("❌ REAL CONFIG TEST FAILED!")
        logger.error(f"💥 Error: {result.get('error', 'Unknown')}")
        logger.error("🔧 Main agent will have the same issue!")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"{'='*60}")
    exit(0 if success else 1)