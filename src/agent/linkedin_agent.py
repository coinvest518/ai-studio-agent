"""LinkedIn Post Conversion Sub-Agent.

Converts tweet text into professional LinkedIn post format.
"""

import logging
from langchain_google_genai import GoogleGenerativeAI
from langsmith import traceable
import os

logger = logging.getLogger(__name__)


@traceable(name="convert_to_linkedin_post")
def convert_to_linkedin_post(tweet_text: str) -> str:
    """Convert tweet text to LinkedIn post format.
    
    Args:
        tweet_text: Original tweet text with hashtags
        
    Returns:
        Professional LinkedIn post text
    """
    logger.info("Converting tweet to LinkedIn post format")
    
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_AI_API_KEY")
    )
    
    prompt = f"""You are the Marketing Intelligence AI for FWDA AI Automation Agency.

Short Post: {tweet_text}

Create a LONG-FORM LINKEDIN POST (300-350 words) that expands on this message.

BRAND: FWDA builds custom AI automation workflows for SMBs (coaches, agencies, consultants, trades, wellness, beauty, fitness, local businesses).

REQUIREMENTS:
- Tone: Thought-leadership, future-focused, plain English
- Structure:
  * Hook with trend/statistic
  * Problem SMBs face
  * How FWDA solves it (AI Agents, Workflow Automation, System Integration)
  * Tangible benefits (time, leads, costs, capacity)
  * Direct CTA
- Include 3 emojis total
- Include 5-8 hashtags: #AIAutomation #SmallBusiness #BusinessGrowth #Productivity #AIAgents #WorkflowAutomation #DigitalTransformation #ServiceBusiness
- Include 2 links:
  * https://fwda.site
  * https://cal.com/bookme-daniel/ai-consultation-smb
- No generic clichés
- Benefits must be measurable
- Active voice
- Clarity over complexity

Return ONLY the LinkedIn post text. No explanations."""

    try:
        response = llm.invoke(prompt)
        linkedin_post = response.strip()
        
        # Remove triple quotes and dashes if present
        linkedin_post = linkedin_post.replace('"""', '').replace("'''", "").replace('---', '').strip()
        
        logger.info("LinkedIn post created: %d characters", len(linkedin_post))
        return linkedin_post
        
    except Exception as e:
        logger.exception("Failed to convert to LinkedIn post: %s", e)
        # Fallback: return original tweet
        return tweet_text
