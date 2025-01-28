from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from .blogpostcreator import BlogPostCreator

from dotenv import load_dotenv
import os
import re


load_dotenv()

app = FastAPI(title="Blog Post Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ai-blogger-lgbr.onrender.com", "https://www.writemycontex.com"],  # Frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (e.g., POST, GET, DELETE)
    allow_headers=["*"],  # Allow all headers (e.g., Authorization, Content-Type)
)

class BlogRequest(BaseModel):
    keyword: str
    userprompt: str = None
    web_references: int


@app.post("/generate-blog-post/")
async def generate_blog_post(data: BlogRequest):
    """
    Endpoint to generate a blog post.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    wp_url = os.getenv("WP_URL")
    wp_user = os.getenv("WP_USER")
    wp_pass = os.getenv("WP_PASS")

    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured.")
    
    if not wp_url or not wp_user or not wp_pass:
        raise HTTPException(status_code=400, detail="WordPress credentials not configured.")

    try:
        # Initialize BlogPostCreator
        creator = BlogPostCreator(
            keyword=data.keyword,
            number_of_web_references=data.web_references,
            wp_url=wp_url,
            wp_user=wp_user,
            wp_pass=wp_pass,
            userprompt=data.userprompt,
            api_key=api_key
        )

        # Generate blog post content
        links = creator.get_links()
        response = creator.create_blog_post()
        print("🚀 ~ response:", response)

        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate blog post.")
        
        # Save generated content as a file
        title = re.sub(r"#\s*", "",  response.splitlines()[0].strip('"'))

        # creator.save_file(response, f"{title}.md")

        return {
            "message": "Blog post generated successfully!",
            "title": title,
            "content": response,
            "links": links,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/publish-to-wordpress/")
async def publish_to_wordpress(title: str = Form(...), content: str = Form(...)):
    """
    Endpoint to publish content to WordPress.
    """
    wp_url = os.getenv("WP_URL")
    wp_user = os.getenv("WP_USER")
    wp_pass = os.getenv("WP_PASS")

    if not wp_url or not wp_user or not wp_pass:
        raise HTTPException(status_code=400, detail="WordPress credentials not configured.")

    try:
        # Post to WordPress
        creator = BlogPostCreator(keyword="", number_of_web_references=0, wp_url=wp_url, wp_user=wp_user, wp_pass=wp_pass, userprompt="", api_key="")
        creator.postwordpress(content=content, title=title)

        return {"message": "Content published to WordPress successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

