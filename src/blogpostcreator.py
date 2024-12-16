import os
import re
import bs4

from googlesearch import search  # Import for Google Search
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

import os
import requests

class BlogPostCreator:
    def __init__(self, keyword, number_of_web_references, wp_url, wp_user, wp_pass, userprompt):
        self.keyword = keyword
        self.number_of_web_references = number_of_web_references
        self.wp_url = wp_url
        self.wp_user = wp_user
        self.wp_pass = wp_pass
        self.userprompt = userprompt

    def parse_links(self, search_results):
        print("-----------------------------------")
        print("Parsing links ...")
        # No need for regex parsing here since `googlesearch-python` directly provides links
        return search_results

    def save_file(self, content: str, filename: str):
        print("-----------------------------------")
        print("Saving file in blogs ...")
        directory = "blogs"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f" 🥳 File saved as {filepath}")
    
    def postwordpress(self, content: str, title: str, category_id=1):  # Default category_id 1 (Uncategorized)
        print("-----------------------------------")
        print("Posting content to WordPress ...")
        
        # Construct the endpoint for WordPress REST API
        url = f"{self.wp_url}/wp-json/wp/v2/posts"
        
        # Prepare the payload for the POST request
        payload = {
            'title': title,
            'content': content,
            'status': 'draft',  # To directly publish the post
            'categories': [category_id]  # Pass the category ID instead of name
        }

        # Authentication using Application Password
        auth = (self.wp_user, self.wp_pass)  # Use the application password here
        
        # Make the POST request to create the post
        response = requests.post(url, json=payload, auth=auth)
        
        if response.status_code == 201:
            print(f"✅ Post successfully created: {response.json()['link']}")
        else:
            print(f"❌ Failed to create post. Error: {response.status_code} - {response.text}")



    def get_links(self):
        try:
            print("-----------------------------------")
            print("Getting links using Google...")

            # Convert the generator to a list
            results = list(search(self.keyword, num_results=self.number_of_web_references))

            # Parse links (if necessary, e.g., further processing)
            links = self.parse_links(results)
            print("Retrieved links:")
            for idx, link in enumerate(links, start=1):
                print(f"{idx}. {link}")

            return links

        except Exception as e:
            print(f"An error occurred while getting links: {e}")

    def create_blog_post(self):
            try:
                print("-----------------------------------")
                print("Creating blog post ...")
                wp_url="https://wp-admin.pdfgpt.io/"
                wp_user="utsav.prajapati@bacancy.com"
                wp_pass="i5llP@XHuwG&EIIjuZuTCI2f"  # Add your WordPress credentials her

                # Define self and docs variables
                self = BlogPostCreator(keyword=self.keyword, number_of_web_references=self.number_of_web_references,  wp_url=wp_url, wp_user=wp_user, wp_pass=wp_pass, userprompt=self.userprompt)
                docs = []

                # Define splitter variable
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=400,
                    add_start_index=True,
                )

                # Load documents
                bs4_strainer = bs4.SoupStrainer(('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'))

                document_loader = WebBaseLoader(
                        web_path=self.get_links()  # `get_links` now returns a list of strings

                )

                docs = document_loader.load()

                # Split documents
                splits = splitter.split_documents(docs)

                # step 3: Indexing and vector storage
                vector_store = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

                # step 4: retrieval
                retriever = vector_store.as_retriever(search_type="similarity", search_kwards={"k": 10})

                # step 5 : Generation
                llm = ChatOpenAI(model="gpt-4o-mini") 

                template = """
                    Given the following information, generate a blog post                   
                        
                    
                    Instructions:
                    The blog should be properly and beautifully formatted using markdown.
                    The blog title should be SEO optimized.
                    The blog title, should be crafted with the keyword in mind and should be catchy and engaging. But not overly expressive.
                    Generate a title that is concise and direct. Avoid using introductory phrases like 'Exploring' or 'Discover'. For example:

                    Incorrect: 'Exploring Gulu: 10 Best Things to Do in Gulu'

                    Correct: '10 Best Things to Do in Gulu'

                    Incorrect: 'Who is Jordan Mungujakisa: Exploring the Mind of a Mobile App Alchemist'

                    Correct: 'The story of Jordan Mungujakisa'

                    Please provide titles in the correct format.

                    Do not include : in the title.

                    

                    Each sub-section should have at least 3 paragraphs.

                    

                    Each section should have at least three subsections.

                    

                    Sub-section headings should be clearly marked.

                    

                    Clearly indicate the title, headings, and sub-headings using markdown.

                    Each section should cover the specific aspects as outlined.

                    For each section, generate detailed content that aligns with the provided subtopics. Ensure that the content is informative and covers the key points.

                    Ensure that the content is consistent with the title and subtopics. Do not mention an entity in the title and not write about it in the content.

                    Ensure that the content flows logically from one section to another, maintaining coherence and readability.

                    Where applicable, include examples, case studies, or insights that can provide a deeper understanding of the topic.

                    Always include discussions on ethical considerations, especially in sections dealing with data privacy, bias, and responsible use. Only add this where it is applicable.

                    In the final section, provide a forward-looking perspective on the topic and a conclusion.

                    

                    Please ensure proper and standard markdown formatting always.

                    

                    Make the blog post sound as human and as engaging as possible, add real world examples and make it as informative as possible.

                    

                    You are a professional blog post writer and SEO expert.

                    Each blog post should have atleast 5 sections with 3 sub-sections each.

                    Each sub section should have atleast 3 paragraphs.

                    Follow google content qualities rules, Most importantly follow  E-E-A-T rule
                    

                    You are a professional content writer creating blog posts for an AI product. Your goal is to craft engaging, informative, and credible content while adhering to the principles of E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness). Follow these specific instructions for each principle:

                    Experience:
                    - Include personal stories, case studies, or real-world examples showcasing how the AI product has been used effectively.
                    - Use a conversational tone to describe firsthand experiences.
                    - Highlight practical challenges and benefits encountered during the usage of the AI product.

                    Examples to include:
                    1. A detailed account of how someone used the AI product to improve their daily tasks.
                    2. A narrative about solving a real problem using the AI product.
                    3. Insights from testing the AI tool in various scenarios.

                    ---

                    Expertise:
                    - Explain technical concepts and features of the AI product with clarity and depth, catering to readers with varying levels of understanding.
                    - Reference relevant research, industry trends, or advanced AI methodologies.
                    - Provide actionable tips or expert advice that establishes the content creator as knowledgeable.

                    Examples to include:
                    1. A breakdown of how a key feature of the AI product works.
                    2. Comparisons between the AI product and similar tools.
                    3. Tips for getting the most out of the product from a technical perspective.

                    ---

                    Authoritativeness:
                    - Cite credible sources like research papers, news articles, or reports to validate claims.
                    - Include endorsements or feedback from recognized experts in AI or related fields.
                    - Highlight partnerships, certifications, or awards related to the AI product.

                    Examples to include:
                    1. Mention of studies or statistics that support the AI product's claims.
                    2. Quotes or testimonials from AI experts.
                    3. Recognition or awards the product has received.

                    ---

                    Trustworthiness:
                    - Be transparent about how the AI product works, including its limitations and security measures.
                    - Use real user reviews or testimonials to establish credibility.
                    - Clearly explain data privacy, compliance, and ethical practices.

                    Examples to include:
                    1. A blog post about how the company ensures user data privacy.
                    2. Testimonials from real customers with verifiable success stories.
                    3. A post addressing common questions or misconceptions about the product.

                    ---

                    General Tone and Style:
                    - Be clear, concise, and engaging.
                    - Use a professional yet approachable tone.
                    - Avoid making exaggerated or unsupported claims.

                    Your task is to create high-quality, trustworthy, and user-friendly content while avoiding common pitfalls that lead to poor-quality or failed evaluations. Follow these instructions carefully:

                    ---

                    Key Objectives:
                    1. Ensure the content is accurate, original, and relevant to the audience.
                    2. Demonstrate Experience, Expertise, Authoritativeness, and Trustworthiness (E-E-A-T) principles.
                    3. Avoid any practices or content types that result in failed quality checks or negative user experiences.

                    ---

                    Avoid the Following Poor-Quality Practices:

                    1. Harmful or Misleading Content:
                    - Do not create content with deceptive claims, inaccurate facts, or potentially harmful advice.
                    - Avoid clickbait titles or statements that do not align with the actual content.

                    2. Lack of Transparency:
                    - Always include clear details about the purpose of the page and the credentials of the author or organization.
                    - Avoid anonymous or unverifiable claims. Include an "About Us" or "Contact" section where necessary.

                    3. Copied or Low-Effort Content:
                    - Avoid duplicating content from other sources without adding value or originality.
                    - Do not use auto-generated or poorly curated content that lacks human effort.

                    4. Distracting Ads or Poor Design:
                    - Ensure that the content is not overwhelmed by ads or monetization elements.
                    - Avoid intrusive pop-ups or design choices that degrade the user experience.

                    5. Irrelevant or Minimal Content:
                    - Avoid publishing pages with little to no actionable or useful information.
                    - Ensure the main content (MC) aligns with the page’s title and purpose.

                    6. Unverified or Unsupported Claims:
                    - Do not make exaggerated promises or unsupported statements (e.g., "Guaranteed 10x results overnight").
                    - Always include credible sources, citations, or references to back up claims.

                    ---

                    Best Practices to Follow:
                    1. Content Quality:
                    - Write detailed, informative, and well-researched posts that satisfy the user’s intent.
                    - Ensure originality, and add unique insights or perspectives to distinguish your content.

                    2. User-Centered Design:
                    - Make your page easy to navigate and free from unnecessary distractions.
                    - Include appropriate headings, visuals, and a clear call-to-action (CTA) where relevant.

                    3. Transparency:
                    - Provide clear information about the author or organization behind the content.
                    - Include credentials, contact information, or links to support credibility.

                    4. Regular Updates:
                    - Ensure that the content is timely and accurate. Regularly update older posts to maintain relevance.

                    ---


                    - Ensure Accuracy: Provide real examples and credible sources.
                    - Avoid Misleading Claims: Do not overpromise features or results of AI tools.
                    - Demonstrate Trustworthiness: Include testimonials or expert recommendations.


                    Use simple indian english, human tone, avoid starting by "In today's fast-paced world," and similar kind of sentences, use simple english words, Do not use any type of hook to start blog or any paragraph,
                    The content should be tailored for a [specific audience: e.g., students, professionals, researchers] and include the following elements based on the nature of the topic:

                    Engaging Hook: Start with a compelling hook or attention-grabbing statement to draw the reader in.

                    Context/Background Information: Provide brief background or context on the topic to set the stage for the discussion.

                    Clear Structure: Break the content into clear, logically structured sections (e.g., introduction, body, conclusion), using transitional phrases to connect ideas smoothly.

                    Tone:

                    If the topic is informative or academic, maintain a formal, neutral tone.
                    If the topic is motivational or self-improvement, use an inspirational, encouraging tone.
                    For persuasive content, adopt a convincing and authoritative tone.
                    For casual or conversational topics, use a friendly, approachable tone.
                    Examples and Anecdotes: Include relevant examples or short anecdotes that help clarify the topic and make it relatable to the audience.

                    Analogies and Metaphors: If appropriate, use analogies or metaphors to simplify complex ideas and make them easier to understand.

                    Problem-Solution Structure: If the topic involves challenges, outline the problem and propose practical solutions, offering actionable insights or recommendations.

                    Rhetorical Questions: Use rhetorical questions to engage the reader and provoke thought.

                    Data and Statistics: Include relevant data, facts, or research findings to support claims, making the content more credible.

                    Emotional Appeal: When appropriate, appeal to the reader's emotions, especially for topics related to personal growth, challenges, or motivation.
                    

                    Context: {context}

                    Please also folow below guidance to create blog post: {userprompt}

                    Blog Post: 
                
                """

                prompt = PromptTemplate.from_template(template=template)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

               # Set up the chain with correct context and handling for inputs
                chain = (
                    {"context": retriever | format_docs, "keyword": RunnablePassthrough(), "userprompt": RunnablePassthrough() or None}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                final_input = template.format(keyword=self.keyword, context=retriever, userprompt=self.userprompt)

                # You need to pass keyword and userprompt as arguments directly to the chain if required
                # inputs = {
                #     "keyword": self.keyword,  # Assuming 'keyword' is what is required for context
                #     "userprompt": self.userprompt  # 'userprompt' can be passed as an optional parameter
                # }

                # Call the chain's invoke method with the appropriate inputs
                return chain.invoke(final_input)

            except Exception as e:
                return e
            
