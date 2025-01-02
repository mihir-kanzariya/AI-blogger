import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer
MODEL_NAME = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16  # Use mixed precision
).to("mps")  # Move model to Metal Performance Shaders

st.title("Humanize Blog Content")
st.write("Rewrite AI-generated blog content (up to 1000 words) to make it sound more natural and human-like.")

# Function to process a batch of paragraphs
def process_batch(paragraphs):
    if not paragraphs:
        return []
    
    # Prepare input prompts
    prompts = [f"Rewrite this paragraph to make it sound natural and human-like:\n\n{p}" for p in paragraphs]
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to("mps")

    # Generate outputs in batch
    outputs = model.generate(
        input_ids,
        max_length=512,
        min_length=100,
        num_beams=5,
        early_stopping=True
    )

    # Decode the outputs
    humanized_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Add fallback logic
    for i, (original, humanized) in enumerate(zip(paragraphs, humanized_texts)):
        if humanized.strip() == original.strip() or not humanized.strip():
            humanized_texts[i] = f"[Fallback] Could not humanize this paragraph:\n{original}"

    return humanized_texts

# Input text area
input_text = st.text_area("Paste your blog content here (up to 1000 words):", height=300)

if st.button("Humanize Blog"):
    if input_text.strip():
        with st.spinner("Processing your blog..."):
            try:
                # Split input into paragraphs
                paragraphs = input_text.split("\n\n")

                # Batch size for processing
                batch_size = 4  # Adjust based on available memory
                humanized_paragraphs = []

                # Process paragraphs in batches
                for i in range(0, len(paragraphs), batch_size):
                    batch = paragraphs[i:i + batch_size]
                    humanized_paragraphs.extend(process_batch(batch))

                # Combine humanized paragraphs into the final blog
                humanized_blog = "\n\n".join(humanized_paragraphs)

                # Display the result
                st.subheader("Humanized Blog Content")
                st.write(humanized_blog)
                st.download_button("Download Humanized Blog", humanized_blog, file_name="humanized_blog.txt")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter some content to humanize.")
