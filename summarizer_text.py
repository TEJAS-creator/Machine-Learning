from transformers import pipeline
summarizer = pipeline("summarization")
article = """
Text here
"""

summary = summarizer(article,max_length=150,min_length=50, do_sample=False)
summary[0]["summary_text"]


# 🔹 Text Tasks
# Sentiment Analysis → Classify text as positive, negative, neutral.
# Text Generation → Generate new text (autocomplete, stories, etc.).
# Translation → Translate text from one language to another.

# Question Answering → Extract answers from context passages.
# Text Classification → Categorize text into predefined classes.
# Zero-Shot Classification → Classify text into categories without explicit training.
# Summarization → Shorten text while keeping main meaning.
# Conversational → Maintain back-and-forth chatbot-like dialogue.

# 🔹 Token-Level Tasks
# Named Entity Recognition (NER) → Identify entities like names, places, dates.
# Part-of-Speech Tagging → Label words as noun, verb, adjective, etc.
# Fill-Mask → Predict missing words in a sentence.

# 🔹 Vision (Image) Tasks
# Image Classification → Identify what an image contains.
# Object Detection → Detect and locate objects in an image.
# Image Segmentation → Divide image into regions/objects.
# Image-to-Text (Captioning) → Generate captions for images.

# 🔹 Audio (Speech) Tasks
# Automatic Speech Recognition (ASR) → Convert speech to text.
# Text-to-Speech → Convert text into natural-sounding voice.
# Audio Classification → Identify sounds (e.g., music, barking, sirens).

# 🔹 Multimodal Tasks
# Document Question Answering → Answer questions about text/images in documents.
# Visual Question Answering (VQA) → Answer questions about an image
