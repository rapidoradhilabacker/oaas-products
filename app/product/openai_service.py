import json
import base64
from typing import Dict
from app.config import OPEN_AI_SETTINGS
from openai import OpenAI
from app.tracing import tracer
import imghdr

class OpenAIService():
    def __init__(self):
        self.client = OpenAI(api_key=OPEN_AI_SETTINGS.api_key)

    def detect_image_format(self, img_bytes: bytes) -> str:
        # First try imghdr detection
        format = imghdr.what(None, img_bytes)
        if format in ['jpeg', 'png', 'gif', 'webp']:
            return format
        
        # Manual format detection as fallback
        if len(img_bytes) >= 2:
            if img_bytes.startswith(b'\xff\xd8'):
                return 'jpeg'
            if img_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'png'
            if img_bytes.startswith((b'GIF87a', b'GIF89a')):
                return 'gif'
            if len(img_bytes) >= 12 and img_bytes.startswith(b'RIFF') and img_bytes[8:12] == b'WEBP':
                return 'webp'
        
        raise ValueError("Unsupported image format")

    async def extract_product_info(self, image_bytes: list[bytes]) -> dict:
        with tracer.start_as_current_span('extract_product_info') as span:
            json_template = {
                "product_name": "",
                "short_description": "",
                "long_description": ""
            }

            # Construct the instruction string
            instruction = f"""
            You are an expert at analyzing product images and extracting relevant information. 
            Carefully examine the provided product images (multiple views/angles) and extract 
            the following details into JSON format.

            Return ONLY the completed JSON with these fields:
            {json.dumps(json_template, indent=2)}

            Extraction guidelines:

            1.  **Product Name:**
                * Identify the primary product name from any visible packaging or labels across 
                all images. Combine information from multiple views if necessary.
                * If the name differs between images, choose the most complete/accurate version.

            2.  **Short Description:**
                * Provide a concise, **single-sentence (1 sentence)** summary describing the product's main function or primary benefit.
                * If a suitable descriptive phrase or tagline is visible on the packaging, use or adapt it.
                * If no description is visible, formulate a standard 1-sentence description based on the identified product type and visual context (e.g., "A cleaning spray for household surfaces.", "Moisturizing body lotion for daily use.").

            3.  **Long Description:**
                * Provide a more detailed description, consisting of exactly **three sentences (3 sentences)**, covering potential features, benefits, usage context, or key ingredients/materials.
                * Prioritize using details visible on the packaging (like feature lists, ingredient highlights, usage directions) to construct the sentences.
                * If insufficient detail is visible, supplement or formulate the 3 sentences by inferring common attributes, typical uses, or standard benefits associated with this type of product. Ensure the description is plausible and relevant to the product shown (e.g., "This olive oil is cold-pressed for maximum flavor. It's ideal for salads, dipping, and finishing dishes. Rich in antioxidants, it supports a healthy diet.").

            Important instructions:
            * Analyze the image carefully. For Product Name, Short Description, and Long Description, use visible text first. If essential information for these fields is missing, infer reasonably based on the product type identified in the image.
            * For Product Code, strictly extract only visible text. Do not infer.
            * Aim for accuracy and relevance in both extracted and inferred information.
            * Convert measurements to standard units if easily possible, otherwise keep original units.
            * Preserve specific technical terms, ingredient names, or brand language if visible.
            * If a field (other than Product Code) cannot be reasonably determined or inferred, leave it as an empty string `""`.
            * Return ONLY the valid JSON output without any additional text before or after the JSON structure.
            * Maintain the original language from the product if applicable, especially for names and specific terms.
            """


            # Prepare message content with all images
            content_parts = [{"type": "text", "text": instruction}]
            
            # Add all images to the request
            for img_bytes in image_bytes:
                image_format = self.detect_image_format(img_bytes)
                b64_image = base64.b64encode(img_bytes).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{b64_image}",
                        "detail": "high"
                    }
                })

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": content_parts
                }],
                max_tokens=1000
            )

            # Existing JSON parsing logic (unchanged)
            content = response.choices[0].message.content.strip()
            try:
                extracted_data = json.loads(content)
            except json.JSONDecodeError:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                extracted_data = json.loads(content[start_idx:end_idx]) if start_idx >=0 else json_template

            # Ensure template compliance
            return {
                **json_template,
                **{k: v for k, v in extracted_data.items() if k in json_template}
            }