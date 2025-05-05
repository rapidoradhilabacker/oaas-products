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

    async def extract_product_info(self, image_bytes: bytes) -> dict:
        with tracer.start_as_current_span('extract_product_info') as span:
            json_template = {
                "product_name": "",
                "product_code": "",
                "short_description": "",
                "long_description": ""
            }

            instruction = f"""
            You are an expert at extracting product information from images. Carefully analyze the provided product image and extract the following details into JSON format.

            Return ONLY the completed JSON with these fields:
            {json.dumps(json_template, indent=2)}

            Extraction guidelines:

            1. Product Name:
            - Look for the primary product name/title displayed prominently
            - Typically found at the top of packaging or near the brand logo
            - May include model names or variants (e.g., "Pro", "Max")

            2. Product Code:
            - Search for alphanumeric codes, SKUs, or model numbers
            - Common labels: "Item #", "Model", "SKU", "Product Code"
            - Often found near barcodes or in technical specifications

            3. Short Description:
            - Extract concise product highlights (1-2 sentences)
            - Look for bullet points or brief feature summaries
            - Typically emphasizes key selling points

            4. Long Description:
            - Find detailed technical specifications or full product narratives
            - May be in paragraphs or section labeled "Description"
            - Includes materials, dimensions, capabilities, usage instructions

            Important instructions:
            - Prioritize text clarity over decorative elements
            - Convert measurements to standard units when possible
            - Preserve technical terminology exactly as written
            - If any field isn't visible, leave it as empty string
            - Return ONLY valid JSON without additional text
            - Never invent information - only extract visible data
            - Maintain original language terms from the product
            """

            image_format = self.detect_image_format(image_bytes)
            mime_type = f'image/{image_format}'
            b64_image = base64.b64encode(image_bytes).decode("utf-8")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            content = response.choices[0].message.content.strip()

            try:
                extracted_data = json.loads(content)
            except json.JSONDecodeError:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    extracted_data = json.loads(content[start_idx:end_idx])
                else:
                    extracted_data = json_template

            for key in json_template.keys():
                if key not in extracted_data:
                    extracted_data[key] = ""

            extracted_data['file_type'] = image_format
            return extracted_data