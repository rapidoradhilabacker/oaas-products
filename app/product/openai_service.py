import json
import base64
from typing import Dict, Optional
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
            if len(img_bytes) >= 12 and img_bytes.startswith(b'BM'):
                return 'bmp'
            # identify pdf 
            if len(img_bytes) >= 4 and img_bytes.startswith(b'%PDF'):
                return 'pdf'
        
        # If we can't detect the format, default to jpeg as a fallback
        # This is a common format and OpenAI API may still process it
        print("Warning: Could not detect image format, defaulting to jpeg")
        return 'jpeg'

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
                    "image_url": {  # type: ignore
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

    async def extract_combined_product_info(self, image_bytes: list[bytes], products_count: int, file_names: list[str]) -> list[dict]:
        with tracer.start_as_current_span('extract_combined_product_info') as span:
            json_template = {
                "product_code": "",
                "product_name": "",
                "short_description": "",
                "long_description": "",
                "file_type": "",
                "file_names": []
            }

            # Construct the instruction string
            instruction = f"""
            You are an expert at analyzing product images and extracting relevant information. 
            I'm providing you with {products_count} different products in a combined set of images.
            
            Your task is to identify each distinct product and extract the following details for EACH product:
            
            1. Product Code: If visible, extract the product code or SKU from the packaging.
               - If not visible, generate a plausible code based on the product name and characteristics.
               - Format should be: [COMPANY_PREFIX]-[CATEGORY]-[PRODUCT]-[NUMBER], e.g., "ADH-RED-APPLE-001"
            
            2. Product Name: Identify the primary product name.
            
            3. Short Description: Provide a concise, single-sentence summary describing the product's main function.
            
            4. Long Description: Provide a more detailed description in exactly three sentences covering features, benefits, and usage.
            
            5. For each product, identify which images belong to it. Multiple images may show the same product from different angles.
            
            6. File Names: Use these exact file names in your response:
               {json.dumps(file_names)}
               - Each product should include the appropriate file name(s) in its file_name array.
            
            Use the following JSON template for each product:
            {json.dumps(json_template, indent=4)}
            
            Return EXACTLY {products_count} products in a JSON array, with each product having all the fields from the template.
            
            Important instructions:
            * Ensure you identify and separate EXACTLY {products_count} distinct products.
            * For each product, include all relevant information and associate the correct images.
            * For each product, include the EXACT file_name(s) from the provided list in the file_name array.
            * If multiple images show the same product, group them together.
            * Return ONLY the valid JSON output without any additional text.
            """

            # Prepare message content with all images
            content_parts = [{"type": "text", "text": instruction}]
            
            # Add all images to the request
            for img_bytes in image_bytes:
                image_format = self.detect_image_format(img_bytes)
                b64_image = base64.b64encode(img_bytes).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {  # type: ignore
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
                max_tokens=2000
            )

            # Parse JSON response
            content = response.choices[0].message.content.strip()
            try:
                extracted_data = json.loads(content)
                if not isinstance(extracted_data, list):
                    extracted_data = [extracted_data]
                
                # Ensure we have exactly the requested number of products
                if len(extracted_data) != products_count:
                    print(f"Warning: Expected {products_count} products but got {len(extracted_data)}")
                
                # Ensure each product has all required fields
                result = []
                for i, product in enumerate(extracted_data):
                    product_data = {**json_template}
                    for key in json_template:
                        if key in product:
                            product_data[key] = product[key]
                    result.append(product_data)
                
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        extracted_data = json.loads(content[start_idx:end_idx])
                        if not isinstance(extracted_data, list):
                            extracted_data = [extracted_data]
                        
                        # Process each product
                        result = []
                        for product in extracted_data:
                            product_data = {**json_template}
                            for key in json_template:
                                if key in product:
                                    product_data[key] = product[key]
                            result.append(product_data)
                        
                        return result
                    except json.JSONDecodeError:
                        # If still failing, return template data
                        return [{**json_template} for _ in range(products_count)]
                else:
                    # Return template data if no JSON found
                    return [{**json_template} for _ in range(products_count)]

    async def extract_combined_product_info_from_invoice(self, company_name: Optional[str], image_bytes: list[bytes], file_names: list[str]) -> list[dict]:
        with tracer.start_as_current_span('extract_combined_product_info_from_invoice') as span:
            json_template = {
                "product_code": "",
                "product_name": "",
                "short_description": "",
                "long_description": "",
                "file_type": "",
                "file_names": [],
                "price": 0.0
            }

            # Construct the instruction string
            instruction = f"""
            You are an expert at analyzing invoice documents and extracting product information. 
            I'm providing you with an invoice that contains information about different products.
            
            Your task is to identify each distinct product from the invoice and extract the following details for EACH product:
            
            1. Product Code: Extract the product code, SKU, or item number from the invoice.
               - If not visible, generate a plausible code based on the product name.
               - Format should be: [COMPANY_PREFIX]-[PRODUCT]-[NUMBER], e.g., "ADH-LAXMI-BHOG-ATTA-05-KG" (For the product LAXMI BHOG ATTA 05 KG and company name ADHIL STORES)
               - Company prefix should be extracted from the invoice or use {company_name or ""} if not visible.
            
            2. Product Name: Extract the product name or description from the invoice line items.
            
            3. Short Description: Provide a concise, single-sentence summary describing the product based on information in the invoice.
            
            4. Long Description: Provide a more detailed description in exactly three sentences based on the information available in the invoice.
            
            5. Price: Extract the price for each product as a numeric value (e.g., 19.99).
               - This is critical information that must be extracted from the invoice.
               - Do not include currency symbols in the price value.
               - If multiple prices are shown (like unit price and total price), prefer the unit price.
            
            6. File Names: Use these exact file names in your response:
               {json.dumps(file_names)}
               - Each product should include the appropriate file name(s) in its file_name array.
            
            Use the following JSON template for each product:
            {json.dumps(json_template, indent=4)}
            
            Return each product having all the fields from the template.
            
            Important instructions:
            * Focus on extracting information from the invoice structure - look for line items, product tables, or itemized lists.
            * Pay special attention to extracting accurate price information for each product.
            * Ensure you identify and separate each distinct product.
            * For each product, include all relevant information from the invoice.
            * For each product, include the EXACT file_name(s) from the provided list in the file_name array.
            * Return ONLY the valid JSON output without any additional text.
            """

            # Prepare message content with all images
            content_parts = [{"type": "text", "text": instruction}]
            
            # Add all images to the request
            for img_bytes in image_bytes:
                image_format = self.detect_image_format(img_bytes)
                b64_image = base64.b64encode(img_bytes).decode("utf-8")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {  # type: ignore
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
                max_tokens=2000
            )

            # Parse JSON response
            content = response.choices[0].message.content.strip()
            try:
                extracted_data = json.loads(content)
                if not isinstance(extracted_data, list):
                    extracted_data = [extracted_data]
                                
                # Ensure each product has all required fields
                result = []
                for i, product in enumerate(extracted_data):
                    product_data = {**json_template}
                    for key in json_template:
                        if key in product:
                            product_data[key] = product[key]
                    result.append(product_data)
                
                return result
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    try:
                        extracted_data = json.loads(content[start_idx:end_idx])
                        if not isinstance(extracted_data, list):
                            extracted_data = [extracted_data]
                        
                        # Process each product
                        result = []
                        for product in extracted_data:
                            product_data = {**json_template}
                            for key in json_template:
                                if key in product:
                                    product_data[key] = product[key]
                            result.append(product_data)
                        
                        return result
                    except json.JSONDecodeError:
                        # If still failing, return template data
                        return [{**json_template} for _ in range(len(image_bytes))]
                else:
                    # Return template data if no JSON found
                    return [{**json_template} for _ in range(len(image_bytes))]