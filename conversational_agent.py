from aci import ACI
import json, anthropic, re, requests
from dotenv import load_dotenv
from aci.types.functions import FunctionDefinitionFormat

load_dotenv()

class ConversationalAmazonAgent:
    def __init__(self, model="claude-3-5-sonnet-20240620", linked_account="echobuy"):
        self.client = anthropic.Anthropic()
        self.aci = ACI()
        self.model = model
        self.linked_account = linked_account
        self.tools = ["BRAVE_SEARCH__WEB_SEARCH", "FIRECRAWL__BATCH_SCRAPE"]
        self.tool_definitions = [
            self.aci.functions.get_definition(tool_name, format=FunctionDefinitionFormat.ANTHROPIC)
            for tool_name in self.tools
        ]
        self.current_products = {}
        self.shopping_session = {
            'products_viewed': [],
            'user_preferences': {},
            'budget_range': None,
            'comparison_mode': False,
            'purchase_intent': None,
            'conversation_context': []
        }
        self.reset()

    def reset(self):
        """Clears the conversation history to start fresh."""
        self.current_products = {}
        self.shopping_session = {
            'products_viewed': [],
            'user_preferences': {},
            'budget_range': None,
            'comparison_mode': False,
            'purchase_intent': None,
            'conversation_context': []
        }
        self.messages = [
            {
                "role": "user",
                "content": (
                    "You are a friendly, conversational personal shopping assistant having a VOICE conversation with a customer. "
                    "ENHANCED CAPABILITIES:\n"
                    "- Detect purchase intent: 'I want this', 'add to cart', 'buy this one', 'I'll take it'\n"
                    "- Offer comparisons: 'compare these', 'show differences', 'which is better'\n"
                    "- Extract preferences: budget, brand preferences, use cases, priorities\n"
                    "- Remember context: reference previous products and conversations\n"
                    "- Handle interruptions: if user changes topic, adapt smoothly\n\n"
                    "VOICE CONVERSATION RULES:\n"
                    "- Keep responses short, natural, and conversational (2-3 sentences max initially)\n"
                    "- Speak like a helpful friend, not a formal assistant\n"
                    "- Use casual language and contractions ('I'll', 'let's', 'that's')\n"
                    "- Ask ONE question at a time, not numbered lists\n"
                    "- Show enthusiasm and personality\n"
                    "- When you find products, describe them conversationally like you're showing them to a friend\n"
                    "- Keep the conversation flowing naturally\n"
                    "- NEVER read URLs out loud - instead say you'll display them or send them\n\n"
                    "PURCHASE INTENT HANDLING:\n"
                    "- When user shows purchase intent, respond: 'Perfect! I'll help you get that ordered. Let me show you the checkout options!'\n"
                    "- Then use format: [PURCHASE_INTENT: product_name | url | price]\n\n"
                    "COMPARISON MODE:\n"
                    "- When requested, use format: [COMPARE_PRODUCTS: product1_name|url|price|key_features vs product2_name|url|price|key_features]\n\n"
                    "LINK HANDLING:\n"
                    "- For single links: [DISPLAY_LINK: product_name | url]\n"
                    "- For product cards: [PRODUCT_CARD: name|url|price|rating|key_feature1|key_feature2|image_hint]\n\n"
                    "Your workflow:\n"
                    "1. Extract user preferences and budget from conversation\n"
                    "2. Use BRAVE_SEARCH__WEB_SEARCH to find Amazon product links\n"
                    "3. Use FIRECRAWL__BATCH_SCRAPE to get detailed product info\n"
                    "4. Present findings as rich product cards with comparisons\n"
                    "5. Detect purchase intent and guide to checkout\n"
                    "6. Store product info and session context\n\n"
                    "Always greet the user warmly when starting."
                )
            },
            {
                "role": "assistant",
                "content": "Hey there! I'm your personal shopping assistant, and I'm super excited to help you find exactly what you're looking for on Amazon today. What can I help you discover?"
            }
        ]

    def extract_links(self, result):
        """Extracts up to 5 unique Amazon product links from search results."""
        amazon_links = []
        data = result.get("data", {})
        web_results = data.get("web", {}).get("results", [])
        for item in web_results:
            url = item.get("url", "")
            if "amazon.com" in url and ("/dp/" in url or "/gp/product/" in url) and "/s?" not in url and "/b?" not in url:
                clean_url = url.split("?")[0].split("/ref=")[0]
                amazon_links.append(clean_url)
        return list(set(amazon_links))[:5]

    def handle_tool_call(self, tool_call):
        """Handles a tool call request from the model."""
        return self.aci.handle_function_call(
            tool_call.name,
            tool_call.input,
            linked_account_owner_id=self.linked_account,
            format=FunctionDefinitionFormat.ANTHROPIC,
        )

    def extract_product_info(self, scraped_data):
        """Extract detailed product info and enrich with metadata."""
        try:
            if isinstance(scraped_data, dict) and 'data' in scraped_data:
                results = scraped_data['data']
                if isinstance(results, list):
                    for i, result in enumerate(results):
                        if isinstance(result, dict):
                            url = result.get('url', '')
                            content = result.get('content', '')
                            
                            # Enhanced product info extraction
                            product_info = self._extract_enhanced_product_info(content, url)
                            if product_info:
                                product_key = f"product_{i+1}"
                                self.current_products[product_key] = product_info
                                
                                # Add to session context
                                self.shopping_session['products_viewed'].append(product_info)
                                
        except Exception as e:
            print(f"Error extracting product info: {e}")

    def _extract_enhanced_product_info(self, content, url):
        """Extract comprehensive product info from scraped content."""
        try:
            product_info = {
                'url': url,
                'name': 'Amazon Product',
                'price': 'Price not found',
                'rating': 'Rating not found',
                'key_features': [],
                'availability': 'Check Amazon',
                'prime_eligible': False
            }
            
            # Extract title
            if "Amazon.com:" in content:
                match = re.search(r'Amazon\.com:\s*([^|]+)', content)
                if match:
                    product_info['name'] = match.group(1).strip()
            
            # Extract price
            price_patterns = [
                r'\$(\d+\.?\d*)',
                r'Price:\s*\$(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*dollars?'
            ]
            for pattern in price_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    product_info['price'] = f"${match.group(1)}"
                    break
            
            # Extract rating
            rating_patterns = [
                r'(\d\.?\d*)\s*out of 5 stars',
                r'Rating:\s*(\d\.?\d*)',
                r'(\d\.?\d*)\s*stars?'
            ]
            for pattern in rating_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    product_info['rating'] = f"{match.group(1)}/5 stars"
                    break
            
            # Extract key features (simplified)
            if 'features' in content.lower() or 'specifications' in content.lower():
                # Look for bullet points or feature lists
                feature_matches = re.findall(r'[•\-\*]\s*([^•\-\*\n]{10,100})', content)
                product_info['key_features'] = feature_matches[:3]  # Top 3 features
            
            # Check Prime eligibility
            if 'prime' in content.lower() and ('eligible' in content.lower() or 'free' in content.lower()):
                product_info['prime_eligible'] = True
            
            return product_info
            
        except Exception as e:
            print(f"Error in enhanced extraction: {e}")
            return {
                'url': url,
                'name': 'Amazon Product',
                'price': 'See Amazon',
                'rating': 'N/A',
                'key_features': [],
                'availability': 'Check Amazon',
                'prime_eligible': False
            }

    def detect_user_intent(self, user_input):
        """Detect user's shopping intent and preferences."""
        user_input_lower = user_input.lower()
        
        # Purchase intent keywords
        purchase_keywords = ['buy', 'purchase', 'order', 'i want', 'i\'ll take', 'add to cart', 'checkout']
        comparison_keywords = ['compare', 'difference', 'which is better', 'vs', 'versus']
        budget_patterns = [r'\$(\d+)', r'under (\d+)', r'budget.*?(\d+)', r'spend.*?(\d+)']
        
        intent = {
            'purchase_intent': any(keyword in user_input_lower for keyword in purchase_keywords),
            'comparison_request': any(keyword in user_input_lower for keyword in comparison_keywords),
            'budget_mentioned': None
        }
        
        # Extract budget
        for pattern in budget_patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                intent['budget_mentioned'] = int(match.group(1))
                self.shopping_session['budget_range'] = intent['budget_mentioned']
                break
        
        return intent

    def process_response_for_enhanced_features(self, response_text):
        """Process response for enhanced shopping features."""
        processed_response = {
            'spoken_text': response_text,
            'links_to_display': [],
            'product_cards': [],
            'comparison_data': None,
            'purchase_intent_data': None
        }
        
        # Extract different types of special formatting
        patterns = {
            'display_link': r'\[DISPLAY_LINK:\s*([^|]+)\s*\|\s*([^\]]+)\]',
            'product_card': r'\[PRODUCT_CARD:\s*([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]*)\|([^|]*)\|([^\]]*)\]',
            'compare_products': r'\[COMPARE_PRODUCTS:\s*([^\]]+)\]',
            'purchase_intent': r'\[PURCHASE_INTENT:\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^\]]+)\]'
        }
        
        # Process each pattern
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, response_text)
            
            if pattern_name == 'display_link' and matches:
                processed_response['links_to_display'] = [
                    {'name': name.strip(), 'url': url.strip()} 
                    for name, url in matches
                ]
            
            elif pattern_name == 'product_card' and matches:
                processed_response['product_cards'] = [
                    {
                        'name': match[0].strip(),
                        'url': match[1].strip(),
                        'price': match[2].strip(),
                        'rating': match[3].strip(),
                        'feature1': match[4].strip() if match[4] else '',
                        'feature2': match[5].strip() if match[5] else '',
                        'image_hint': match[6].strip() if match[6] else ''
                    }
                    for match in matches
                ]
            
            elif pattern_name == 'compare_products' and matches:
                # Parse comparison data
                comparison_text = matches[0]
                processed_response['comparison_data'] = comparison_text
            
            elif pattern_name == 'purchase_intent' and matches:
                processed_response['purchase_intent_data'] = {
                    'product_name': matches[0][0].strip(),
                    'url': matches[0][1].strip(),
                    'price': matches[0][2].strip()
                }
        
        # Remove all special formatting from spoken text
        for pattern in patterns.values():
            processed_response['spoken_text'] = re.sub(pattern, '', processed_response['spoken_text'])
        
        processed_response['spoken_text'] = self._optimize_for_voice(processed_response['spoken_text'])
        
        return processed_response

    def chat(self, user_input: str):
        """Enhanced chat handling with intent detection and context."""
        # Detect user intent
        intent = self.detect_user_intent(user_input)
        
        # Add to conversation context
        self.shopping_session['conversation_context'].append({
            'user_input': user_input,
            'intent': intent,
            'timestamp': 'now'  # You might want to use actual timestamps
        })
        
        self.messages.append({"role": "user", "content": user_input})

        # Enhanced system prompt with session context
        context_info = ""
        if self.shopping_session['budget_range']:
            context_info += f"User budget: ${self.shopping_session['budget_range']}. "
        if self.shopping_session['products_viewed']:
            context_info += f"Products discussed: {len(self.shopping_session['products_viewed'])} items. "
        if intent['purchase_intent']:
            context_info += "User showing PURCHASE INTENT - guide to checkout! "
        if intent['comparison_request']:
            context_info += "User wants to COMPARE products - show comparison! "

        voice_system_prompt = (
            "You are having a natural VOICE conversation as a personal shopping assistant. "
            f"SESSION CONTEXT: {context_info}"
            "Key voice conversation rules:\n"
            "- Keep responses conversational and concise (2-4 sentences)\n"
            "- Use natural speech patterns with contractions and casual language\n"
            "- Sound enthusiastic and helpful like a friend helping to shop\n"
            "- Ask ONE specific question at a time, never numbered lists\n"
            "- When presenting products, use [PRODUCT_CARD: name|url|price|rating|feature1|feature2|image_hint]\n"
            "- For purchase intent, use [PURCHASE_INTENT: product_name | url | price]\n"
            "- For comparisons, use [COMPARE_PRODUCTS: detailed_comparison_text]\n"
            "- NEVER speak URLs out loud - they're for the display system\n"
            "- Remember previous products and user preferences in conversation\n"
            "Remember: This is AUDIO - they're hearing you speak, not reading text!"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=voice_system_prompt,
            messages=self.messages,
            tools=self.tool_definitions,
        )

        response_message = response.content[-1]

        while response_message.type == "tool_use":
            tool_call = response_message
            
            self.messages.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_call.id, "name": tool_call.name, "input": tool_call.input}]
            })

            tool_result_data = self.handle_tool_call(tool_call)

            if tool_call.name == "BRAVE_SEARCH__WEB_SEARCH":
                product_links = self.extract_links(tool_result_data)
                if product_links:
                    tool_content = json.dumps({"links": product_links})
                else:
                    tool_content = json.dumps({
                        "message": "No specific products found in this search. Let me try a different approach or get more details."
                    })
            elif tool_call.name == "FIRECRAWL__BATCH_SCRAPE":
                self.extract_product_info(tool_result_data)
                tool_content = json.dumps(tool_result_data)
            else:
                tool_content = json.dumps(tool_result_data)

            self.messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": tool_content}]
            })

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=(
                    "Continue the voice conversation naturally. You just received tool results. "
                    f"SESSION CONTEXT: {context_info}"
                    "If you found products, show them as rich product cards with [PRODUCT_CARD: format]. "
                    "If user showed purchase intent, use [PURCHASE_INTENT: format]. "
                    "If comparison requested, use [COMPARE_PRODUCTS: format]. "
                    "Remember: NEVER speak URLs - use display formats for all visual elements. "
                    "Keep it conversational and flowing - this is a voice chat!"
                ),
                messages=self.messages,
                tools=self.tool_definitions,
            )
            response_message = response.content[-1]

        final_response_text = response_message.text
        
        # Process the enhanced response
        processed_response = self.process_response_for_enhanced_features(final_response_text)
        
        self.messages.append({"role": "assistant", "content": final_response_text})
        return processed_response

    def _optimize_for_voice(self, text):
        """Post-process the response to make it more natural for voice conversations."""
        lines = text.split('\n')
        optimized_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '•', '-')):
                continue
            
            optimized_lines.append(line)
        
        result = ' '.join(optimized_lines)
        
        # Remove any remaining special formats from spoken text
        special_formats = [
            r'\[DISPLAY_LINK:[^\]]+\]',
            r'\[PRODUCT_CARD:[^\]]+\]',
            r'\[COMPARE_PRODUCTS:[^\]]+\]',
            r'\[PURCHASE_INTENT:[^\]]+\]'
        ]
        
        for pattern in special_formats:
            result = re.sub(pattern, '', result).strip()
        
        # Make it more conversational
        replacements = {
            'I apologize': "Sorry about that",
            'Could you please': "Can you",
            'I would be happy to': "I'd love to",
            'assistance': "help",
            'purchase': "buy",
            'provide me with': "tell me",
            'information': "info",
            'however': "but",
            'therefore': "so",
            'additionally': "also",
            'Furthermore': "Plus",
            'In order to': "To",
            'I recommend': "I'd suggest",
            'specifications': "details",
            'http': "",
            'www.': "",
            'amazon.com': "Amazon"
        }
        
        for formal, casual in replacements.items():
            result = result.replace(formal, casual)
        
        # Ensure it's not too long for voice
        sentences = result.split('.')
        if len(sentences) > 4:
            result = '. '.join(sentences[:3]) + '.'
        
        return result.strip()