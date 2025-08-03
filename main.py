import io, os, json, asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from elevenlabs.client import ElevenLabs
import logging

from conversational_agent import ConversationalAmazonAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Setup ---
app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000", 
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

# Global agent instance - each WebSocket connection will get its own
eleven_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection for real-time audio conversation with enhanced features."""
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    
    # Create a new agent instance for this connection
    agent = ConversationalAmazonAgent()
    
    try:
        while True:
            try:
                data = await websocket.receive()
                
                # Check if it's bytes (audio) or text (command)
                if 'bytes' in data:
                    audio_data = data['bytes']
                    logger.info(f"Received audio data from client. Size: {len(audio_data)} bytes")

                    try:
                        # Validate audio data
                        if len(audio_data) < 1000:  # Too small to be valid audio
                            logger.warning("Audio data too small, skipping")
                            await websocket.send_text("Audio too short. Please speak for at least 1 second.")
                            continue

                        audio_file = io.BytesIO(audio_data)
                        audio_file.name = "audio.webm"  # Add filename hint

                        # Add timeout for transcription
                        transcribed_text = eleven_client.speech_to_text.convert(
                            file=audio_file,
                            model_id="scribe_v1",
                            tag_audio_events=True,
                            language_code="eng",
                            diarize=True,
                        )

                        user_query = transcribed_text.text.strip()
                        logger.info(f"Transcribed Text: '{user_query}'")
                        
                        if not user_query:
                            logger.warning("Transcription is empty")
                            await websocket.send_text("I didn't catch that. Could you please speak again?")
                            continue

                        logger.info("Getting response from conversational agent...")
                        agent_response = agent.chat(user_query)  # Enhanced response object
                        
                        spoken_text = agent_response['spoken_text']
                        
                        logger.info(f"Agent Spoken Response: '{spoken_text}'")

                        # Handle enhanced features
                        await handle_enhanced_features(websocket, agent_response)

                        # Generate TTS response for the spoken text only
                        if spoken_text.strip():  # Only generate audio if there's something to say
                            await generate_and_send_audio(websocket, spoken_text, "response")

                    except Exception as e:
                        error_message = f"Error processing audio: {str(e)}"
                        logger.error(error_message)
                        await websocket.send_text("Sorry, I had trouble processing your audio. Please try again.")
                        
                elif 'text' in data:
                    # Handle JSON commands
                    try:
                        command_data = json.loads(data['text'])
                        command_type = command_data.get('type')
                        
                        if command_type == 'reset':
                            logger.info("Resetting agent conversation...")
                            agent.reset()
                            logger.info("Agent reset completed.")
                            # Send confirmation back to client
                            await websocket.send_text(json.dumps({"type": "reset_complete"}))
                            
                        elif command_type == 'intro':
                            logger.info("Playing intro message...")
                            intro_text = command_data.get('text', "Hello! I'm your shopping assistant.")
                            
                            # Generate intro audio
                            await generate_and_send_audio(websocket, intro_text, "intro")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON command: {e}")
                        await websocket.send_text("Invalid command format.")
                    except Exception as e:
                        error_message = f"Error processing command: {str(e)}"
                        logger.error(error_message)
                        await websocket.send_text(error_message)
            
            except WebSocketDisconnect:
                logger.info("Client disconnected normally.")
                break
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                try:
                    await websocket.send_text("An error occurred. Please try again.")
                except:
                    break  # Connection is likely broken

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket endpoint: {e}")
    finally:
        logger.info("WebSocket connection closed.")

async def handle_enhanced_features(websocket: WebSocket, agent_response):
    """Handle enhanced shopping features like product cards, purchase intent, etc."""
    try:
        # Handle product cards
        if agent_response.get('product_cards'):
            logger.info(f"Sending {len(agent_response['product_cards'])} product cards")
            cards_message = {
                "type": "product_cards",
                "cards": agent_response['product_cards']
            }
            await websocket.send_text(json.dumps(cards_message))
            await asyncio.sleep(0.3)  # Small delay for UI processing

        # Handle purchase intent
        if agent_response.get('purchase_intent_data'):
            logger.info("Sending purchase intent modal trigger")
            purchase_message = {
                "type": "purchase_intent",
                "product": agent_response['purchase_intent_data']
            }
            await websocket.send_text(json.dumps(purchase_message))
            await asyncio.sleep(0.2)

        # Handle comparison data
        if agent_response.get('comparison_data'):
            logger.info("Sending comparison data")
            comparison_message = {
                "type": "comparison",
                "data": agent_response['comparison_data']
            }
            await websocket.send_text(json.dumps(comparison_message))
            await asyncio.sleep(0.2)

        # Handle regular links (fallback)
        if agent_response.get('links_to_display') and not agent_response.get('product_cards'):
            logger.info("Sending regular links")
            links_message = {
                "type": "display_links",
                "links": agent_response['links_to_display']
            }
            await websocket.send_text(json.dumps(links_message))
            await asyncio.sleep(0.2)

    except Exception as e:
        logger.error(f"Error handling enhanced features: {e}")

async def generate_and_send_audio(websocket: WebSocket, text: str, audio_type: str = "response"):
    """Generate TTS audio and send it to the client with error handling."""
    try:
        logger.info(f"Converting {audio_type} to speech with ElevenLabs...")
        
        # Add retry logic for TTS generation
        max_retries = 2
        for attempt in range(max_retries):
            try:
                audio_stream = eleven_client.text_to_speech.convert(
                    text=text,
                    voice_id="JBFqnCBsd6RMkjVDRZzb", 
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                )
                
                # Collect audio chunks
                audio_bytes_list = []
                chunk_count = 0
                for chunk in audio_stream:
                    audio_bytes_list.append(chunk)
                    chunk_count += 1
                
                if chunk_count == 0:
                    raise Exception("No audio data received from TTS service")
                
                full_audio_bytes = b"".join(audio_bytes_list)
                
                if len(full_audio_bytes) < 100:  # Validate audio size
                    raise Exception(f"Generated audio too small: {len(full_audio_bytes)} bytes")
                
                logger.info(f"Generated audio: {len(full_audio_bytes)} bytes, {chunk_count} chunks")
                
                # Send audio to client
                await websocket.send_bytes(full_audio_bytes)
                logger.info(f"Sent {audio_type} audio back to client.")
                return
                
            except Exception as attempt_error:
                logger.warning(f"TTS attempt {attempt + 1} failed: {attempt_error}")
                if attempt == max_retries - 1:
                    raise attempt_error
                await asyncio.sleep(1)  # Wait before retry
                
    except Exception as e:
        error_message = f"Error generating {audio_type} audio: {str(e)}"
        logger.error(error_message)
        
        # Send fallback text message
        fallback_message = "I'm having trouble with audio right now. " + text
        try:
            await websocket.send_text(fallback_message)
        except:
            logger.error("Failed to send fallback text message")