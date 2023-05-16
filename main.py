import os
import time
import json
import pyaudio
import queue
from google.cloud import speech
from PIL import Image, ImageGrab
import pytesseract
import openai
import threading

# Set OpenAI API key
openai.api_key = ""

# Set up Google ASR client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            input_device_index=2, # Adjust this with the index of your device
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses):
    """Iterates through server responses and prints them."""
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        print(transcript)

def transcribe_audio():
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='en-US',
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)
        responses = client.streaming_recognize(streaming_config, requests)

        listen_print_loop(responses)

ocr_text = ""  # define ocr_text as a global variable

def process_screenshot():
    global ocr_text  # declare ocr_text as global
    # Capture and save the screenshot
    image = ImageGrab.grab()
    image.save("screenshot.png")

    # OCR the screenshot
    image = Image.open("screenshot.png")
    ocr_text = pytesseract.image_to_string(image)
    print("OCR text from screenshot:", ocr_text)

def threaded_transcribe_and_process_screenshot():
    # Create threads for real-time transcription and OCR processing
    transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
    ocr_thread = threading.Thread(target=process_screenshot)

    # Start the threads
    transcription_thread.start()
    ocr_thread.start()

    # Wait for OCR thread to finish
    ocr_thread.join()

def generate_quiz(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Save the full API response to a JSON file
    with open("openai_response.json", "w") as outfile:
        json.dump(response.to_dict(), outfile, indent=4)

    quiz = response.choices[0].text.strip()
    return quiz

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print("Next question in:", timeformat, end='\r')
        time.sleep(1)
        t -= 1
    print('\nGenerating next quiz...')

while True:
    threaded_transcribe_and_process_screenshot()
    prompt = f"Here is a single OCR and speech-to-text result from a screenshot during a lecture to some college students:\n\nOCR Results:\n{ocr_text}\n\nCan you try and come up with a multiple choice quiz question and answer based on the results of the OCR and what was said? If you can't find a good one, then simply try and quiz them on what was said."

    quiz = generate_quiz(prompt)
    print("Generated Quiz:\n", quiz)
    countdown(25)  # Countdown to the next quiz question