import assemblyai as aai
import json
import os

config_file_path = os.path.join(os.path.dirname(__file__), "audio_config.json")
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)

def stt(audio):
    """
    Transcribe audio to text using AssemblyAI or AWS Transcribe.
    """
    try:
        # Transcription using AssemblyAI
        aai.settings.api_key = config.get("aai_api_key", None)
        config = aai.TranscriptionConfig(language_code=config.get("en_language_code", None))
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio)
        transcribed_text = transcript.text
        """
        # Transcription using AWS Transcribe
        s3 = boto3.client(
            "s3",
            aws_access_key_id=config.get("aws_access_key_id", None),
            aws_secret_access_key=config.get("aws_secret_access_key", None),
            region_name="us-east-1"  # Cambia con la tua regione
        )
        s3.upload_file(file_name, bucket_name, s3_key)
        transcribe_client = boto3.client('transcribe', 
                                        aws_access_key_id='',
                                        aws_secret_access_key='',
                                        region_name='eu-north-1')
        """
        print("Transcription successful.")
        return transcribed_text
    except Exception as e:
        print("Error during transcription:", e)
        return None