import boto3
import json
import os

config_file_path = os.path.join(os.path.dirname(__file__), "audio_config.json")
with open(config_file_path, "r") as config_file:
    config = json.load(config_file)

def tts(message):
    try:
        if len(message) > 1500:
            message = message[:1500]
        polly_client = boto3.client('polly', 
                                aws_access_key_id=config.get("aws_access_key_id", None),
                                aws_secret_access_key=config.get("aws_secret_access_key", None),
                                region_name=config.get("region_name", None))
        response = polly_client.synthesize_speech(
            Text=message,
            OutputFormat=config.get("OutputFormat", None),
            VoiceId=config.get("engVoiceId", None),
        )
        print("Audio synthesis successful.")
    except Exception as e:
        print("Error during audio synthesis:", e)
        return None
    
    return response['AudioStream']