import base64
import json
import os
import random
from openai import OpenAI
from pathlib import PurePosixPath

# Initialize OpenAI client
client = OpenAI()

# Define file types and paths
file_types = ["accent", "age", "emotion", "gender", "lang", "pitch", "speed", "volume"]
audio_prefix = PurePosixPath("mini_cmp/single_round")  # Use PurePosixPath to force forward slashes
output_dir = "score"

# Ensure the score directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through each type
for file_type in file_types:
    json_file_path = f"processed_explicit_generation_{file_type}.json"
    output_json_path = os.path.join(output_dir, f"explicit_generation_{file_type}_score.json")

    # Read JSON file
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File does not exist: {json_file_path}")
        continue

    # Randomly select 128 items
    if len(data) < 1024:
        print(f"Warning: {json_file_path} has fewer than 128 items, only {len(data)} items available")
        selected_data = data
    else:
        selected_data = random.sample(data, 1024)

    # Open output file in write mode (streaming output)
    with open(output_json_path, "w", encoding="utf-8") as f:
        f.write("[\n")  # Start JSON array

        for i, item in enumerate(selected_data):
            original_id = item["original_id"]
            label = item["original_label"]
            model_response_audio_path = item["model_response_audio_path"]

            # Extract audio filename
            audio_filename = os.path.basename(model_response_audio_path)

            # Construct the correct relative path
            relative_path = PurePosixPath("root/autodl-tmp/wavbench/minicpm_explicit_output/generated_audio") / audio_filename

            # Concatenate to form the full audio file path
            audio_file_path = audio_prefix / relative_path

            # Convert to string for file operations
            audio_file_path_str = str(audio_file_path)

            # Print the path for debugging
            print(f"Attempting to access audio file: {audio_file_path_str}")

            # Check if the file exists
            if not os.path.exists(audio_file_path_str):
                print(f"Audio file does not exist: {audio_file_path_str}")
                continue

            # Read audio file content
            try:
                with open(audio_file_path_str, "rb") as audio_file:
                    wav_data = audio_file.read()
            except Exception as e:
                print(f"Failed to read audio file: {audio_file_path_str}, error: {e}")
                continue

            # Convert audio file content to base64 encoding
            encoded_string = base64.b64encode(wav_data).decode('utf-8')

            # Dynamically generate text prompt
            text_prompt = f"""
# Answer response evaluations

**Important:** Evaluation must include a “<score>” rating.

Listen to the response recording and evaluate the response recording for {label} {file_type} information.

## Scoring Criteria

**1 point**: The {file_type} message of the recorded response is {label}.
**0 points**: The {file_type} message of the recorded response is not {label}.

## Evaluation Requirements

The response **must** follow this format:

<score>X</score> (**X is 1 or 0**)
"""

            # Send request to OpenAI API
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-audio-preview",
                    modalities=["text", "audio"],
                    audio={"voice": "alloy", "format": "wav"},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": text_prompt
                                },
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": encoded_string,
                                        "format": "wav"
                                    }
                                }
                            ]
                        },
                    ]
                )

                # Extract transcript
                transcript = completion.choices[0].message.audio.transcript
                print(f"Transcript for {original_id}: {transcript}")

                # Check if transcript is in <score>X</score> format
                if transcript.startswith("<score>") and transcript.endswith("</score>"):
                    score = transcript[7:-8]  # Extract X (score)
                    if score in ["0", "1"]:
                        result = {"id": original_id, "score": score}
                        # Stream write current result
                        if i > 0:
                            f.write(",\n")  # Add comma and newline before non-first results
                        json.dump(result, f, ensure_ascii=False)
                    else:
                        print(f"Invalid score value: {score} for id: {original_id}")
                else:
                    print(f"Transcript format is incorrect: {transcript} for id: {original_id}")
            except Exception as e:
                print(f"API call failed for id: {original_id}, error: {e}")
                continue

        f.write("\n]")  # End JSON array

    print(f"Evaluation completed for {file_type}, results saved to {output_json_path}")