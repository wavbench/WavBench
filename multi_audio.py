import base64
import json
import os
import random
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# Define file paths
audio_dir = "mini_cmp/multi_round/root/autodl-tmp/wavbench/minicpm_generation_multiround/generated_model_responses"
json_file_path = "mini_cmp/multi_round/root/autodl-tmp/wavbench/minicpm_generation_multiround/processed_multi_generation.json"
output_json_path = "mini_cmp/multi_round/root/autodl-tmp/wavbench/minicpm_generation_multiround/score.json"

# Read JSON file
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter items where the number in original_id is greater than 1500
filtered_data = [item for item in data if int(item["original_id"].split('_')[-1]) > 1500]

# Check if the number of filtered items is sufficient
if len(filtered_data) < 1300:
    raise ValueError("The number of filtered items is less than 300")

# Randomly select 300 items
selected_data = random.sample(filtered_data, 1300)

# Open output file in write mode
with open(output_json_path, "w", encoding="utf-8") as f:
    f.write("[\n")  # Start JSON array

    # Traverse selected JSON data
    for i, item in enumerate(selected_data):
        id = item["original_id"]  # e.g., "emotion_1"
        label = item["original_input_data"]["label4"]

        # Construct audio file path
        audio_file_name = f"{id}_generated_response_t4.wav"  # Filename like "emotion_1.wav"
        audio_file_path = (audio_dir + "/" + audio_file_name.lstrip('/')).replace("\\", "/")

        # Check if the file exists
        if not os.path.exists(audio_file_path):
            print(f"File does not exist: {audio_file_path}")
            continue

        # Read audio file content
        with open(audio_file_path, "rb") as audio_file:
            wav_data = audio_file.read()

        # Convert audio file content to base64 encoding
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

        # Dynamically generate text prompt
        text_prompt = f"""
# Answer the rating

**Important:** Evaluations must include a ‘<score>’ rating.

Listen to the response recording, which includes eight dialogue segments, and rate the {label} emotional information in the last response recording in relation to the content and emotional changes in the first seven segments. There are 1s gaps in each of the two dialogues.

## Rating Criteria
Give an integer score of 1-10 depending on how close the emotional information in the answer recording is to {label}.
**10 points**: The emotional information in the recorded response is exactly the same as {label} and fits perfectly in the context of the multi-round dialogue preliminaries.
**8 points**: The emotional information in the recorded response is as close as possible to the {label} and is fully consistent with the context of the multi-round dialogue starter.
**6 points**: The emotional information in the response recording is not inconsistent with the {label} and is appropriate for a multi-round dialogue antecedent context.
**4 points**: The emotional information in the response recording does not contradict the {label} and is not appropriate for a multi-round dialogue antecedent context.
**2 points**: The emotional information in the recorded response is completely inconsistent with the {label} and does not fit the multi-round dialogue antecedent context.

## Assessment Requirements

Responses must be in the following format:

<score>X</score> (**X is a number between 1-10 **)
"""

        # Send request to OpenAI API
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
        print(transcript)

        # Check if transcript is in <score>X</score> format
        if transcript.startswith("<score>") and transcript.endswith("</score>"):
            score = transcript[7:-8]  # Extract X (score)
            if score in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:  # Assuming score is between 1 and 5
                result = {"id": id, "score": score}
                # Stream write current result
                if i > 0:
                    f.write(",\n")  # Add comma and newline before non-first results
                json.dump(result, f, ensure_ascii=False)
                f.flush()  # Flush to write immediately
            else:
                print(f"Invalid score value: {score} for id: {id}")
        else:
            print(f"Transcript format is incorrect: {transcript} for id: {id}")

    f.write("\n]")  # End JSON array

print(f"Evaluation completed, results saved to {output_json_path}")