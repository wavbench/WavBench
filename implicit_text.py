import os
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key="",
    base_url="",
)

text_prompt = f"""
**Important:** Evaluation must include a "<score>" rating.

Two paragraphs of text exist below, with the text separated by the ##### separator. The text after the separator is a response to the text before the separator. You need to give an integer score from 1-10 based on the quality of the answer text content.

## Scoring Rubric
You are asked to give a judgment on how reasonable the answer text after the separator is for the text before the separator, giving an integer score of 1-5.
**10 points**: The content of the response text is lively and fits well with the information and emotion of the scene in which the text is asked.
**8 points**: The content of the response fits the scenario of the text, but the response is stiff and there is no clear interaction with the text.
**6 points**: The content of the response text is only relevant to the scenario information in the question text, but does not directly answer the question in a positive way.
**4 points**: The content of the response text is not relevant to the scenario information in the question text, and the scenario information and other constructive factors are completely incompatible.
**2 points**: The content of the response text is not a fluent expression, with grammatical errors and unclear semantics.

## Assessment Requirements

Responses must be in the following format:

<score>X</score> (**X is a number between 1 - 10 **)
"""

# Read the id list from score.json
try:
    with open('score.json', 'r', encoding='utf-8') as f:
        score_data = json.load(f)
    ids = [item['id'] for item in score_data]
except Exception as e:
    print(f"Error reading score.json: {e}")
    exit(1)

# Read data from processed_responses_implicit_generation.json
try:
    with open('processed_responses_implicit_generation.json', 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
except Exception as e:
    print(f"Error reading processed_responses_implicit_generation.json: {e}")
    exit(1)

# Iterate through each id
for id in ids:
    # Find the corresponding entry
    for item in processed_data:
        if item['original_id'] == id:
            original_text = item['original_user_input_text']
            generated_text = item['generated_model_text_response']
            break
    else:
        print(f"ID {id} not found in processed_responses_implicit_generation.json")
        continue

    # Combine the texts
    combined_text = original_text + ' ##### ' + generated_text

    # Send request to Qwen3 model
    try:
        completion = client.chat.completions.create(
            model="qwen3-32b",
            messages=[
                {"role": "system", "content": text_prompt},
                {"role": "user", "content": combined_text},
            ],
            extra_body={"enable_thinking": False},
        )
        response = completion.model_dump_json()
        # Extract the score
        content = json.loads(response)['choices'][0]['message']['content']
        score = content.split('<score>')[1].split('</score>')[0]
    except Exception as e:
        print(f"Error processing ID {id}: {e}")
        continue

    # Save id and score to score_txt.json
    try:
        with open('score_txt.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps({"id": id, "score": score}) + '\n')
    except Exception as e:
        print(f"Error writing to score_txt.json: {e}")
