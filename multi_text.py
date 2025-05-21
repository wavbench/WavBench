import os
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key="",
    base_url="",
)

text_prompt = f"""
**Important:** Evaluations must include a ‘<score>’ rating.

There are eight paragraphs of text below, with the text separated by #####. They correspond to the content of each of the eight multi-round dialogues. You are asked to rate the reasonableness of the last paragraph based on the content of the first seven paragraphs, giving it an integer score of 1-10.

## Scoring Criteria
Give an integer score of 1-5 for how well the last paragraph makes sense in relation to the first seven paragraphs.
**10 marks**: The textual content of the response is lively and fits well with the message and emotion of the first seven scenes.
**8 marks**: The content of the answer fits the textual scenario, but the answer is stiff and there is no clear interaction with the text of the first seven paragraphs.
**6 marks**: The textual content of the answer relates only to the information of the scenario in the text of the first seven paragraphs but does not answer the question directly and positively.
**4 marks**: The textual content of the answer is not relevant to the scenario information in the text of the first seven paragraphs and the scenario information is not at all consistent with the other constructive elements.
**2 marks**: The response is poorly expressed, with grammatical errors and unclear semantics.

## Assessment Requirements

Responses must be in the following format:

<score>X</score> (**X is a number between 1-10 **)
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
    with open('processed_multi_generation.json', 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
except Exception as e:
    print(f"Error reading processed_responses_implicit_generation.json: {e}")
    exit(1)

# Iterate through each id
for id in ids:
    # Find the corresponding entry
    for item in processed_data:
        if item['original_id'] == id:
            user1 = item["original_input_data"]["user1"]
            response1 = item["original_input_data"]["model1"]
            user2 = item["original_input_data"]["user2"]
            response2 = item["original_input_data"]["model2"]
            user3 = item["original_input_data"]["user3"]
            response3 = item["original_input_data"]["model3"]
            user4 = item["original_input_data"]["user4"]
            response4 = item["original_input_data"]["model4"]
            break
    else:
        print(f"ID {id} not found in processed_responses_implicit_generation.json")
        continue

    # Combine the texts
    combined_text = user1 + ' ##### ' + response1 + ' ##### ' + user2 + ' ##### ' + response2 + ' ##### ' + user3 + ' ##### ' + response3 + ' ##### ' + user4 + ' ##### ' + response4

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