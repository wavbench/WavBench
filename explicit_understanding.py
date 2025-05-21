import json

# Define the list of JSON files to process
json_files = [
    'mini_cmp/single_round/root/autodl-tmp/wavbench/minicpm_explicit_output/processed_explicit_understanding_accent.json',
    'mini_cmp/single_round/root/autodl-tmp/wavbench/minicpm_explicit_output/processed_explicit_understanding_volume.json',
    'mini_cmp/single_round/root/autodl-tmp/wavbench/minicpm_explicit_output/processed_explicit_understanding_emotion.json',
    'mini_cmp/single_round/root/autodl-tmp/wavbench/minicpm_explicit_output/processed_explicit_understanding_gender.json',
    'mini_cmp/single_round/root/autodl-tmp/wavbench/minicpm_explicit_output/processed_explicit_understanding_lang.json',
    'mini_cmp/single_round/root/autodl-tmp/wavbench/minicpm_explicit_output/processed_explicit_understanding_pitch.json',
    'mini_cmp/single_round/root/autodl-tmp/wavbench/minicpm_explicit_output/processed_explicit_understanding_speed.json'
]

def is_correct(original_label, model_response_text, is_lang_file=False):
    """
    Determine if an entry is correct.
    Parameters:
        original_label: the original label
        model_response_text: the model response text
        is_lang_file: whether it is the language file
    Returns:
        True if correct, False otherwise
    """
    if is_lang_file and original_label.lower() == 'chinese':
        # For the language file, if original_label is 'chinese', check for Chinese characters
        return any(char in model_response_text for char in ['中', '普通', '汉'])
    else:
        # General rule: check if original_label is in model_response_text (case-insensitive)
        return original_label.lower() in model_response_text.lower()

# Open the score.txt file and write the results
with open('score.txt', 'w', encoding='utf-8') as f:
    for filename in json_files:
        # Determine if it is the language file
        is_lang_file = filename == 'processed_explicit_understanding_lang.json'
        
        # Read the JSON file
        with open(filename, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            correct_count = 0  # Number of correct entries
            total_count = 0  # Total number of entries
            
            # Iterate through each entry
            for item in data:
                original_label = item['original_label']
                model_response_text = item['model_response_text']
                if is_correct(original_label, model_response_text, is_lang_file):
                    correct_count += 1
                total_count += 1
            
            # Calculate the correctness rate
            if total_count > 0:
                correct_rate = correct_count / total_count
            else:
                correct_rate = 0.0  # If the file is empty, the correctness rate is 0
            
            # Write the results to score.txt
            f.write(f"{filename}: {correct_rate:.4f}\n")
