import json

# 定义要处理的 JSON 文件列表
json_files = [
    'processed_explicit_understanding_accent.json',
    'processed_explicit_understanding_volume.json',
    'processed_explicit_understanding_emotion.json',
    'processed_explicit_understanding_gender.json',
    'processed_explicit_understanding_lang.json',
    'processed_explicit_understanding_pitch.json',
    'processed_explicit_understanding_speed.json'
]


def is_correct(original_label, model_response_text, is_lang_file=False):
    """
    判断一个条目是否正确。
    参数：
        original_label: 原始标签
        model_response_text: 模型响应文本
        is_lang_file: 是否是语言文件
    返回：
        True 表示正确，False 表示错误
    """
    if is_lang_file and original_label.lower() == 'chinese':
        # 对于语言文件，如果 original_label 是 "chinese"，检查中文字符
        return any(char in model_response_text for char in ['中', '普通', '汉'])
    else:
        # 普通规则：检查 original_label 是否在 model_response_text 中（忽略大小写）
        return original_label.lower() in model_response_text.lower()


# 打开 score.txt 文件，写入结果
with open('score.txt', 'w', encoding='utf-8') as f:
    for filename in json_files:
        # 判断是否是语言文件
        is_lang_file = filename == 'processed_explicit_understanding_lang.json'

        # 读取 JSON 文件
        with open(filename, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            correct_count = 0  # 正确条目数
            total_count = 0  # 总条目数

            # 遍历每个条目
            for item in data:
                original_label = item['original_label']
                model_response_text = item['model_response_text']
                if is_correct(original_label, model_response_text, is_lang_file):
                    correct_count += 1
                total_count += 1

            # 计算正确率
            if total_count > 0:
                correct_rate = correct_count / total_count
            else:
                correct_rate = 0.0  # 如果文件为空，正确率为 0

            # 将结果写入 score.txt
            f.write(f"{filename}: {correct_rate:.4f}\n")