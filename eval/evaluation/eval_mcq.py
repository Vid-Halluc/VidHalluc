import json
import sys
import re


def extract_answer(model_answer):
    """Extract all answer choices (A-D) from the model's response.

    Returns a set of uppercase letters found in the answer.
    """
    choices = re.findall(r'\b[A-D]\b', model_answer)
    if choices:
        return set([c.upper() for c in choices])
    return set()


def match_model_answer(data):
    """
    For cases where the model output does not include a direct choice letter,
    this function searches the provided "Choices" mapping for an option whose
    value appears in the model answer text. The found choice key is stored in
    the field "Matched Answer" (or "None" if no match is found).
    """
    for key, clips in data.items():
        for clip_id, clip_data in clips.items():
            model_answer = clip_data.get("Model Answer", "").lower()
            choices = clip_data.get("Choices", {})
            matched_answer = None
            for choice_key, choice_value in choices.items():
                if choice_value.lower() in model_answer:
                    matched_answer = choice_key
                    break
            clip_data["Matched Answer"] = matched_answer if matched_answer else "None"


def compute_MCQ(gt_data, pred_data):
    """
    Compute the number of MCQ questions and the number answered correctly.
    For each question, the ground truth "Correct Answer" is compared with the
    prediction's "Model Answer" (or "Matched Answer" if no answer letter is found).
    """
    count = 0
    correct_count = 0
    for key, gt_values in gt_data.items():
        pred_values = pred_data.get(key, {})
        if not pred_values:
            print(f"Missing key in prediction: {key}")
            continue
        for video, gt_question in gt_values.items():
            count += 1
            correct_answer = gt_question['Correct Answer'].strip().upper()
            correct_answer_set = set(re.findall(r'[A-D]', correct_answer))
            pred_question = pred_values.get(video, {})
            if not pred_question:
                print(f"Missing video {video} in prediction for key {key}")
                continue
            model_answer_set = extract_answer(pred_question.get('Model Answer', ""))
            # If no direct answer letter is found, try the "Matched Answer"
            if not model_answer_set:
                matched = pred_question.get("Matched Answer", "").strip().upper()
                if matched and matched != "NONE":
                    model_answer_set = set([matched])
            if model_answer_set == correct_answer_set:
                correct_count += 1
    return count, correct_count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file.json>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

    # The JSON file must have "gt" and "pred" top-level keys
    gt_data = data.get("gt", {})
    pred_data = data.get("pred", {})
    if not gt_data or not pred_data:
        print("Input JSON must contain both 'gt' and 'pred' keys.")
        sys.exit(1)

    # Run match_model_answer on the prediction data to populate "Matched Answer" if needed.
    match_model_answer(pred_data)

    total_questions, correct_questions = compute_MCQ(gt_data, pred_data)
    if total_questions > 0:
        print(correct_questions, total_questions)
        accuracy = correct_questions / total_questions
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("No data available for accuracy computation.")
