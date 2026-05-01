"""
    Convert the data to reasoning traces samples to train the reward model. 
"""

import pandas as pd
import json
import random
import re


unknown_counter = 0
UNKNOWN_LIMIT = 150

def remove_label_pattern(text):
    jsutification = re.sub(r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))", "", text, flags=re.IGNORECASE).strip()
    return jsutification.replace("\n", " ")

def sample_training_examples(row):
    global unknown_counter

    label = row["label"].lower()
    verdict_list = [v.lower() for v in row["Verdict_list"]]

    # Step 1: Identify indices of different types
    correct_indices = [i for i, v in enumerate(verdict_list) if v == label]
    true_indices = [i for i, v in enumerate(verdict_list) if v == "true" and v != label]
    false_indices = [i for i, v in enumerate(verdict_list) if v == "false" and v != label]
    conflicting_indices = [i for i, v in enumerate(verdict_list) if v == "conflicting" and v != label]
    unknown_indices = [i for i, v in enumerate(verdict_list) if v == "unknown"]

    selected_indices = []

    # Step 2: Sample correct ones if available
    if len(correct_indices) >= 2:
        selected_indices.extend(random.sample(correct_indices, 2))
        num_remaining = 4
    elif len(correct_indices) == 1:
        selected_indices.append(correct_indices[0])
        num_remaining = 5
    else:
        num_remaining = 6

    # Step 3: Fill remaining slots with diverse wrong answers
    wrong_indices = []
    if label != "true" and true_indices:
        wrong_indices.append(random.choice(true_indices))
    if label != "false" and false_indices:
        wrong_indices.append(random.choice(false_indices))
    if label != "conflicting" and conflicting_indices:
        wrong_indices.append(random.choice(conflicting_indices))

    # Step 4: Ensure diversity while filling remaining slots
    wrong_indices = list(set(wrong_indices))  # Remove duplicates
    needed = num_remaining - len(wrong_indices)

    # Add extra wrong ones if needed
    all_wrong_indices = true_indices + false_indices + conflicting_indices
    random.shuffle(all_wrong_indices)
    wrong_indices.extend(all_wrong_indices[:needed])

    selected_indices.extend(wrong_indices[:num_remaining])

    # Step 5: Handle case where all values are "unknown"
    if not selected_indices and unknown_indices:
        # If everything is unknown, just take 5 unknowns
        selected_indices = random.sample(unknown_indices, min(5, len(unknown_indices)))
    elif unknown_indices and unknown_counter < UNKNOWN_LIMIT:
        # Only pop if there is something to pop
        if selected_indices:
            selected_indices.pop()
        selected_indices.append(random.choice(unknown_indices))
        unknown_counter += 1

    return selected_indices


# data = pd.read_json("reasoning_traces/quantemp_English_train.jsonl", lines=True)
data = pd.DataFrame(read_json("output/reasoning_generation/English_train.json"))

# Apply function to each row. sampled_indices contain the indices of the sampled dataset
data["sampled_indices"] = data.apply(sample_training_examples, axis=1)


final_training_data = []

for idx in range(len(data)): 
    class_label = 0
    item = data.loc[idx]
    
    
    label = item['label'].lower()
    
    
    
    
    for decoding_idx, decoding_sample in enumerate(item["sampled_indices"]):
        
        justification = remove_label_pattern(item["Reasoning_traces"][decoding_sample])
        verdict = item['Verdict_list'][decoding_sample].lower()
        
        sample_id = str(item["query_id"]) + "_" + chr(97 + decoding_idx) 
        
        
        if verdict == label:
            class_label = 1
        else:
            class_label = 0

        
        
        # Handles empty justification
        if len(justification.split(" ")) < 3:
            continue
        
        
        final_training_data.append({
            "sample_id": sample_id, 
            "input_text": f"Claim: {item['claim']}\nVerdict: {verdict}\nJustification: {justification}",
            "Label": label, 
            "Verdict": verdict,
            "Class": class_label})    
    
print(len(final_training_data))
#with open("output/training_data_for_RM/English_train.json", "w") as fp:
#    json.dump(final_training_data, fp, indent = 4)
pd.DataFrame(final_training_data).to_json(
     "output/training_data_for_RM/English_val.jsonl",
     orient="records",
     lines=True,
     force_ascii=False
 )

print("Finshed preprocessing the data.")
# print(final_training_data[100])

# for sample in final_training_data:
#     if sample['Verdict'].lower() == 'unknown':
#         print(sample['sample_id'])
        
# import json
# import pandas as pd

# with open("output/training_data_for_RM/English_val.json", "r", encoding="utf-8") as f:
#     data = json.load(f)          # list of dicts

# pd.DataFrame(data).to_json(
#     "output/training_data_for_RM/English_val.jsonl",
#     orient="records",
#     lines=True,
#     force_ascii=False
# )
