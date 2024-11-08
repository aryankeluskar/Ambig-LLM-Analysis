import json

baseline = json.loads(open('Experiments/o_sample_output_1729209119.json').read())
file_to_compare = json.loads(open('Experiments/o_low_temp_output_1729792390.json').read())

SIZE = 1000

import os
from openai import OpenAI
import numpy as np

import dotenv
dotenv.load_dotenv()

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)

def cosine_similarity(vec1, vec2):
    # Ensure the vectors are numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Compute the dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    # Prevent division by zero
    if magnitude == 0:
        return 0.0
    
    return dot_product / magnitude

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_distance(text1, text2, model="text-embedding-3-large"):
   text1 = str(text1).lower()
   text2 = str(text2).lower()
   embedding1 = get_embedding(text1, model)
   embedding2 = get_embedding(text2, model)
   return cosine_similarity(embedding1, embedding2)

# distances to calculate:
# 1. between the 'file_to_compare' question and the baseline question
# 2. between the answers to the 'file_to_compare' question and the baseline question
# 3. between the answers to the 'file_to_compare' question to ground truth answers
# 4. between the answers to the baseline question to ground truth

sum_question_distance = 0
sum_answer_distance = 0
sum_ambig_answer_distance = 0
sum_disambig_answer_distance = 0

out = []

import time
start = time.time()

# 8
# range(8) = [0, 1, 2, 3, 4, 5, 6, 7]

for i in range(SIZE):
    print(f"Processing Question {i}/{SIZE}")

    ambig_answer_distance = 0
    disambig_answer_distance = 0
    for ans in file_to_compare[i]["ground_truth"]:
        curr_ambig_answer_distance = get_distance(ans, baseline[i]["llm_response"])
        curr_disambig_answer_distance = get_distance(ans, file_to_compare[i]["llm_response"])
        if curr_ambig_answer_distance > ambig_answer_distance:
            ambig_answer_distance = curr_ambig_answer_distance
        if curr_disambig_answer_distance > disambig_answer_distance:
            disambig_answer_distance = curr_disambig_answer_distance

    curr = {
        "data_id": file_to_compare[i]["data_id"],
        "ambig_question": file_to_compare[i]["ambig_question"],
        "ambig_prompt_response": baseline[i]["llm_response"],
        # "disambig_question": file_to_compare[i]["disambig_question"],
        "disambig_prompt_response": file_to_compare[i]["llm_response"],
        "ground_truth": file_to_compare[i]["ground_truth"],
        # "question_distance": get_distance(file_to_compare[i]["ambig_question"], file_to_compare[i]["disambig_question"]),
        "answer_distance": get_distance(baseline[i]["llm_response"], file_to_compare[i]["llm_response"]),
        "ambig_answer_distance": ambig_answer_distance,
        "disambig_answer_distance": disambig_answer_distance
    }

    # sum_question_distance += curr["question_distance"]
    sum_answer_distance += curr["answer_distance"]
    sum_ambig_answer_distance += curr["ambig_answer_distance"]
    sum_disambig_answer_distance += curr["disambig_answer_distance"]

    out.append(curr)

    json.dump(out, open('Experiments/eval_o_low_temp_openai_similarity_out.json', 'w'))

# print(f"Average Question Distance: {sum_question_distance/SIZE}")
print(f"Average Answer Distance: {sum_answer_distance/SIZE}")
print(f"Average Ambig Answer Distance: {sum_ambig_answer_distance/SIZE}")
print(f"Average Disambig Answer Distance: {sum_disambig_answer_distance/SIZE}")

print(f"Time taken: {time.time() - start} seconds")

with open('Experiments/eval_4o_low_temp_openai_similarity.txt', 'a') as f:
    # f.write(f"Average Question Distance: {sum_question_distance/SIZE}\n")
    f.write(f"Average Answer Distance: {sum_answer_distance/SIZE}\n")
    f.write(f"Average Ambig Answer Distance: {sum_ambig_answer_distance/SIZE}\n")
    f.write(f"Average Disambig Answer Distance: {sum_disambig_answer_distance/SIZE}\n")
    f.write(f"Time taken: {time.time() - start} seconds\n")

# This took 1 hour, 10 minutes and 22.39 seconds