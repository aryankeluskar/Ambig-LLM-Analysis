import json

baseline = json.loads(open('Experiments/mini_sample_output_1729208141.json').read())
file_to_compare = json.loads(open('Experiments/mini_add_context_output_1729750587.json').read())

SIZE = 1000

import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

import dotenv
dotenv.load_dotenv()

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_distance(text1, text2, model="text-embedding-3-small"):
   text1 = str(text1).lower()
   text2 = str(text2).lower()
   embedding1 = get_embedding(text1, model)
   embedding2 = get_embedding(text2, model)
   return cosine_similarity([embedding1], [embedding2])[0][0]

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
        "disambig_question": file_to_compare[i]["disambig_question"],
        "disambig_prompt_response": file_to_compare[i]["llm_response"],
        "ground_truth": file_to_compare[i]["ground_truth"],
        "question_distance": get_distance(file_to_compare[i]["ambig_question"], file_to_compare[i]["disambig_question"]),
        "answer_distance": get_distance(baseline[i]["llm_response"], file_to_compare[i]["llm_response"]),
        "ambig_answer_distance": ambig_answer_distance,
        "disambig_answer_distance": disambig_answer_distance
    }

    sum_question_distance += curr["question_distance"]
    sum_answer_distance += curr["answer_distance"]
    sum_ambig_answer_distance += curr["ambig_answer_distance"]
    sum_disambig_answer_distance += curr["disambig_answer_distance"]

    out.append(curr)

    json.dump(out, open('Experiments/eval_mini_add_context_openai_similarity_out.json', 'w'))

print(f"Average Question Distance: {sum_question_distance/SIZE}")
print(f"Average Answer Distance: {sum_answer_distance/SIZE}")
print(f"Average Ambig Answer Distance: {sum_ambig_answer_distance/SIZE}")
print(f"Average Disambig Answer Distance: {sum_disambig_answer_distance/SIZE}")

# eval took 0 cents, 1 hour 9 minutes and 9.66 seconds

# Average Question Distance: 0.9083237230070496
# Average Answer Distance: 0.7534230850972128
# Average Ambig Answer Distance: 0.6205994227389853
# Average Disambig Answer Distance: 0.6394161715249731