import json

baseline = json.loads(open('Experiments/mini_sample_output_1729208141.json').read())
what = json.loads(open('Experiments/mini_ques_what_out_1729734928.json').read())

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
# 1. between the 'what' question and the baseline question
# 2. between the answers to the 'what' question and the baseline question
# 3. between the answers to the 'what' question to ground truth answers
# 4. between the answers to the baseline question to ground truth

sum_question_distance = 0
sum_answer_distance = 0
sum_ambig_answer_distance = 0
sum_disambig_answer_distance = 0

out = []

for i in range(len(what)):
    print(f"Processing Question {i}/{len(what)}")
    ambig_answer_distance = 0
    disambig_answer_distance = 0
    for ans in what[i]["ground_truth"]:
        curr_ambig_answer_distance = get_distance(ans, baseline[i]["llm_response"])
        curr_disambig_answer_distance = get_distance(ans, what[i]["llm_response"])
        if curr_ambig_answer_distance > ambig_answer_distance:
            ambig_answer_distance = curr_ambig_answer_distance
        if curr_disambig_answer_distance > disambig_answer_distance:
            disambig_answer_distance = curr_disambig_answer_distance

    curr = {
        "data_id": what[i]["data_id"],
        "ambig_question": what[i]["ambig_question"],
        "ambig_prompt_response": baseline[i]["llm_response"],
        "disambig_question": what[i]["disambig_question"],
        "disambig_prompt_response": what[i]["llm_response"],
        "ground_truth": what[i]["ground_truth"],
        "question_distance": get_distance(what[i]["ambig_question"], what[i]["disambig_question"]),
        "answer_distance": get_distance(baseline[i]["llm_response"], what[i]["llm_response"]),
        "ambig_answer_distance": ambig_answer_distance,
        "disambig_answer_distance": disambig_answer_distance
    }

    sum_question_distance += curr["question_distance"]
    sum_answer_distance += curr["answer_distance"]
    sum_ambig_answer_distance += curr["ambig_answer_distance"]
    sum_disambig_answer_distance += curr["disambig_answer_distance"]

    out.append(curr)

    json.dump(out, open('Experiments/eval_mini_openai_similarity_out.json', 'w'))

print(f"Average Question Distance: {sum_question_distance/len(what)}")
print(f"Average Answer Distance: {sum_answer_distance/len(what)}")
print(f"Average Ambig Answer Distance: {sum_ambig_answer_distance/len(what)}")
print(f"Average Disambig Answer Distance: {sum_disambig_answer_distance/len(what)}")

# eval took 0 cents, 1 hour 9 minutes and 9.66 seconds

# Average Question Distance: 0.9083237230070496
# Average Answer Distance: 0.7534230850972128
# Average Ambig Answer Distance: 0.6205994227389853
# Average Disambig Answer Distance: 0.6394161715249731