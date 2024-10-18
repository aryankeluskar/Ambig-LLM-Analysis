import os
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

client = OpenAI(
  api_key=os.environ.get("DMML_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "",
        }
    ],
    model="gpt-4o-mini",
)

str(chat_completion.choices[0].message.content)

# go through sample.json and ask questions to gpt
import json
all_qs = json.load(open("data/filtered_train.json"))

# randomly sample 0 questions
import random
sample = random.sample(all_qs, 1000)

# store the sample input into sample_input_<current_unix_time>.json
import time
current_unix_time = int(time.time())
with open(f"Experiments/mini_sample_input_{current_unix_time}.json", "w") as f:
    json.dump(sample, f)

count = 0
out = []
for i in sample:
    count += 1
    print(f"Question {count}")

    baseline = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Answer the question as concisely as possible with ONLY one answer without any other text:  {i['nq_question']}",
            }
        ],
        model="gpt-4o-mini"
    )

    # print(f"\t\tAnswer: \t {baseline.choices[0].message.content}")
    # print("\t\tGround Truth: \t", i['nq_answer'])

    curr = {
        "data_id": i["nq_id"],
        "ambig_question": i["nq_question"],
        "llm_response": baseline.choices[0].message.content,
        "ground_truth": i["nq_answer"],
    }

    out.append(curr)

    # store the output into sample_output_<current_unix_time>.json
    with open(f"Experiments/mini_sample_output_{current_unix_time}.json", "w") as f:
        json.dump(out, f)


# 1000 questions took 2 cents, 10 minutes and 40.48 seconds to complete