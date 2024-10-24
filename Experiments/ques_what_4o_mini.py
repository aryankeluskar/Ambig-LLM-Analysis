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
# import random
# sample = random.sample(all_qs, 1000)

# load mini_sample_input_1729208141.json to maintain consistency
sample = json.load(open("Experiments/mini_sample_input_1729208141.json"))

# store the sample input into sample_input_<current_unix_time>.json
import time
current_unix_time = int(time.time())
with open(f"Experiments/mini_ques_what_input_{current_unix_time}.json", "w") as f:
    json.dump(sample, f)

sample = json.load(open("Experiments/mini_sample_input_1729208141.json"))

count = 0
out = []
for i in sample:
    count += 1
    print(f"Question {count}")

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Rewrite this question replacing all questions with a what, but retain the meaning by specifying what entity or what person or what timeframe the \"what\" answering. Also specify the current year is 2018 if needed to answer a time-based question. The Question: {i['nq_question']}",
            }
        ],
        model="gpt-4o-mini"
    )

    what_ques = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Answer the question as concisely as possible with ONLY one answer without any other text:  {chat_completion.choices[0].message.content}",
            }
        ],
        model="gpt-4o-mini"
    )

    # print(f"\t\tAnswer: \t {baseline.choices[0].message.content}")
    # print("\t\tGround Truth: \t", i['nq_answer'])

    curr = {
        "data_id": i["nq_id"],
        "ambig_question": i["nq_question"],
        "disambig_question": chat_completion.choices[0].message.content,
        "llm_response": what_ques.choices[0].message.content,
        "ground_truth": i["nq_answer"],
    }

    out.append(curr)

    # store the output into sample_output_<current_unix_time>.json
    with open(f"Experiments/mini_ques_what_output_{current_unix_time}.json", "w") as f:
        json.dump(out, f)


# 1000 questions took 3 cents, 22 minutes and 51.23 seconds to complete