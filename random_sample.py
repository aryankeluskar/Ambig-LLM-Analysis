import os
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)

# go through sample.json and ask questions to gpt
import json
all_qs = json.load(open("filtered_train.json"))

# randomly sample 10 questions
import random
sample = random.sample(all_qs, 20)

print(sample)

output = []

for i in sample:
    # print("-"*20)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"rewrite this question replacing all questions with a what, but retain the meaning by specifying what entity or what person or what smallest timeframe (like day if possible, or month or year) the \"what\" answering. specify the year is 2018 is needed to answer a time-based question. {i['nq_question']}",
            }
        ],
        model="gpt-4o", temperature=0.2
    )

    print(chat_completion.choices[0].message.content)


    final_answer = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Answer the question as concisely as possible with ONLY one answer without any other text:  {chat_completion.choices[0].message.content}",
            }
        ],
        model="gpt-4o", temperature=0.2
    )

    output.append({
        "data_id": i["nq_id"],
        "ambig_question": i["nq_question"],
        "disambig_question": chat_completion.choices[0].message.content,
        "llm_response": final_answer.choices[0].message.content,
        "ground_truth": i["nq_answer"],
    })

import json
json.dump(output, open("random_sample_out.json", "w"))
