import boto3
from botocore.exceptions import ClientError
import json

# go through sample.json and ask questions to gpt
import json
all_qs = json.load(open("data/filtered_train.json"))

# randomly sample 1000 questions
# import random
# sample = random.sample(all_qs, 10)

# load mini_sample_input_1729208141.json to maintain consistency
sample = json.load(open("Experiments/mini_sample_input_1729208141.json"))

# store the sample input into sample_input_<current_unix_time>.json
import time
current_unix_time = int(time.time())
with open(f"Experiments/mistral_ques_what_input_{current_unix_time}.json", "w") as f:
    json.dump(sample, f)

def call_mistral(user_message):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "mistral.mixtral-8x7b-instruct-v0:1"

    body = json.dumps({
        "prompt": user_message,
    })

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=body,
        )

        response_body = json.loads(response.get('body').read())
        return response_body['outputs'][0]['text'].strip()

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        e = str(e)
        if "ThrottlingException" in e:
            print("ThrottlingException")
            time.sleep(2)
            return call_mistral(user_message)

count = 0
out = []
for i in sample:
    count += 1
    print(f"Question {count}")

    prompt = f"Rewrite this question replacing all questions with a what, but retain the meaning by specifying what entity or what person or what timeframe the \"what\" answering. Also specify the current year is 2018 if needed to answer a time-based question. The Question: {i['nq_question']}"
    response = call_mistral(prompt)
    # remove "Anwer: " from the output
    ques_what = response.replace("Answer: ", "")    

    content = f"Provide only the answer as concisely as possible, nothing else: {formatted}"
    response = call_mistral(content)
    formatted = response.replace("Answer: ", "")

    curr = {
        "data_id": i["nq_id"],
        "ambig_question": i["nq_question"],
        "disambig_question": ques_what,
        "llm_response": response,
        "formatted_response": formatted,
        "ground_truth": i["nq_answer"],
    }

    out.append(curr)

    # store the output into sample_output_<current_unix_time>.json
    with open(f"Experiments/mistral_ques_what_output_{current_unix_time}.json", "w") as f:
        json.dump(out, f)

# 1000 questions took 0 cents, XX to complete on AWS Bedrock