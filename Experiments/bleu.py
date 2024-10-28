import evaluate 
import json

baseline = json.loads(open('Experiments/mini_sample_output_1729208141.json').read())
baseline = baseline[:10]
b_references = []
b_predictions = []

file_to_compare = json.loads(open('Experiments/mini_ques_what_output_1729734928.json').read())
file_to_compare = file_to_compare[:10]
f_references = []
f_predictions = []

for i in baseline:
    b_references.append([i["ground_truth"]])
    b_predictions.append(i["llm_response"])

# print(b_predictions)
# print(b_references)

for i in file_to_compare:
    f_references.append([i["ground_truth"]])
    f_predictions.append(i["llm_response"])

# print(f_predictions)
# print(f_references)

bleu = evaluate.load("bleu")

b_results = bleu.compute(predictions=b_predictions, references=b_references, max_order=1)
f_results = bleu.compute(predictions=f_predictions, references=f_references, max_order=1)

print(b_results)
print(f_results)

# predictions = ["hello there general kenobi", "foo foobar"]
# references = [
#     ["hello there general kenobi"],
#     ["foo bar foobar"]
# ]
# bleu = evaluate.load("bleu")
# results = bleu.compute(predictions=predictions, references=references)
# print(results["bleu"])