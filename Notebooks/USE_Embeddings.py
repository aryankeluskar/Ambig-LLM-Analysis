# import os
# from openai import OpenAI
# import dotenv
# dotenv.load_dotenv()

# client = OpenAI(
#   api_key=os.environ.get("OPENAI_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "",
#         }
#     ],
#     model="gpt-4o-mini",
    
# )

# str(chat_completion.choices[0].message.content)

from absl import logging
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def get_embedding(text):
  return model(text)

sentence1 = ["I love Coca Cola"]
sentence = ["I love Coke"]

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

# print(get_embedding(word))
# print(get_embedding(sentence))

def get_distance(text1, text2):
    for i in range(len(text1)):
        text1[i] = str(text1[i]).lower()
    for i in range(len(text2)):
        text2[i] = str(text2[i]).lower()
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return np.inner(embedding1, embedding2) #/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))

print(get_distance(sentence1, sentence))


import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

import dotenv
dotenv.load_dotenv()

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)

def gpt_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_distance(text1, text2, model="text-embedding-3-small"):
   text1 = str(text1).lower()
   text2 = str(text2).lower()
   embedding1 = gpt_embedding(text1, model)
   embedding2 = gpt_embedding(text2, model)
   return cosine_similarity([embedding1], [embedding2])[0][0]

print(get_distance(sentence1[0], sentence[0]))