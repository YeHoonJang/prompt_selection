import os
import openai
import random
import time

from tqdm import tqdm
import pandas as pd

from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv("env/settings.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

dataset = load_dataset("Yehoon/arc_hella", split="train")
rand_idx = random.sample([i for i in range(dataset.shape[0])], int(len(dataset)*0.01))

result = {"prompt": [], "label": [], "gpt_label": []}
for idx in tqdm(rand_idx, desc="Labeling ..."):
    question = dataset[idx]["question"]
    options = dataset[idx]["options"]
    answer = dataset[idx]["answer"]

    prompt = f"""You're a helpful task classification assistant.
Identify the task type for the given <Task> below.
You can only choose the correct answer from the list [qa, sc].

<Task Description ans example>
* qa: a task that answer the correct answer to a question.
```
### User
Which archaeological method is best for identifying large-scale land modifications?

Options: ['0. test pits', '1. ground-penetrating radar', '2. aerial photography', '3. proton magnetometers']

### Assistant
2
```

* sc: a task that chooses the appropriate and logically following phrase to finish a given unfinished sentence including arithmetic inference.
```
### User
Older adults may show less major depression than younger adults because they

Options: ['0. Have learned to cope with loss', '1. See major depression as a stigma', '2. Have very different brain biochemistry', '3. Are no longer concerned with emotional matters']

### Assistant
0
```
You should answer the task type and not say anything after that.

<Task>
{question}

Options: {options}

{answer}"""

    message = [{"role": "assistant", "content": prompt}]
    # gpt3_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=message
    # )
    gpt4_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message
    )

    result["prompt"].append(prompt)
    result["label"].append(dataset[idx]["label"])
    result["gpt_label"].append(gpt4_response.choices[0]["message"]["content"])
    time.sleep(3)

df = pd.DataFrame(result)
df.to_csv("data/prompt_label.csv", index=False)



