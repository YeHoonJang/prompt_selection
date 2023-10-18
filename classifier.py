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

def gpt_classifier()


question = dataset[idx]["question"]
options = f'Options: {dataset[idx]["options"]}'
answer = dataset[idx]["answer"]

prompt = f"""You're a helpful task classification assistant.
Identify the task type for the given <Task> below.
You can only choose the correct answer from the list [qa, sc].

<Task Description and example>
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

{options}

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

# Save as csv
df = pd.DataFrame(result)
df["question"] = df["prompt"].apply(lambda x: x.split("<Task>\n")[-1].split("Options:")[0].strip())
df["options"] = df["prompt"].apply(lambda x: x.split("<Task>\n")[-1].split("Options:")[-1].split("\n")[0].strip())
df["answer"] = df["prompt"].apply(lambda x: x.split("<Task>\n")[-1].split("Options:")[-1].split("\n")[-1].strip())
# df.to_csv("/home/yehoon/workspace/promt_selection/data/result/prompt_test.csv", index=False)

# # Push to hub
# dataset = load_dataset("csv", data_files="/home/yehoon/workspace/promt_selection/data/result/prompt_test.csv")
# dataset.push_to_hub("Yehoon/arc_hella_test")
