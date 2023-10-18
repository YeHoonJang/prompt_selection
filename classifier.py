import os
import openai
import time

from dotenv import load_dotenv

load_dotenv("env/settings.env")
openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt_classifier(data):
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
    {data}"""

    message = [{"role": "assistant", "content": prompt}]
    # gpt3_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=message
    # )
    gpt4_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=message
    )
    time.sleep(3)
    return gpt4_response.choices[0]["message"]["content"]
