from classifier import gpt_classifier

# TODO load prompt by .txt path
def select_prompt(opt, instruction, input, response):
    data = "\n\n".join([instruction, input, response])
    prompt_label = gpt_classifier(data)
    prompts = {
        "qa": """You're a helpful AI assistant. Choose the correct answer for the question ### User or the appropriate answer that completes the ### User's sentence. You can only choose the option in the 'Options' list. If you are unsure of the correct answer, simply answer -1. You have to answer only the character or number. Below <example> is an example.

<example>
### User
George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?

Options: ["A. wet palms", "B. dry palms", "C. palms covered with oil", "D. palms covered with lotion"]

### Assistant
B

""",
    "sc": """### System
You're a helpful AI assistant. Complete the 'Description' with an appropriate ending. You can only choose the option in the 'Options' list. Take a look at the <example> below and think carefully before answering question of ### User.

<example>
Description: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. the

Options: [", the man adds wax to the windshield and cuts it.", ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.", ", the man puts on a christmas coat, knitted with netting.", ", the man continues removing the snow on his car."]

Answer: ", the man continues removing the snow on his car."

""",
    "general": """You're a helpful AI assistant. Choose the correct answer for the question ### User or the appropriate answer that completes the ### User's sentence. You can only choose the option in the 'Options' list. If you are unsure of the correct answer, simply answer -1. You have to answer only the character or number. Below <example 1> and <example 2> are examples.

<example 1>
### User
Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then

Options: [", the man adds wax to the windshield and cuts it.", ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.", ", the man puts on a christmas coat, knitted with netting.", ", the man continues removing the snow on his car."]

### Assistant
", the man continues removing the snow on his car."


<example 2>
### User
George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?

Options: ["A. wet palms", "B. dry palms", "C. palms covered with oil", "D. palms covered with lotion"]

### Assistant
B

"""}

    if prompt_label.lower() == "qa":
        prompt_blurb = prompts["qa"]
    elif prompt_label.lower() == "sc":
        prompt_blurb = prompts["sc"]
    else:
        prompt_blurb = prompts["general"]

    return prompt_blurb


def select_prompt_using_label(opt, label):
    prompt_label = label
    prompts = {
        "qa": """You're a helpful AI assistant. Choose the correct answer for the question ### User or the appropriate answer that completes the ### User's sentence. You can only choose the option in the 'Options' list. If you are unsure of the correct answer, simply answer -1. You have to answer only the character or number. Below <example> is an example.

<example>
### User
George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?

Options: ["A. wet palms", "B. dry palms", "C. palms covered with oil", "D. palms covered with lotion"]

### Assistant
B

""",
    "sc": """### System
You're a helpful AI assistant. Complete the 'Description' with an appropriate ending. You can only choose the option in the 'Options' list. Take a look at the <example> below and think carefully before answering question of ### User.

<example>
Description: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. the

Options: [", the man adds wax to the windshield and cuts it.", ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.", ", the man puts on a christmas coat, knitted with netting.", ", the man continues removing the snow on his car."]

Answer: ", the man continues removing the snow on his car."

""",
    "general": """You're a helpful AI assistant. Choose the correct answer for the question ### User or the appropriate answer that completes the ### User's sentence. You can only choose the option in the 'Options' list. If you are unsure of the correct answer, simply answer -1. You have to answer only the character or number. Below <example 1> and <example 2> are examples.

<example 1>
### User
Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then

Options: [", the man adds wax to the windshield and cuts it.", ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.", ", the man puts on a christmas coat, knitted with netting.", ", the man continues removing the snow on his car."]

### Assistant
", the man continues removing the snow on his car."


<example 2>
### User
George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?

Options: ["A. wet palms", "B. dry palms", "C. palms covered with oil", "D. palms covered with lotion"]

### Assistant
B

"""}

    if prompt_label.lower() == "qa":
        prompt_blurb = prompts["qa"]
    elif prompt_label.lower() == "sc":
        prompt_blurb = prompts["sc"]
    else:
        prompt_blurb = prompts["general"]

    return prompt_blurb
