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
        "sa": """### System
    You're a helpful AI assistant. Analyze the sentiment of ### User's text and choose the appropriate sentiment. You can only choose the sentiment option in the 'Options' list. Take a look at the <example> below and think carefully before answering question of ### User.

    <example>
    its a totally average film with a few semi-alright action sequences that make the plot seem a little better and remind the viewer of the classic van dam films. parts of the plot don't make sense and seem to be added in to use up time. the end plot is that of a very basic type that doesn't leave the viewer guessing and any twists are obvious from the beginning. the end scene with the flask backs don't make sense as they are added in and seem to have little relevance to the history of van dam's character. not really worth watching again, bit disappointed in the end production, even though it is apparent it was shot on a low budget certain shots and sections in the film are of poor directed quality

    Options: ["0. Negative", "1. Positive", "2.Neutral"]

    Answer: 0. Negative

    """,
        "mr": """### System
    You're a helpful AI assistant. Make arithmetic reason or inference about the '### User' text and give the correct answer. The reason or inference must be reasonable, as shown in the <example> below. Take a look at the <example> below and think carefully before answering question of ### User.

    <example>
    Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

    Reasoning: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.

    Answer: 18

    """,
        "general": """You're a helpful AI assistant. Choose the correct answer for the question ### User or the appropriate answer that completes the ### User's sentence. You can only choose the option in the 'Options' list. If you are unsure of the correct answer, simply answer -1. You have to answer only the character or number. Below <example 1>, <example 2>, <example 3> and <example 4> are examples.

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


    <example 3>
    ### User
    its a totally average film with a few semi-alright action sequences that make the plot seem a little better and remind the viewer of the classic van dam films. parts of the plot don't make sense and seem to be added in to use up time. the end plot is that of a very basic type that doesn't leave the viewer guessing and any twists are obvious from the beginning. the end scene with the flask backs don't make sense as they are added in and seem to have little relevance to the history of van dam's character. not really worth watching again, bit disappointed in the end production, even though it is apparent it was shot on a low budget certain shots and sections in the film are of poor directed quality

    Options: ["0. Negative", "1. Positive", "2.Neutral"]

    ### Assistant
    0. Negative


    <example 4>
    ### User
    Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

    Reasoning: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.

    ### Assistant
    Answer: 18

    """}

    if prompt_label.lower() == "qa":
        prompt_blurb = prompts["qa"]
    elif prompt_label.lower() == "sc":
        prompt_blurb = prompts["sc"]
    elif prompt_label.lower() == "sa":
        prompt_blurb = prompts["sa"]
    elif prompt_label.lower() == "mr":
        prompt_blurb = prompts["mr"]
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
    "sa": """### System
You're a helpful AI assistant. Analyze the sentiment of ### User's text and choose the appropriate sentiment. You can only choose the sentiment option in the 'Options' list. Take a look at the <example> below and think carefully before answering question of ### User.

<example>
its a totally average film with a few semi-alright action sequences that make the plot seem a little better and remind the viewer of the classic van dam films. parts of the plot don't make sense and seem to be added in to use up time. the end plot is that of a very basic type that doesn't leave the viewer guessing and any twists are obvious from the beginning. the end scene with the flask backs don't make sense as they are added in and seem to have little relevance to the history of van dam's character. not really worth watching again, bit disappointed in the end production, even though it is apparent it was shot on a low budget certain shots and sections in the film are of poor directed quality

Options: ["0. Negative", "1. Positive", "2.Neutral"]

Answer: 0. Negative

""",
    "mr": """### System
You're a helpful AI assistant. Make arithmetic reason or inference about the '### User' text and give the correct answer. The reason or inference must be reasonable, as shown in the <example> below. Take a look at the <example> below and think carefully before answering question of ### User.

<example>
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Reasoning: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.

Answer: 18

""",
    "general": """You're a helpful AI assistant. Choose the correct answer for the question ### User or the appropriate answer that completes the ### User's sentence. You can only choose the option in the 'Options' list. If you are unsure of the correct answer, simply answer -1. You have to answer only the character or number. Below <example 1>, <example 2>, <example 3> and <example 4> are examples.

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


<example 3>
### User
its a totally average film with a few semi-alright action sequences that make the plot seem a little better and remind the viewer of the classic van dam films. parts of the plot don't make sense and seem to be added in to use up time. the end plot is that of a very basic type that doesn't leave the viewer guessing and any twists are obvious from the beginning. the end scene with the flask backs don't make sense as they are added in and seem to have little relevance to the history of van dam's character. not really worth watching again, bit disappointed in the end production, even though it is apparent it was shot on a low budget certain shots and sections in the film are of poor directed quality

Options: ["0. Negative", "1. Positive", "2.Neutral"]

### Assistant
0. Negative


<example 4>
### User
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Reasoning: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day. She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.

### Assistant
Answer: 18

"""}

    if prompt_label.lower() == "qa":
        prompt_blurb = prompts["qa"]
    elif prompt_label.lower() == "sc":
        prompt_blurb = prompts["sc"]
    elif prompt_label.lower() == "sa":
        prompt_blurb = prompts["sa"]
    elif prompt_label.lower() == "mr":
        prompt_blurb = prompts["mr"]
    else:
        prompt_blurb = prompts["general"]

    return prompt_blurb
