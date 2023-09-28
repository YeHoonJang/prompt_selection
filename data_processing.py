import pandas as pd

from datasets import load_dataset, Dataset
from functools import partial
from transformers import AutoTokenizer

from utils.utils import preprocess_batch


def data_combine():
    arc = load_dataset("ai2_arc", name="ARC-Easy", split="test")
    hellaswag = load_dataset("hellaswag", split="validation")

    arc = arc.map(
        lambda sample: {
            **sample,
            "options": [f"{label}. {text}" for label, text in zip(sample["choices"]["label"], sample["choices"]["text"])]
        }
    )
    hellaswag = hellaswag.map(
        lambda sample: {
            **sample,
            "options": [f"{idx}. {text}" for idx, text in enumerate(sample["endings"])]
        }
    )

    arc_df = pd.DataFrame({"question": arc["question"], "options": arc["options"], "answer": arc["answerKey"], "label": "qa"})
    hella_df = pd.DataFrame({"question": hellaswag["ctx"], "options": hellaswag["options"], "answer": hellaswag["label"], "label": "sc"})

    df = pd.concat([arc_df, hella_df])
    df = df.sample(frac=1, random_state=42)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.remove_columns(["__index_level_0__"])

    dataset.push_to_hub("Yehoon/arc_hella")


# Pre-processing Dataset
def create_prompt_formats(opt, sample):
    if opt.prompt_style == "upstage":
        INSTRUCTION_KEY = "### User"
        INPUT_KEY = "### System"
        RESPONSE_KEY = "### Assistant"
        END_KEY = ""

    else:
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"

    # Custom Prompt
    if opt.custom_prompt == "none":
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    else:
        INTRO_BLURB = opt.intro_blurb

    # Prompt format by dataset
    if "dolly" in opt.dataset.lower():
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
        input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
        response = f"{RESPONSE_KEY}\n{sample['response']}"

    elif "scienceqa" in opt.dataset.lower():
        choice_prefixes = [chr(ord('A') + i) for i in range(26)]

        def format_options(options, choice_prefixes):
            return ' '.join([f"({c}) {o}" for c, o in zip(choice_prefixes, options)])

        options = format_options(sample["choices"], choice_prefixes)
        answer = choice_prefixes[sample["answer"]]

        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\n\nOptions:\n{options}"
        input_context = f"{INPUT_KEY}\n{sample['hint']}" if sample["hint"] else None
        response = f"{RESPONSE_KEY}\n{answer}"

    # arc dataset
    elif "ai2" in opt.dataset.lower():
        options = [f"{label}. {text}" for label, text in zip(sample["choices"]["label"], sample["choices"]["text"])]

        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\nOptions: {str(options)}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['answerKey']}"

    elif "hellaswag" in opt.dataset.lower():
        options = [f"{idx}. {text}" for idx, text in enumerate(sample["endings"])]

        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['ctx']}\nOptions: {str(options)}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['label']}"

    elif "mmlu" in opt.dataset.lower():
        options = [f"{idx}. {text}" for idx, text in enumerate(sample["choices"])]

        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\nOptions: {str(options)}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['answer']}"

    elif "truthful_qa" in opt.dataset.lower():
        options = [f"{idx}. {text}" for idx, text in enumerate(sample["mc1_targets"]["choices"])]

        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\nOptions: {str(options)}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['mc1_targets']['labels'].index(1)}"

    elif "arc_hella" in opt.dataset.lower():
        blurb = f"{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}\n{sample['question']}\n\nOptions: {sample['options']}"
        input_context = ""
        response = f"{RESPONSE_KEY}\n{sample['answer']}"

    inference_response = f"{RESPONSE_KEY}\n"
    end = f"{END_KEY}"

    if opt.train:
        parts = [part for part in [blurb, instruction, input_context, response, end] if part]
    elif opt.inference:
        parts = [part for part in [blurb, instruction, input_context, inference_response] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample


def preprocess_dataset(opt, tokenizer: AutoTokenizer, max_length: int, seed, dataset):

    # Add prompt to each sample
    print("Preprocessing dataset...")

    _create_prompt_formats = partial(create_prompt_formats, opt)

    print(f"Creating prompts for {(opt.dataset).split('/')[-1]} dataset...")
    dataset = dataset.map(_create_prompt_formats)  # , batched=True)
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)

    if opt.train:
        if "dolly" in opt.dataset.lower():
            if opt.llama_ft:
                remove_columns = ["instruction", "context", "response", "category"]
            else:
                remove_columns = ["instruction", "context", "response", "text", "category"]
            dataset = dataset.map(
                _preprocessing_function,
                batched=True,
                remove_columns=remove_columns,
            )
        elif "scienceqa" in opt.dataset.lower():
            if opt.llama_ft:
                remove_columns = ['image', 'question', 'choices', 'answer', 'hint', 'task', 'grade', 'subject', 'topic', 'category', 'skill', 'lecture', 'solution']
            else:
                remove_columns = ["text", 'image', 'question', 'choices', 'answer', 'hint', 'task', 'grade', 'subject', 'topic', 'category', 'skill', 'lecture', 'solution']
            dataset = dataset.map(
                _preprocessing_function,
                batched=True,
                remove_columns=remove_columns
            )

        # Filter out samples that have input_ids exceeding max_length
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    return dataset
