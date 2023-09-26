from datasets import load_dataset, Dataset
import pandas as pd


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

arc_df = pd.DataFrame({"question": arc["question"], "options": arc["options"], "answer": arc["answerKey"], "label":"qa"})
hella_df = pd.DataFrame({"question": hellaswag["ctx"], "options": hellaswag["options"], "answer": hellaswag["label"], "label":"sc"})

df = pd.concat([arc_df, hella_df])
df = df.sample(frac=1, random_state=42)

dataset = Dataset.from_pandas(df)
dataset = dataset.remove_columns(["__index_level_0__"])

dataset.push_to_hub("Yehoon/arc_hella")
