# General & Task Description Version
python /home/yehoon/workspace/promt_selection/inference.py --inference --model="meta-llama/Llama-2-7b-hf" --dataset="Yehoon/arc_hella" --test_split="train" --prompt_style="upstage" --custom_prompt="/home/yehoon/workspace/promt_selection/data/prompt/task_descript_prompt.txt" --output_dir="/home/yehoon/workspace/outputs/" --output_name="0907_ft_llama2_upstage_way" --generated_name="/home/yehoon/workspace/promt_selection/outputs/task_description_prompt.csv"


# Prompt Select Version
python /home/yehoon/workspace/promt_selection/inference.py --inference --select_prompt --model="meta-llama/Llama-2-7b-hf" --dataset="Yehoon/arc_hella" --test_split="train" --prompt_style="upstage" --custom_prompt="/home/yehoon/workspace/promt_selection/data/prompt/empty.txt" --output_dir="/home/yehoon/workspace/outputs/" --output_name="0907_ft_llama2_upstage_way" --generated_name="/home/yehoon/workspace/promt_selection/outputs/...name!!"