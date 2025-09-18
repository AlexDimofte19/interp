import json

import pandas as pd
import torch
import vllm


def generate_full_inputs(tokenizer, system_prompts, user_prompts):
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
        )
        for system_prompt in system_prompts
        for user_prompt in user_prompts
    ]


def generate_data(model_name_or_path: str, num_rollouts: int, max_new_tokens: int):
    """Generate a csv with prompts and responses on which we later train probes.

    We follow the setup from https://github.com/safety-research/persona_vectors/tree/main
    For now we only generate data for the humorous trait, found in `data/humorous.json`.
    """
    with open("data/humorous.json") as f:
        source_data = json.load(f)

    instruction_pairs = source_data["instruction"]
    positive_system_prompts = [pair["pos"] for pair in instruction_pairs]
    negative_system_prompts = [pair["neg"] for pair in instruction_pairs]
    questions = source_data["questions"]

    model = vllm.LLM(
        model=model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )

    sampling_params = vllm.SamplingParams(
        n=num_rollouts,
        max_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
    )

    tokenizer = model.get_tokenizer()

    positive_full_inputs = generate_full_inputs(tokenizer, positive_system_prompts, questions)
    negative_full_inputs = generate_full_inputs(tokenizer, negative_system_prompts, questions)

    print(f"Generated {len(positive_full_inputs)} positive full inputs")
    print(f"Generated {len(negative_full_inputs)} negative full inputs")

    positive_generations = model.generate(positive_full_inputs, sampling_params)
    positive_responses = [response.text for generation in positive_generations for response in generation.outputs]
    extended_pos_inputs = [pos_input for pos_input in positive_full_inputs for _ in range(num_rollouts)]

    negative_generations = model.generate(negative_full_inputs, sampling_params)
    negative_responses = [response.text for generation in negative_generations for response in generation.outputs]
    extended_neg_inputs = [neg_input for neg_input in negative_full_inputs for _ in range(num_rollouts)]

    print(f"Generated {len(positive_responses)} positive responses")
    print(f"Generated {len(negative_responses)} negative responses")

    # TODO: To follow https://arxiv.org/pdf/2507.21509 we need to add a judge LM here that would use data['eval_prompt'] to score the trait expression.
    # Then we would filter positive to be >50 and negative to be <50.

    data = pd.DataFrame(
        {
            "prompt": extended_pos_inputs + extended_neg_inputs,
            "response": positive_responses + negative_responses,
            "label": [1] * len(positive_responses) + [0] * len(negative_responses),
        }
    )

    data.to_csv("data/humorous.csv", index=False)
