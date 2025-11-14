#!/usr/bin/env python3

import json
import re
import time
from pathlib import Path
from typing import Any, Generator

from colorama import Fore, Style
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from munch import munchify


MODEL = "gpt-5.1-2025-11-13"
REASONING_EFFORT = "high"
VERBOSITY = "medium"

# Source: https://platform.openai.com/docs/models/gpt-5.1
COST_INPUT = 1.25 / 1_000_000
COST_OUTPUT = 10 / 1_000_000


load_dotenv(find_dotenv(), override=True)  # take environment variables from .env
client = OpenAI()


def log_request(request: dict[str, Any], output_folder: Path) -> None:
    request_log = json.dumps(request, indent=2)
    print(Fore.MAGENTA, end="")
    print("Request:")
    print(request_log)
    print(Style.RESET_ALL)
    (output_folder / "request.json").write_text(request_log, encoding="utf-8")


def log_response(response: ChatCompletion, output_folder: Path) -> None:
    response_log = response.model_dump_json(indent=2)
    print(Fore.CYAN, end="")
    print("Response:")
    print(response_log)
    print(Style.RESET_ALL)
    (output_folder / "response.json").write_text(response_log, encoding="utf-8")


def print_llm_output(response_text: str, output_folder: Path) -> None:
    print(Fore.GREEN, end="")
    print("LLM output:")
    print(response_text)
    print(Style.RESET_ALL)
    (output_folder / "response.md").write_text(response_text, encoding="utf-8")


def load_file_with_includes(filepath: Path) -> Generator[str, None, None]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("@include"):
                included_file = line.split()[1]
                included_file_path = filepath.parent / included_file
                yield from load_file_with_includes(included_file_path)
            else:
                yield line


def load_messages(system_filename: Path, user_filename: Path) -> list[dict[str, str]]:
    system_message = "".join(load_file_with_includes(system_filename))
    user_message = "".join(load_file_with_includes(user_filename))

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return messages


def extract_code_block(response_text: str) -> str | None:
    # This regex looks for content between triple backticks, possibly with a language specifier.
    pattern = r"```(?:\w*\n)?(.*?)```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None


def estimate_cost(response: ChatCompletion, output_folder: Path, time_taken: float | None = None, batch: bool = False) -> None:
    usage = response.usage

    if usage is None:
        print("No usage information available to estimate cost.")
        return

    prompt = usage.prompt_tokens
    completion = usage.completion_tokens

    prompt_tokens_details = usage.prompt_tokens_details
    completion_tokens_details = usage.completion_tokens_details

    prompt_cached = prompt_tokens_details.cached_tokens if prompt_tokens_details is not None else 0
    reasoning = completion_tokens_details.reasoning_tokens if completion_tokens_details is not None else 0


    prompt_uncached = prompt - prompt_cached if prompt_cached is not None else prompt
    output = completion - reasoning if reasoning is not None else completion

    prompt_uncached_cost = prompt_uncached * COST_INPUT
    prompt_cached_cost = prompt_cached * COST_INPUT / 10 if prompt_cached is not None else 0
    cost_output = output * COST_OUTPUT
    cost_reasoning = reasoning * COST_OUTPUT if reasoning is not None else 0

    total_cost = prompt_uncached_cost + prompt_cached_cost + cost_output + cost_reasoning

    costs: dict[str, Any] = {
        "prompt_uncached_tokens": prompt_uncached,
        "prompt_uncached_cost": prompt_uncached_cost if not batch else prompt_uncached_cost / 2,
        "prompt_cached_tokens": prompt_cached,
        "prompt_cached_cost": prompt_cached_cost if not batch else prompt_cached_cost / 2,
        "output_tokens": output,
        "cost_output": cost_output if not batch else cost_output / 2,
        "reasoning_tokens": reasoning,
        "cost_reasoning": cost_reasoning if not batch else cost_reasoning / 2,
        "total_cost": total_cost if not batch else total_cost / 2,
    }

    if time_taken:
        costs["time_taken_in_seconds"] = time_taken

    costs_log = json.dumps(costs, indent=2)
    print(Fore.YELLOW, end="")
    print("Estimated cost (USD):")
    print(costs_log)
    print(Style.RESET_ALL)
    (output_folder / "costs.json").write_text(costs_log, encoding="utf-8")


def create_request(system_filename: Path, user_filename: Path) -> dict[str, Any]:
    return {
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "verbosity": VERBOSITY,
        "messages": load_messages(system_filename, user_filename),
    }


def prepare_batch_line(system_filename: Path, user_filename: Path, repetition: int) -> dict[str, Any]:
    name = user_filename.stem
    if name.endswith(".prompt"):
        name = name[:-7]
    return {
        "custom_id": f"{name}-{repetition:02d}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": create_request(system_filename, user_filename),
    }


def run_chat(system_filename: Path, user_filename: Path, output_folder: Path) -> None:
    request = create_request(system_filename, user_filename)
    log_request(request, output_folder)

    start_time = time.time()
    response = client.chat.completions.create(**request)
    end_time = time.time()
    print(f"Request took {end_time - start_time:.2f} seconds.")

    process_response(response, output_folder, end_time - start_time)


def extract_answer(response: ChatCompletion) -> str:
    content = response.choices[0].message.content
    return content if content else ""


def save_code_block(response_text: str, output_folder: Path) -> None:
    code_block = extract_code_block(response_text)

    if code_block:
        print("Saving extracted code block.")
        (output_folder / "code.cu").write_text(code_block, encoding="utf-8")
    else:
        print("No code block found in response. Saving full response instead.")
        (output_folder / "code.cu").write_text(response_text, encoding="utf-8")


def process_response(response: ChatCompletion, output_folder: Path, time_taken: float | None = None) -> None:
    log_response(response, output_folder)

    response_text = extract_answer(response)
    print_llm_output(response_text, output_folder)

    save_code_block(response_text, output_folder)

    estimate_cost(response, output_folder, time_taken)


def prepare_batch_file(system_filename: Path, user_filenames: list[Path], output_filename: Path, repetitions: int = 5, start: int = 1) -> None:
    with open(output_filename, "w", encoding="utf-8") as f:
        for prompt in user_filenames:
            for repetition in range(start, start + repetitions):
                line = prepare_batch_line(system_filename, prompt, repetition)
                json.dump(line, f)
                f.write("\n")


def parse_batch_output(output_filename: Path, results_folder: Path) -> None:
    with open(output_filename, "r", encoding="utf-8") as f:
        for line in f:
            response = json.loads(line)
            custom_id = response["custom_id"]
            body = response["response"]["body"]
            llm_response = munchify(body)

            output_folder = results_folder / custom_id
            output_folder.mkdir(exist_ok=True, parents=True)

            response_text = extract_answer(llm_response)
            save_code_block(response_text, output_folder)
            estimate_cost(llm_response, output_folder, batch=True)


# Example
# folder = Path(__file__).parent / "examples"
# run_chat(folder / "system.md", folder / "user.md", folder)

def main():
    # Folder configuration
    prompts_folder = Path("../prompts")
    system_prompt = prompts_folder / "_system.md"
    results_folder = Path("../results")

    # Load Histogram assignment prompts
    histogram_prompts = list(sorted(prompts_folder.glob("histogram*.prompt.md")))
    # print(histogram_prompts)

    # Load Game of Life assignment prompts
    gol_prompts = list(sorted(prompts_folder.glob("game_of_life*.prompt.md")))
    # print(gol_prompts)

    # Load KNN assignment prompts
    knn_prompts = list(sorted(prompts_folder.glob("knn*.prompt.md")))
    # print(knn_prompts)

    # Generate batch files with LLM queries (uncomment desired line)
    prepare_batch_file(system_prompt, histogram_prompts, results_folder / "histogram-request.jsonl", repetitions=10)
    prepare_batch_file(system_prompt, gol_prompts, results_folder / "gol-request.jsonl", repetitions=10)
    prepare_batch_file(system_prompt, knn_prompts, results_folder / "knn-request.jsonl", repetitions=10)

    # Process batch LLM responses (uncomment desired line)
    # parse_batch_output(results_folder / "histogram-response.jsonl", results_folder / "histogram")
    # parse_batch_output(results_folder / "gol-response.jsonl", results_folder / "gol")
    # parse_batch_output(results_folder / "knn-response.jsonl", results_folder / "knn")


    # Select assignment for local execution (uncomment desired line)
    # prompts = histogram_prompts
    # prompts = gol_prompts
    # prompts = knn_prompts
    prompts: list[Path] = []

    # Run selected prompts locally
    for prompt in prompts:
        print("Running prompt:", prompt)
        name = prompt.stem
        if name.endswith(".prompt"):
            name = name[:-7]
        output_folder = results_folder / name
        output_folder.mkdir(exist_ok=True, parents=True)
        run_chat(system_prompt, prompt, output_folder)


if __name__ == "__main__":
    main()
