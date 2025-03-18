import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

OUT_DIR = SCRIPT_DIR / "generated" / "gpt-code"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_FILE = SCRIPT_DIR / ".." / ".."/ "results" / "gol-response.jsonl"

bool_prompt_ids = ['01']

int64_func = """

// EDITOR NOTE: Code below was not generated by LLM. It was added to ensure correct compilation for testing purposes.
#include <cstdint>
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    (void) input;
    (void) output;
    (void) grid_dimensions;
}
"""

bool_func = """

// EDITOR NOTE: Code below was not generated by LLM. It was added to ensure correct compilation for testing purposes.
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    (void) input;
    (void) output;
    (void) grid_dimensions;
}
"""

def decorate_code(code, prompt_id):
    if prompt_id in bool_prompt_ids:
        code += int64_func
    else:
        code += bool_func

    code = code.replace('extern "C"', "")
    return code


def handle_one_json_response(json_raw_str):
    obj = json.loads(json_raw_str)

    full_id = obj['custom_id'].replace("game_of_life", "")
    prompt_id = full_id.split("-")[0]
    attempt = full_id.split("-")[1]

    code = obj['response']['body']['choices'][0]['message']['content']

    print(f"{full_id} -> {prompt_id} ... {attempt}")

    PROMPT_DIR = OUT_DIR / prompt_id
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)

    out_file = PROMPT_DIR / f"{attempt}" / "gol.cu"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    code = decorate_code(code, prompt_id)

    with open(out_file, "w") as f:
        f.write(code)


in_contents = IN_FILE.read_text()
jsons = [x for x in in_contents.split("\n") if x]

for record in jsons:
    handle_one_json_response(record)
