import os
import sys

def get_footer_for(prompt_id):
    bools = ['01']
    tiled = ['02_tiled']
    # rest is rowed

    if prompt_id in bools:
        return '''
#include <cstdint>
void run_game_of_life(const std::uint64_t* input, std::uint64_t* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}
// this label is used to identify the memory layout of the input and output arrays by the framework
// MEMORY_LAYOUT: BOOLS
'''

    footer = '''
void run_game_of_life(const bool* input, bool* output, int grid_dimensions) {
    (void)input;
    (void)output;
    (void)grid_dimensions;
}

// This label is used by the framework to identify the memory layout of the input and output arrays.
// MEMORY_LAYOUT: <layout>
'''

    return footer.replace('<layout>', 'TILES' if prompt_id in tiled else 'ROWS')

def process_file(file_path):
    print(f"Processing {file_path}")

    dir_of_the_file = os.path.dirname(file_path)
    dir_of_the_file = os.path.basename(dir_of_the_file)

    prompt_id = dir_of_the_file[len('game_of_life'):].split('-')[0]
    attempt_id = dir_of_the_file[len('game_of_life'):].split('-')[1]
    
    with open(file_path, 'r') as f:
        content = f.read()

    footer = get_footer_for(prompt_id)

    content += '\n\n// The content below was not generated by GPT; it was added to ensure the framework can compile the code.\n\n'
    content += footer    

    with open(file_path, 'w') as f:
        f.write(content)    

def main():
    if len(sys.argv) != 2:
        print("Usage: python remove-extern.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    # Find all .cu files
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.cu'):
                process_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
