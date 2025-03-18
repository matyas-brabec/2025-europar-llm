import os
import sys

def process_file(file_path):
    """Process a .cu file to modify extern "C" declarations."""
    print(f"Processing {file_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Flag to track if any changes were made
    changes_made = False
    
    # New content
    new_lines = []
    
    for line in lines:
        if 'extern "C"' in line:
            changes_made = True
            # Add commented original line
            new_lines.append(f"// original GPT: {line}")
            
            # Calculate whitespace before 'extern'
            whitespace = line[:line.find('extern')]
            # Create new line without extern "C" but preserve function signature and indentation
            modified_content = line.replace('extern "C"', '').strip()
            modified_line = whitespace + modified_content + '\n'
            new_lines.append(modified_line)
        else:
            new_lines.append(line)
    
    # Write back if changes were made
    if changes_made:
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        print(f"Updated {file_path}")
    else:
        print(f"No changes needed in {file_path}")

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
