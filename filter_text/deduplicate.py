import os

def deduplicate_file(file_path):
    """
    Removes trailing whitespace from each line and deduplicates identical lines.
    
    Args:
        file_path: Path to the text file to process
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False
    
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Strip trailing whitespace and create a set to remove duplicates
        unique_lines = set()
        for line in lines:
            stripped_line = line.rstrip()
            if stripped_line:  # Skip empty lines
                unique_lines.add(stripped_line)
        
        # Write the deduplicated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(unique_lines))
        
        print(f"Successfully processed '{file_path}':")
        print(f"  - Removed trailing whitespace")
        print(f"  - Removed duplicate lines")
        print(f"  - Original line count: {len(lines)}")
        print(f"  - New line count: {len(unique_lines)}")
        
        return True
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

if __name__ == "__main__":
    file_path = "filter_text/text_1.txt"
    deduplicate_file(file_path)
