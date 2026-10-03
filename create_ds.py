import os
import glob

def build_cpp_corpus(source_directory: str, output_filepath: str):
    print(f"\nScanning {source_directory} for C++ files...")
    
    # Added .h and .cc to ensure all C++ variations are caught
    cpp_files = glob.glob(f"{source_directory}/**/*.cpp", recursive=True)
    cpp_files += glob.glob(f"{source_directory}/**/*.hpp", recursive=True)
    cpp_files += glob.glob(f"{source_directory}/**/*.h", recursive=True)
    cpp_files += glob.glob(f"{source_directory}/**/*.cc", recursive=True)
    
    total_files = 0
    total_bytes = 0
    
    # 'a' mode appends to the file, preserving data from previous loop iterations
    with open(output_filepath, 'a', encoding='utf-8') as outfile:
        for filepath in cpp_files:
            
            if not os.path.isfile(filepath):
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                # Filter 1: Check length
                if len(content) < 50 or len(content) > 15000:
                    continue
                    
                # Filter 2: ASCII only
                if not content.isascii():
                    content = content.encode('ascii', 'ignore').decode()
                
                # Write content and context separator
                outfile.write(content)
                outfile.write("\n\n// <|endoftext|>\n\n")
                
                total_files += 1
                total_bytes += len(content)
                
            except UnicodeDecodeError:
                continue

    print(f"Directory {source_directory} completed!")
    print(f"Files merged: {total_files}")
    print(f"Size added: {total_bytes / (1024 * 1024):.2f} MB")

# Execution
if __name__ == "__main__":
    output_file = "./data/cpp_corpus/raw_code.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Automatically finds all items in the directory that start with "new"
    source_dirs = glob.glob("./cpp_folder*") # to account for multiple folder like cpp_folder1, cpp_folder2 ..
    
    for source_dir in source_dirs:
        if os.path.isdir(source_dir):
            build_cpp_corpus(source_dir, output_file)
            
    print(f"\nAll folders successfully merged into {output_file}")