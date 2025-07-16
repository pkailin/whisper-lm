import sys

def calculate_match_percentage(file_path):
    """Calculate percentage of lines where predicted == actual"""
    total_lines = 0
    matches = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('<DIV>')
                if len(parts) != 3:
                    print(f"Skipping line {line_num}: invalid format")
                    continue
                
                predicted = parts[1].strip()
                actual = parts[2].strip()
                
                total_lines += 1
                if predicted == actual:
                    matches += 1
        
        if total_lines == 0:
            print("No valid lines found")
            return
        
        percentage = (matches / total_lines) * 100
        print(f"Matches: {matches}/{total_lines} ({percentage:.2f}%)")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    
    calculate_match_percentage(sys.argv[1])
