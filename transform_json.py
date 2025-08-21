import json

def transform_json(input_file, output_file):
    """
    Transform the JSON file by adding response content to the input array as an assistant role.
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Transform each entry
    for entry in data:
        if 'input' in entry and 'response' in entry:
            # Get the response content
            response_content = entry['response'][0]['content'] if entry['response'] else ""
            
            # Create the assistant role entry
            assistant_entry = {
                "role": "assistant",
                "content": response_content
            }
            
            # Add the assistant role to the input array
            entry['input'].append(assistant_entry)
    
    # Write the transformed data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Transformation completed! Output saved to {output_file}")
    print(f"Total entries processed: {len(data)}")

if __name__ == "__main__":
    input_file = "eval/qwen3_30b_a3b_deepscaler--0602/ppo/2/deepscaler/merge_filtered_new.json"
    output_file = "merge_filtered_new_transformed.json"
    
    try:
        transform_json(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found!")
    except Exception as e:
        print(f"Error during transformation: {e}") 