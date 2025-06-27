import pandas as pd

# Load the Parquet file
parquet_file = '/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/data/AIME24/aime_2024_problems.parquet'  # Replace with your Parquet file path
df = pd.read_parquet(parquet_file)

# Convert the DataFrame to a JSON file
json_file = '/cpfs04/user/liutianshuo/math/simpleRL-reason/eval/data/AIME24/AIME24.jsonl'  # Replace with your desired output JSON file path
df.to_json(json_file, orient='records', lines=True)

print(f"JSON file has been saved as {json_file}")