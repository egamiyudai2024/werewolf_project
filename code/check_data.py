#check_data.py
import pickle
import os
import config

print("--- Discussion Data Check ---")

# チェック対象のイテレーション番号 (0から始まる)
ITERATION_TO_CHECK = 0

data_filename = f"discussion_data_iter_{ITERATION_TO_CHECK}.pkl"

if not os.path.exists(data_filename):
    print(f"ERROR: Data file '{data_filename}' not found.")
    print("Please run the main script first to generate data.")
else:
    print(f"SUCCESS: Found data file '{data_filename}'. Analyzing content...")
    
    with open(data_filename, 'rb') as f:
        discussion_data = pickle.load(f)
        
    print("\n--- Data Points per Role ---")
    total_points = 0
    for role in config.ROLES:
        if role in discussion_data and discussion_data[role]:
            num_points = len(discussion_data[role])
            print(f"  - {role.capitalize():<10}: {num_points} data points found.")
            total_points += num_points
        else:
            print(f"  - {role.capitalize():<10}: 0 data points found. (This is why its clustering was skipped)")
            
    print("----------------------------")
    print(f"Total data points collected: {total_points}")

print("\n--- Check Finished ---")