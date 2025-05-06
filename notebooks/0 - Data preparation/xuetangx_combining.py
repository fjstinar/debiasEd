import pandas as pd
import os

# Read the CSV files
train_log = pd.read_csv('../data/xuetangx/train_log.csv')
test_log = pd.read_csv('../data/xuetangx/test_log.csv')
train_truth = pd.read_csv('../data/xuetangx/train_truth.csv')
test_truth = pd.read_csv('../data/xuetangx/test_truth.csv')
user_info = pd.read_csv('../data/xuetangx/user_info.csv')

# Combine test and train logs
combined_log = pd.concat([train_log, test_log], ignore_index=True)

# Combine test and train truth data
combined_truth = pd.concat([train_truth, test_truth], ignore_index=True)

# Merge all dataframes
# First merge log data with truth data
merged_data = pd.merge(combined_log, combined_truth, on='enroll_id', how='outer')

# Then merge with user info, matching 'username' with 'user_id'
final_data = pd.merge(merged_data, user_info, left_on='username', right_on='user_id', how='outer')

# Save the combined dataset
final_data.to_csv('../data/xuetangx/combined_dataset.csv', index=False)

print("Data combination complete. Output saved to 'combined_dataset.csv'")
print(f"Final dataset shape: {final_data.shape}")
