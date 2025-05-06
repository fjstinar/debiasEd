import pandas as pd
import os

# Read the CSV files
student_metadata = pd.read_csv('../data/eedi/student_metadata_task_1_2.csv')
train_data = pd.read_csv('../data/eedi/train_task_1_2.csv')
answer_metadata = pd.read_csv('../data/eedi/answer_metadata_task_1_2.csv')

# Merge student data with training data on UserId
combined_data = pd.merge(train_data, student_metadata, on='UserId', how='outer')

# Merge with answer metadata on AnswerId
final_data = pd.merge(combined_data, answer_metadata, on='AnswerId', how='outer')

# Count unique users by gender
gender_counts = final_data[['UserId', 'Gender']].drop_duplicates()['Gender'].value_counts()
print("\nUnique users by gender:")
print(gender_counts)

# Save the combined dataset
final_data.to_csv('../data/eedi/combined_eedi_data.csv', index=False)

print("\nData combination complete. Output saved to 'combined_eedi_data.csv'")
print(f"Final dataset shape: {final_data.shape}")
