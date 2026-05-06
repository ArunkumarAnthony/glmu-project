import pandas as pd
from chembl_webresource_client.new_client import new_client

print("Fetching raw GlmU data from ChEMBL...")

# 1. Search for GlmU targets
target_api = new_client.target
targets = pd.DataFrame.from_dict(target_api.search('GlmU'))
target_ids = targets['target_chembl_id'].tolist()

# 2. Fetch ALL raw activity data for these targets
activity_api = new_client.activity
activities = activity_api.filter(target_chembl_id__in=target_ids)

# 3. Convert to DataFrame and save immediately with ZERO processing
df_raw = pd.DataFrame.from_dict(activities)

output_file = 'glmu_raw_chembl_dump.csv'
df_raw.to_csv(output_file, index=False)

print(f"Done. Saved {len(df_raw)} totally raw, unprocessed rows to {output_file}")