import glob
import csv
import pandas as pd
interesting_files = glob.glob(r"C:\Users\LEGA\Documents\Geofisica\MB\magnetometry\full\*.INT") 

# header_saved = False
# with open(r'C:\Users\LEGA\Documents\Geofisica\MB\magnetometry\full\Merged.INT', 'w') as fout:
#     writer = csv.writer(fout)
#     for filename in interesting_files:
#         with open(filename) as fin:
#             header =  next(fin)
#             if not header_saved:
#                 writer.writerows(header) # you may need to work here. The writerows require an iterable.
#                 header_saved = True
#             writer.writerows(fin.readlines())
# Read and concatenate all CSV files
df_list = [pd.read_csv(file) for file in interesting_files]
concatenated_df = pd.concat(df_list, ignore_index=True)

# Save to a new CSV
concatenated_df.to_csv(r"C:\Users\LEGA\Documents\Geofisica\MB\magnetometry\full\merged.INT", index=False)