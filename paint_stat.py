"""
This is used to get some stat from the csv file, i.e. the overall information of the paintings
"""

import matplotlib.pyplot as plt
import pandas as pd


file_path = 'train_info.csv'
file = pd.read_csv(file_path)

# style = file['style'].value_counts()
# for idx, item in style.items():
#     print(idx, item)

print("---------------------")
name_start_with = '1'
label_column = 1
if label_column == 1:
    label = 'artist'
elif label_column == 3:
    label = 'style'
elif label_column == 4:
    label = 'genre'
selected = file[file.filename.str.startswith(name_start_with)]

selected_count = file[label].value_counts()
print(selected_count.shape)

count = 0
total_paint = 0
for name, item in selected_count.items():
    if item >= 390:
        count += 1
        total_paint += item
        print(name, item)
print(count, total_paint)

# style.plot.bar()
# plt.show()
