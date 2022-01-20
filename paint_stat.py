"""
This is used to get some stat from the csv file, i.e. the overall information of the paintings
"""

import matplotlib.pyplot as plt
import pandas as pd


file_path = 'train_info.csv'
file = pd.read_csv(file_path)

style = file['style'].value_counts()
# for idx, item in style.items():
#     print(idx, item)

print("---------------------")

file_1 = file[file.filename.str.startswith('1')]
style_1 = file_1['style'].value_counts()
print(style_1.shape)
count = 0
total_paint = 0
for name, item in style.items():
    if item >= 1000:
        count += 1
        total_paint += item
        print(name, item)
print(count, total_paint)

# style.plot.bar()
# plt.show()
