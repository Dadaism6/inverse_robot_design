import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
df = pd.read_csv("/Users/duanchenda/Desktop/gitplay/inverse_robot_design/compare_result/lrcompare.csv")
print(df)
ground_truth = df["Target Velocity"].tolist()
prediction = df["What turly is by simulator"].tolist()
percentdifference = df["Difference between Target and truly is in percentage"].tolist()

percent_diff = [abs(a) for a in percentdifference]
print(percent_diff)
axis = range(1,len(percent_diff)+1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(axis, ground_truth , label="Ground Truth")
ax.scatter(axis, prediction , label="Generated")
ax.set_ylabel("Generated and Ground Truth Velocity")
ax.set_xlabel("Sample Number")
plt.legend(loc='upper right')
plt.xticks(rotation = 70)
axB = ax.twinx()
axB.bar(axis, percent_diff, fill=False, label='Percentage Difference')
axB.set_ylabel("Percent Difference in %")
plt.legend(loc='upper left')
plt.show()