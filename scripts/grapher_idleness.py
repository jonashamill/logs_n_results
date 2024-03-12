
# We will use the following Python libraries:
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
# import rospkg
# import rospy

# Path to where the results are hosted online
# This loads the data into a Pandas dataframe
# You have seperate files for all trials and all
# robots, so these have to be loaded in individually

min_time = 0
max_time = 1200

conditions = ['P', 'NP', 'NM']
trials = 5
robots = 3


# Set these to the experiment params
experiment = '8th_exp_new_paths'
title_extension = "- New Behaviour + Paths - Sim"

# rospy.init_node("grapher")

# os.system("roscore")

# rp = rospkg.RosPack()
# package_path = rp.get_path('logs_n_results')

package_path = 'logs_n_results'


graph_path = f'{package_path}/results/graphs/{experiment}_idleness'


if not os.path.exists(f"{graph_path}/pngs"):
    os.makedirs(f"{graph_path}/pngs")

if not os.path.exists(f"{graph_path}/pdfs"):
    os.makedirs(f"{graph_path}/pdfs")


# Here we loop through the paths to find the csv containing the results
data = {}
for cond in conditions:
    data[cond] = {}
    for t in range(1, trials+1):
        data[cond][f"T{t}"] = {}
        for r in range(1, robots+1):
            path = f'{package_path}/logs/{experiment}/robot{r}/{cond}/trial_{t}/'
            csv_files = os.listdir(path)
            csv_file = [f for f in csv_files if f.endswith('poselog.csv')][0]
            df = pd.read_csv(os.path.join(path, csv_file), header=None)
            data[cond][f"T{t}"][f"R{r}"] = df




columns = ['x', 'y', 'ox', 'oy', 'oz', 'ow', 'time']
# Here we create variables for each trial/robot/condition
for cond in conditions:
  for t in range(1, trials+1):
    for r in range(1, robots+1):
      
      # Construct variable name  
      var_name = f"{cond}_T{t}_R{r}"
      
      # Get DataFrame
      df = data[cond][f"T{t}"][f"R{r}"]
      
      # Assign to variable
      exec(f"{var_name} = df") 

      df = data[cond][f"T{t}"][f"R{r}"]  
      df.columns = columns

#       df.insert(0, 'condition', cond)
#       df.insert(1, 'trial', str(t))  
#       df.insert(2, 'robot', str(r))


# Add columns to apply labels to data - for some reason this is the only part that doesn't like to be 
# put into a loop
      
# P
P_T1_R2.insert(0, 'condition', 'P')
P_T1_R1.insert(0, 'condition', 'P')
P_T1_R3.insert(0, 'condition', 'P')
P_T1_R2.insert(1, 'trial', '1')
P_T1_R1.insert(1, 'trial', '1')
P_T1_R3.insert(1, 'trial', '1')
P_T1_R1.insert(2, 'robot', '1')
P_T1_R2.insert(2, 'robot', '2')
P_T1_R3.insert(2, 'robot', '3')

P_T2_R2.insert(0, 'condition', 'P')
P_T2_R1.insert(0, 'condition', 'P')
P_T2_R3.insert(0, 'condition', 'P')
P_T2_R2.insert(1, 'trial', '2')
P_T2_R1.insert(1, 'trial', '2')
P_T2_R3.insert(1, 'trial', '2')
P_T2_R1.insert(2, 'robot', '1')
P_T2_R2.insert(2, 'robot', '2')
P_T2_R3.insert(2, 'robot', '3')

P_T3_R2.insert(0, 'condition', 'P')
P_T3_R1.insert(0, 'condition', 'P')
P_T3_R3.insert(0, 'condition', 'P')
P_T3_R2.insert(1, 'trial', '3')
P_T3_R1.insert(1, 'trial', '3')
P_T3_R3.insert(1, 'trial', '3')
P_T3_R1.insert(2, 'robot', '1')
P_T3_R2.insert(2, 'robot', '2')
P_T3_R3.insert(2, 'robot', '3')

P_T4_R2.insert(0, 'condition', 'P')
P_T4_R1.insert(0, 'condition', 'P')
P_T4_R3.insert(0, 'condition', 'P')
P_T4_R2.insert(1, 'trial', '4')
P_T4_R1.insert(1, 'trial', '4')
P_T4_R3.insert(1, 'trial', '4')
P_T4_R1.insert(2, 'robot', '1')
P_T4_R2.insert(2, 'robot', '2')
P_T4_R3.insert(2, 'robot', '3')

# Fixing Trials 5 to 10 for condition 'P'
P_T5_R2.insert(0, 'condition', 'P')
P_T5_R1.insert(0, 'condition', 'P')
P_T5_R3.insert(0, 'condition', 'P')
P_T5_R2.insert(1, 'trial', '5')
P_T5_R1.insert(1, 'trial', '5')
P_T5_R3.insert(1, 'trial', '5')
P_T5_R1.insert(2, 'robot', '1')
P_T5_R2.insert(2, 'robot', '2')
P_T5_R3.insert(2, 'robot', '3')

# P_T6_R2.insert(0, 'condition', 'P')
# P_T6_R1.insert(0, 'condition', 'P')
# P_T6_R3.insert(0, 'condition', 'P')
# P_T6_R2.insert(1, 'trial', '6')
# P_T6_R1.insert(1, 'trial', '6')
# P_T6_R3.insert(1, 'trial', '6')
# P_T6_R1.insert(2, 'robot', '1')
# P_T6_R2.insert(2, 'robot', '2')
# P_T6_R3.insert(2, 'robot', '3')

# P_T7_R2.insert(0, 'condition', 'P')
# P_T7_R1.insert(0, 'condition', 'P')
# P_T7_R3.insert(0, 'condition', 'P')
# P_T7_R2.insert(1, 'trial', '7')
# P_T7_R1.insert(1, 'trial', '7')
# P_T7_R3.insert(1, 'trial', '7')
# P_T7_R1.insert(2, 'robot', '1')
# P_T7_R2.insert(2, 'robot', '2')
# P_T7_R3.insert(2, 'robot', '3')

# P_T8_R2.insert(0, 'condition', 'P')
# P_T8_R1.insert(0, 'condition', 'P')
# P_T8_R3.insert(0, 'condition', 'P')
# P_T8_R2.insert(1, 'trial', '8')
# P_T8_R1.insert(1, 'trial', '8')
# P_T8_R3.insert(1, 'trial', '8')
# P_T8_R1.insert(2, 'robot', '1')
# P_T8_R2.insert(2, 'robot', '2')
# P_T8_R3.insert(2, 'robot', '3')

# P_T9_R2.insert(0, 'condition', 'P')
# P_T9_R1.insert(0, 'condition', 'P')
# P_T9_R3.insert(0, 'condition', 'P')
# P_T9_R2.insert(1, 'trial', '9')
# P_T9_R1.insert(1, 'trial', '9')
# P_T9_R3.insert(1, 'trial', '9')
# P_T9_R1.insert(2, 'robot', '1')
# P_T9_R2.insert(2, 'robot', '2')
# P_T9_R3.insert(2, 'robot', '3')

# P_T10_R2.insert(0, 'condition', 'P')
# P_T10_R1.insert(0, 'condition', 'P')
# P_T10_R3.insert(0, 'condition', 'P')
# P_T10_R2.insert(1, 'trial', '10')
# P_T10_R1.insert(1, 'trial', '10')
# P_T10_R3.insert(1, 'trial', '10')
# P_T10_R1.insert(2, 'robot', '1')
# P_T10_R2.insert(2, 'robot', '2')
# P_T10_R3.insert(2, 'robot', '3')
# NP
NP_T1_R2.insert(0, 'condition', 'NP')
NP_T1_R1.insert(0, 'condition', 'NP')
NP_T1_R3.insert(0, 'condition', 'NP')
NP_T1_R2.insert(1, 'trial', '1')
NP_T1_R1.insert(1, 'trial', '1')
NP_T1_R3.insert(1, 'trial', '1')
NP_T1_R1.insert(2, 'robot', '1')
NP_T1_R2.insert(2, 'robot', '2')
NP_T1_R3.insert(2, 'robot', '3')

NP_T2_R2.insert(0, 'condition', 'NP')
NP_T2_R1.insert(0, 'condition', 'NP')
NP_T2_R3.insert(0, 'condition', 'NP')
NP_T2_R2.insert(1, 'trial', '2')
NP_T2_R1.insert(1, 'trial', '2')
NP_T2_R3.insert(1, 'trial', '2')
NP_T2_R1.insert(2, 'robot', '1')
NP_T2_R2.insert(2, 'robot', '2')
NP_T2_R3.insert(2, 'robot', '3')

NP_T3_R2.insert(0, 'condition', 'NP')
NP_T3_R1.insert(0, 'condition', 'NP')
NP_T3_R3.insert(0, 'condition', 'NP')
NP_T3_R2.insert(1, 'trial', '3')
NP_T3_R1.insert(1, 'trial', '3')
NP_T3_R3.insert(1, 'trial', '3')
NP_T3_R1.insert(2, 'robot', '1')
NP_T3_R2.insert(2, 'robot', '2')
NP_T3_R3.insert(2, 'robot', '3')

NP_T4_R2.insert(0, 'condition', 'NP')
NP_T4_R1.insert(0, 'condition', 'NP')
NP_T4_R3.insert(0, 'condition', 'NP')
NP_T4_R2.insert(1, 'trial', '4')
NP_T4_R1.insert(1, 'trial', '4')
NP_T4_R3.insert(1, 'trial', '4')
NP_T4_R1.insert(2, 'robot', '1')
NP_T4_R2.insert(2, 'robot', '2')
NP_T4_R3.insert(2, 'robot', '3')

# Fixing Trials 5 to 10 for condition 'NP'
NP_T5_R2.insert(0, 'condition', 'NP')
NP_T5_R1.insert(0, 'condition', 'NP')
NP_T5_R3.insert(0, 'condition', 'NP')
NP_T5_R2.insert(1, 'trial', '5')
NP_T5_R1.insert(1, 'trial', '5')
NP_T5_R3.insert(1, 'trial', '5')
NP_T5_R1.insert(2, 'robot', '1')
NP_T5_R2.insert(2, 'robot', '2')
NP_T5_R3.insert(2, 'robot', '3')

# NP_T6_R2.insert(0, 'condition', 'NP')
# NP_T6_R1.insert(0, 'condition', 'NP')
# NP_T6_R3.insert(0, 'condition', 'NP')
# NP_T6_R2.insert(1, 'trial', '6')
# NP_T6_R1.insert(1, 'trial', '6')
# NP_T6_R3.insert(1, 'trial', '6')
# NP_T6_R1.insert(2, 'robot', '1')
# NP_T6_R2.insert(2, 'robot', '2')
# NP_T6_R3.insert(2, 'robot', '3')

# NP_T7_R2.insert(0, 'condition', 'NP')
# NP_T7_R1.insert(0, 'condition', 'NP')
# NP_T7_R3.insert(0, 'condition', 'NP')
# NP_T7_R2.insert(1, 'trial', '7')
# NP_T7_R1.insert(1, 'trial', '7')
# NP_T7_R3.insert(1, 'trial', '7')
# NP_T7_R1.insert(2, 'robot', '1')
# NP_T7_R2.insert(2, 'robot', '2')
# NP_T7_R3.insert(2, 'robot', '3')

# NP_T8_R2.insert(0, 'condition', 'NP')
# NP_T8_R1.insert(0, 'condition', 'NP')
# NP_T8_R3.insert(0, 'condition', 'NP')
# NP_T8_R2.insert(1, 'trial', '8')
# NP_T8_R1.insert(1, 'trial', '8')
# NP_T8_R3.insert(1, 'trial', '8')
# NP_T8_R1.insert(2, 'robot', '1')
# NP_T8_R2.insert(2, 'robot', '2')
# NP_T8_R3.insert(2, 'robot', '3')

# NP_T9_R2.insert(0, 'condition', 'NP')
# NP_T9_R1.insert(0, 'condition', 'NP')
# NP_T9_R3.insert(0, 'condition', 'NP')
# NP_T9_R2.insert(1, 'trial', '9')
# NP_T9_R1.insert(1, 'trial', '9')
# NP_T9_R3.insert(1, 'trial', '9')
# NP_T9_R1.insert(2, 'robot', '1')
# NP_T9_R2.insert(2, 'robot', '2')
# NP_T9_R3.insert(2, 'robot', '3')

# NP_T10_R2.insert(0, 'condition', 'NP')
# NP_T10_R1.insert(0, 'condition', 'NP')
# NP_T10_R3.insert(0, 'condition', 'NP')
# NP_T10_R2.insert(1, 'trial', '10')
# NP_T10_R1.insert(1, 'trial', '10')
# NP_T10_R3.insert(1, 'trial', '10')
# NP_T10_R1.insert(2, 'robot', '1')
# NP_T10_R2.insert(2, 'robot', '2')
# NP_T10_R3.insert(2, 'robot', '3')

# NM
NM_T1_R2.insert(0, 'condition', 'NM')
NM_T1_R1.insert(0, 'condition', 'NM')
NM_T1_R3.insert(0, 'condition', 'NM')
NM_T1_R2.insert(1, 'trial', '1')
NM_T1_R1.insert(1, 'trial', '1')
NM_T1_R3.insert(1, 'trial', '1')
NM_T1_R1.insert(2, 'robot', '1')
NM_T1_R2.insert(2, 'robot', '2')
NM_T1_R3.insert(2, 'robot', '3')

NM_T2_R2.insert(0, 'condition', 'NM')
NM_T2_R1.insert(0, 'condition', 'NM')
NM_T2_R3.insert(0, 'condition', 'NM')
NM_T2_R2.insert(1, 'trial', '2')
NM_T2_R1.insert(1, 'trial', '2')
NM_T2_R3.insert(1, 'trial', '2')
NM_T2_R1.insert(2, 'robot', '1')
NM_T2_R2.insert(2, 'robot', '2')
NM_T2_R3.insert(2, 'robot', '3')

NM_T3_R2.insert(0, 'condition', 'NM')
NM_T3_R1.insert(0, 'condition', 'NM')
NM_T3_R3.insert(0, 'condition', 'NM')
NM_T3_R2.insert(1, 'trial', '3')
NM_T3_R1.insert(1, 'trial', '3')
NM_T3_R3.insert(1, 'trial', '3')
NM_T3_R1.insert(2, 'robot', '1')
NM_T3_R2.insert(2, 'robot', '2')
NM_T3_R3.insert(2, 'robot', '3')

NM_T4_R2.insert(0, 'condition', 'NM')
NM_T4_R1.insert(0, 'condition', 'NM')
NM_T4_R3.insert(0, 'condition', 'NM')
NM_T4_R2.insert(1, 'trial', '4')
NM_T4_R1.insert(1, 'trial', '4')
NM_T4_R3.insert(1, 'trial', '4')
NM_T4_R1.insert(2, 'robot', '1')
NM_T4_R2.insert(2, 'robot', '2')
NM_T4_R3.insert(2, 'robot', '3')

# Fixing Trials 5 to 10 for condition 'NM'
NM_T5_R2.insert(0, 'condition', 'NM')
NM_T5_R1.insert(0, 'condition', 'NM')
NM_T5_R3.insert(0, 'condition', 'NM')
NM_T5_R2.insert(1, 'trial', '5')
NM_T5_R1.insert(1, 'trial', '5')
NM_T5_R3.insert(1, 'trial', '5')
NM_T5_R1.insert(2, 'robot', '1')
NM_T5_R2.insert(2, 'robot', '2')
NM_T5_R3.insert(2, 'robot', '3')

# NM_T6_R2.insert(0, 'condition', 'NM')
# NM_T6_R1.insert(0, 'condition', 'NM')
# NM_T6_R3.insert(0, 'condition', 'NM')
# NM_T6_R2.insert(1, 'trial', '6')
# NM_T6_R1.insert(1, 'trial', '6')
# NM_T6_R3.insert(1, 'trial', '6')
# NM_T6_R1.insert(2, 'robot', '1')
# NM_T6_R2.insert(2, 'robot', '2')
# NM_T6_R3.insert(2, 'robot', '3')

# NM_T7_R2.insert(0, 'condition', 'NM')
# NM_T7_R1.insert(0, 'condition', 'NM')
# NM_T7_R3.insert(0, 'condition', 'NM')
# NM_T7_R2.insert(1, 'trial', '7')
# NM_T7_R1.insert(1, 'trial', '7')
# NM_T7_R3.insert(1, 'trial', '7')
# NM_T7_R1.insert(2, 'robot', '1')
# NM_T7_R2.insert(2, 'robot', '2')
# NM_T7_R3.insert(2, 'robot', '3')

# NM_T8_R2.insert(0, 'condition', 'NM')
# NM_T8_R1.insert(0, 'condition', 'NM')
# NM_T8_R3.insert(0, 'condition', 'NM')
# NM_T8_R2.insert(1, 'trial', '8')
# NM_T8_R1.insert(1, 'trial', '8')
# NM_T8_R3.insert(1, 'trial', '8')
# NM_T8_R1.insert(2, 'robot', '1')
# NM_T8_R2.insert(2, 'robot', '2')
# NM_T8_R3.insert(2, 'robot', '3')

# NM_T9_R2.insert(0, 'condition', 'NM')
# NM_T9_R1.insert(0, 'condition', 'NM')
# NM_T9_R3.insert(0, 'condition', 'NM')
# NM_T9_R2.insert(1, 'trial', '9')
# NM_T9_R1.insert(1, 'trial', '9')
# NM_T9_R3.insert(1, 'trial', '9')
# NM_T9_R1.insert(2, 'robot', '1')
# NM_T9_R2.insert(2, 'robot', '2')
# NM_T9_R3.insert(2, 'robot', '3')

# NM_T10_R2.insert(0, 'condition', 'NM')
# NM_T10_R1.insert(0, 'condition', 'NM')
# NM_T10_R3.insert(0, 'condition', 'NM')
# NM_T10_R2.insert(1, 'trial', '10')
# NM_T10_R1.insert(1, 'trial', '10')
# NM_T10_R3.insert(1, 'trial', '10')
# NM_T10_R1.insert(2, 'robot', '1')
# NM_T10_R2.insert(2, 'robot', '2')
# NM_T10_R3.insert(2, 'robot', '3')



frames = []

for cond in conditions:
  for t in range(1, trials+1):
    for r in range(1, robots+1):
      
      df = data[cond][f"T{t}"][f"R{r}"]
      frames.append(df)



# Concatenate them together into 1 dataframe
all_df = pd.concat(frames, ignore_index=True)


# Your data has high resolution precision on time,
# which means the x-axis will get indexed with individual
# time markers, rather than collating per second (for example)
# This means that we can't plot a distribution, because every
# row of data for every trial appears unique.
# It look like your data is per second, just with some
# arbitrary offset.  Rounding down (0 decimal places)
# seems to fix this.
all_df['time'] = all_df['time'].round(0)

# Make sure our numeric columns are not strings
all_df["robot"] = pd.to_numeric(all_df["robot"])
all_df["trial"] = pd.to_numeric(all_df["trial"])
all_df["x"] = pd.to_numeric(all_df["x"])
all_df["y"] = pd.to_numeric(all_df["y"])
all_df["ox"] = pd.to_numeric(all_df["ox"])
all_df["oy"] = pd.to_numeric(all_df["oy"])
all_df["ow"] = pd.to_numeric(all_df["ow"])

# It looks like you have some NaN's?
# print( "Oops, number of NaN's: ", all_df.isnull().sum().sum() )





# Exclude data based on our min max time values above
all_df = all_df[ ( all_df['time'] >= min_time ) ]
all_df = all_df[ ( all_df['time'] <= max_time ) ]

# We can peek inside each file to
# check formatting like this
print( all_df )

# Initialize a figure and axis object for plotting
# Aggregate data
aggregated_data = {}
for cond in conditions:
    frames = []
    for t in range(1, trials+1):
        for r in range(1, robots+1):
            frames.append(data[cond][f"T{t}"][f"R{r}"])
    aggregated_data[cond] = pd.concat(frames)

# Create heatmaps
for cond in conditions:
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=aggregated_data[cond]['x'], y=aggregated_data[cond]['y'], cmap="Reds", shade=True, bw_adjust=0.5)
    plt.title(f"Heatmap of Robot Positions - Condition {cond}")
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig(f'{graph_path}/pngs/{experiment}_{cond}_heatmap.png')
    plt.savefig(f'{graph_path}/pdfs/{experiment}_{cond}_heatmap.pdf')
    plt.close()
# # Result P

# subset = all_df[ ( all_df['condition'] == 'P' ) ]
# #print( subset )

# # Set y-axis limits
# y_limits = (0, 110)

# # having trouble with hue colours
# colours = dict(zip(subset['robot'].unique(), sns.color_palette(n_colors=len(subset['robot'].unique()))))

# lplot = sns.relplot(data=subset, x='x', y='y', kind='line', hue='robot', palette=colours, height=5, aspect=2, errorbar=('ci',95) )
# lplot.fig.suptitle(f'Neo Threshold {conditions[0]} {title_extension}', fontsize=16)

# # Set y-axis limits
# plt.ylim(y_limits)
# plt.axhline(y=50, linestyle='--', linewidth=0.8, c='hotpink')


# plt.xlabel('Time (s)')
# plt.ylabel('Neophilia')

# plt.savefig(f'{graph_path}/pngs/{experiment}_{conditions[0]}_AL.png')
# plt.savefig(f'{graph_path}/pdfs/{experiment}_{conditions[0]}_AL.pdf')
# plt.close()




# # Results all compared 
# subset = all_df[ ( all_df['time'] < max_time ) ]
# #print( subset )



# # having trouble with hue colours
# colours = dict(zip(subset['condition'].unique(), sns.color_palette(n_colors=len(subset['condition'].unique()))))

# lplot = sns.relplot(data=subset, x='x', y='y', kind='line', hue='condition', palette=colours, height=5, aspect=2, errorbar=('ci',95) )
# lplot.fig.suptitle(f'Neo Threshold Over Time {title_extension}', fontsize=16)

# plt.axhline(y=50, linestyle='--', linewidth=0.8, c='hotpink')


# plt.xlabel('Time (s)')
# plt.ylabel('Neophilia')

# plt.savefig(f'{graph_path}/pngs/{experiment}_ALL_AL.png')
# plt.savefig(f'{graph_path}/pdfs/{experiment}_ALL_AL.pdf')
# plt.close()
