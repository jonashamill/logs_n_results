
# We will use the following Python libraries:
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import rospkg
import rospy

# Path to where the results are hosted online
# This loads the data into a Pandas dataframe
# You have seperate files for all trials and all
# robots, so these have to be loaded in individually

min_time = 0
max_time = 1200

conditions = ['NP', 'TTH_2_6', 'TTH_6_10', 'TTH_100_200', 'TTH_1_1']
trials = 10
robots = 3


# Set these to the experiment params
experiment = '4th_exp_1_tag_buffer_increase_at_600'
title_extension = "- Decrease at 600(s), 1 tag buffer- Sim"

# rospy.init_node("grapher")

os.system("roscore")

rp = rospkg.RosPack()
package_path = rp.get_path('logs_n_results')



graph_path = f'{package_path}/results/graphs/{experiment}'


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
            path = f'{package_path}/logs/{experiment}/robot{r}/{cond}/trial_{t}/metrics/'
            csv_files = os.listdir(path)
            csv_file = [f for f in csv_files if f.endswith('metricspog.csv')][0]
            df = pd.read_csv(os.path.join(path, csv_file), header=None)
            data[cond][f"T{t}"][f"R{r}"] = df




columns = ['time', 'state', 'u_rand_num', 'activation', 'cumul_tags']
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


# Add columns to apply labels to data - for some reason this is the only one it doesn't like to put into a loop
# TTH 2 6
TTH_2_6_T1_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T1_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T1_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T1_R2.insert(1, 'trial', '1')
TTH_2_6_T1_R1.insert(1, 'trial', '1')
TTH_2_6_T1_R3.insert(1, 'trial', '1')
TTH_2_6_T1_R1.insert(2, 'robot', '1')
TTH_2_6_T1_R2.insert(2, 'robot', '2')
TTH_2_6_T1_R3.insert(2, 'robot', '3')

TTH_2_6_T2_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T2_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T2_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T2_R2.insert(1, 'trial', '2')
TTH_2_6_T2_R1.insert(1, 'trial', '2')
TTH_2_6_T2_R3.insert(1, 'trial', '2')
TTH_2_6_T2_R1.insert(2, 'robot', '1')
TTH_2_6_T2_R2.insert(2, 'robot', '2')
TTH_2_6_T2_R3.insert(2, 'robot', '3')

TTH_2_6_T3_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T3_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T3_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T3_R2.insert(1, 'trial', '3')
TTH_2_6_T3_R1.insert(1, 'trial', '3')
TTH_2_6_T3_R3.insert(1, 'trial', '3')
TTH_2_6_T3_R1.insert(2, 'robot', '1')
TTH_2_6_T3_R2.insert(2, 'robot', '2')
TTH_2_6_T3_R3.insert(2, 'robot', '3')

TTH_2_6_T4_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T4_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T4_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T4_R2.insert(1, 'trial', '4')
TTH_2_6_T4_R1.insert(1, 'trial', '4')
TTH_2_6_T4_R3.insert(1, 'trial', '4')
TTH_2_6_T4_R1.insert(2, 'robot', '1')
TTH_2_6_T4_R2.insert(2, 'robot', '2')
TTH_2_6_T4_R3.insert(2, 'robot', '3')

# Fixing Trials 5 to 10 for condition 'tth_2_6'
TTH_2_6_T5_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T5_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T5_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T5_R2.insert(1, 'trial', '5')
TTH_2_6_T5_R1.insert(1, 'trial', '5')
TTH_2_6_T5_R3.insert(1, 'trial', '5')
TTH_2_6_T5_R1.insert(2, 'robot', '1')
TTH_2_6_T5_R2.insert(2, 'robot', '2')
TTH_2_6_T5_R3.insert(2, 'robot', '3')

TTH_2_6_T6_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T6_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T6_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T6_R2.insert(1, 'trial', '6')
TTH_2_6_T6_R1.insert(1, 'trial', '6')
TTH_2_6_T6_R3.insert(1, 'trial', '6')
TTH_2_6_T6_R1.insert(2, 'robot', '1')
TTH_2_6_T6_R2.insert(2, 'robot', '2')
TTH_2_6_T6_R3.insert(2, 'robot', '3')

TTH_2_6_T7_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T7_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T7_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T7_R2.insert(1, 'trial', '7')
TTH_2_6_T7_R1.insert(1, 'trial', '7')
TTH_2_6_T7_R3.insert(1, 'trial', '7')
TTH_2_6_T7_R1.insert(2, 'robot', '1')
TTH_2_6_T7_R2.insert(2, 'robot', '2')
TTH_2_6_T7_R3.insert(2, 'robot', '3')

TTH_2_6_T8_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T8_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T8_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T8_R2.insert(1, 'trial', '8')
TTH_2_6_T8_R1.insert(1, 'trial', '8')
TTH_2_6_T8_R3.insert(1, 'trial', '8')
TTH_2_6_T8_R1.insert(2, 'robot', '1')
TTH_2_6_T8_R2.insert(2, 'robot', '2')
TTH_2_6_T8_R3.insert(2, 'robot', '3')

TTH_2_6_T9_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T9_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T9_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T9_R2.insert(1, 'trial', '9')
TTH_2_6_T9_R1.insert(1, 'trial', '9')
TTH_2_6_T9_R3.insert(1, 'trial', '9')
TTH_2_6_T9_R1.insert(2, 'robot', '1')
TTH_2_6_T9_R2.insert(2, 'robot', '2')
TTH_2_6_T9_R3.insert(2, 'robot', '3')

TTH_2_6_T10_R2.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T10_R1.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T10_R3.insert(0, 'condition', 'tth_2_6')
TTH_2_6_T10_R2.insert(1, 'trial', '10')
TTH_2_6_T10_R1.insert(1, 'trial', '10')
TTH_2_6_T10_R3.insert(1, 'trial', '10')
TTH_2_6_T10_R1.insert(2, 'robot', '1')
TTH_2_6_T10_R2.insert(2, 'robot', '2')
TTH_2_6_T10_R3.insert(2, 'robot', '3')



# TTH 6 10
TTH_6_10_T1_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T1_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T1_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T1_R2.insert(1, 'trial', '1')
TTH_6_10_T1_R1.insert(1, 'trial', '1')
TTH_6_10_T1_R3.insert(1, 'trial', '1')
TTH_6_10_T1_R1.insert(2, 'robot', '1')
TTH_6_10_T1_R2.insert(2, 'robot', '2')
TTH_6_10_T1_R3.insert(2, 'robot', '3')

TTH_6_10_T2_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T2_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T2_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T2_R2.insert(1, 'trial', '2')
TTH_6_10_T2_R1.insert(1, 'trial', '2')
TTH_6_10_T2_R3.insert(1, 'trial', '2')
TTH_6_10_T2_R1.insert(2, 'robot', '1')
TTH_6_10_T2_R2.insert(2, 'robot', '2')
TTH_6_10_T2_R3.insert(2, 'robot', '3')

TTH_6_10_T3_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T3_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T3_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T3_R2.insert(1, 'trial', '3')
TTH_6_10_T3_R1.insert(1, 'trial', '3')
TTH_6_10_T3_R3.insert(1, 'trial', '3')
TTH_6_10_T3_R1.insert(2, 'robot', '1')
TTH_6_10_T3_R2.insert(2, 'robot', '2')
TTH_6_10_T3_R3.insert(2, 'robot', '3')

TTH_6_10_T4_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T4_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T4_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T4_R2.insert(1, 'trial', '4')
TTH_6_10_T4_R1.insert(1, 'trial', '4')
TTH_6_10_T4_R3.insert(1, 'trial', '4')
TTH_6_10_T4_R1.insert(2, 'robot', '1')
TTH_6_10_T4_R2.insert(2, 'robot', '2')
TTH_6_10_T4_R3.insert(2, 'robot', '3')

# Fixing Trials 5 to 10 for condition 'tth_6_10'
TTH_6_10_T5_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T5_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T5_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T5_R2.insert(1, 'trial', '5')
TTH_6_10_T5_R1.insert(1, 'trial', '5')
TTH_6_10_T5_R3.insert(1, 'trial', '5')
TTH_6_10_T5_R1.insert(2, 'robot', '1')
TTH_6_10_T5_R2.insert(2, 'robot', '2')
TTH_6_10_T5_R3.insert(2, 'robot', '3')

TTH_6_10_T6_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T6_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T6_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T6_R2.insert(1, 'trial', '6')
TTH_6_10_T6_R1.insert(1, 'trial', '6')
TTH_6_10_T6_R3.insert(1, 'trial', '6')
TTH_6_10_T6_R1.insert(2, 'robot', '1')
TTH_6_10_T6_R2.insert(2, 'robot', '2')
TTH_6_10_T6_R3.insert(2, 'robot', '3')

TTH_6_10_T7_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T7_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T7_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T7_R2.insert(1, 'trial', '7')
TTH_6_10_T7_R1.insert(1, 'trial', '7')
TTH_6_10_T7_R3.insert(1, 'trial', '7')
TTH_6_10_T7_R1.insert(2, 'robot', '1')
TTH_6_10_T7_R2.insert(2, 'robot', '2')
TTH_6_10_T7_R3.insert(2, 'robot', '3')

TTH_6_10_T8_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T8_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T8_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T8_R2.insert(1, 'trial', '8')
TTH_6_10_T8_R1.insert(1, 'trial', '8')
TTH_6_10_T8_R3.insert(1, 'trial', '8')
TTH_6_10_T8_R1.insert(2, 'robot', '1')
TTH_6_10_T8_R2.insert(2, 'robot', '2')
TTH_6_10_T8_R3.insert(2, 'robot', '3')

TTH_6_10_T9_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T9_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T9_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T9_R2.insert(1, 'trial', '9')
TTH_6_10_T9_R1.insert(1, 'trial', '9')
TTH_6_10_T9_R3.insert(1, 'trial', '9')
TTH_6_10_T9_R1.insert(2, 'robot', '1')
TTH_6_10_T9_R2.insert(2, 'robot', '2')
TTH_6_10_T9_R3.insert(2, 'robot', '3')

TTH_6_10_T10_R2.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T10_R1.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T10_R3.insert(0, 'condition', 'tth_6_10')
TTH_6_10_T10_R2.insert(1, 'trial', '10')
TTH_6_10_T10_R1.insert(1, 'trial', '10')
TTH_6_10_T10_R3.insert(1, 'trial', '10')
TTH_6_10_T10_R1.insert(2, 'robot', '1')
TTH_6_10_T10_R2.insert(2, 'robot', '2')
TTH_6_10_T10_R3.insert(2, 'robot', '3')



# TTH 100 200
TTH_100_200_T1_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T1_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T1_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T1_R2.insert(1, 'trial', '1')
TTH_100_200_T1_R1.insert(1, 'trial', '1')
TTH_100_200_T1_R3.insert(1, 'trial', '1')
TTH_100_200_T1_R1.insert(2, 'robot', '1')
TTH_100_200_T1_R2.insert(2, 'robot', '2')
TTH_100_200_T1_R3.insert(2, 'robot', '3')

TTH_100_200_T2_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T2_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T2_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T2_R2.insert(1, 'trial', '2')
TTH_100_200_T2_R1.insert(1, 'trial', '2')
TTH_100_200_T2_R3.insert(1, 'trial', '2')
TTH_100_200_T2_R1.insert(2, 'robot', '1')
TTH_100_200_T2_R2.insert(2, 'robot', '2')
TTH_100_200_T2_R3.insert(2, 'robot', '3')

TTH_100_200_T3_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T3_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T3_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T3_R2.insert(1, 'trial', '3')
TTH_100_200_T3_R1.insert(1, 'trial', '3')
TTH_100_200_T3_R3.insert(1, 'trial', '3')
TTH_100_200_T3_R1.insert(2, 'robot', '1')
TTH_100_200_T3_R2.insert(2, 'robot', '2')
TTH_100_200_T3_R3.insert(2, 'robot', '3')

TTH_100_200_T4_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T4_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T4_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T4_R2.insert(1, 'trial', '4')
TTH_100_200_T4_R1.insert(1, 'trial', '4')
TTH_100_200_T4_R3.insert(1, 'trial', '4')
TTH_100_200_T4_R1.insert(2, 'robot', '1')
TTH_100_200_T4_R2.insert(2, 'robot', '2')
TTH_100_200_T4_R3.insert(2, 'robot', '3')

# Fixing Trials 5 to 10 for condition 'tth_100_200'
TTH_100_200_T5_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T5_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T5_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T5_R2.insert(1, 'trial', '5')
TTH_100_200_T5_R1.insert(1, 'trial', '5')
TTH_100_200_T5_R3.insert(1, 'trial', '5')
TTH_100_200_T5_R1.insert(2, 'robot', '1')
TTH_100_200_T5_R2.insert(2, 'robot', '2')
TTH_100_200_T5_R3.insert(2, 'robot', '3')

TTH_100_200_T6_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T6_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T6_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T6_R2.insert(1, 'trial', '6')
TTH_100_200_T6_R1.insert(1, 'trial', '6')
TTH_100_200_T6_R3.insert(1, 'trial', '6')
TTH_100_200_T6_R1.insert(2, 'robot', '1')
TTH_100_200_T6_R2.insert(2, 'robot', '2')
TTH_100_200_T6_R3.insert(2, 'robot', '3')

TTH_100_200_T7_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T7_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T7_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T7_R2.insert(1, 'trial', '7')
TTH_100_200_T7_R1.insert(1, 'trial', '7')
TTH_100_200_T7_R3.insert(1, 'trial', '7')
TTH_100_200_T7_R1.insert(2, 'robot', '1')
TTH_100_200_T7_R2.insert(2, 'robot', '2')
TTH_100_200_T7_R3.insert(2, 'robot', '3')

TTH_100_200_T8_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T8_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T8_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T8_R2.insert(1, 'trial', '8')
TTH_100_200_T8_R1.insert(1, 'trial', '8')
TTH_100_200_T8_R3.insert(1, 'trial', '8')
TTH_100_200_T8_R1.insert(2, 'robot', '1')
TTH_100_200_T8_R2.insert(2, 'robot', '2')
TTH_100_200_T8_R3.insert(2, 'robot', '3')

TTH_100_200_T9_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T9_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T9_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T9_R2.insert(1, 'trial', '9')
TTH_100_200_T9_R1.insert(1, 'trial', '9')
TTH_100_200_T9_R3.insert(1, 'trial', '9')
TTH_100_200_T9_R1.insert(2, 'robot', '1')
TTH_100_200_T9_R2.insert(2, 'robot', '2')
TTH_100_200_T9_R3.insert(2, 'robot', '3')

TTH_100_200_T10_R2.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T10_R1.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T10_R3.insert(0, 'condition', 'tth_100_200')
TTH_100_200_T10_R2.insert(1, 'trial', '10')
TTH_100_200_T10_R1.insert(1, 'trial', '10')
TTH_100_200_T10_R3.insert(1, 'trial', '10')
TTH_100_200_T10_R1.insert(2, 'robot', '1')
TTH_100_200_T10_R2.insert(2, 'robot', '2')
TTH_100_200_T10_R3.insert(2, 'robot', '3')


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

NP_T6_R2.insert(0, 'condition', 'NP')
NP_T6_R1.insert(0, 'condition', 'NP')
NP_T6_R3.insert(0, 'condition', 'NP')
NP_T6_R2.insert(1, 'trial', '6')
NP_T6_R1.insert(1, 'trial', '6')
NP_T6_R3.insert(1, 'trial', '6')
NP_T6_R1.insert(2, 'robot', '1')
NP_T6_R2.insert(2, 'robot', '2')
NP_T6_R3.insert(2, 'robot', '3')

NP_T7_R2.insert(0, 'condition', 'NP')
NP_T7_R1.insert(0, 'condition', 'NP')
NP_T7_R3.insert(0, 'condition', 'NP')
NP_T7_R2.insert(1, 'trial', '7')
NP_T7_R1.insert(1, 'trial', '7')
NP_T7_R3.insert(1, 'trial', '7')
NP_T7_R1.insert(2, 'robot', '1')
NP_T7_R2.insert(2, 'robot', '2')
NP_T7_R3.insert(2, 'robot', '3')

NP_T8_R2.insert(0, 'condition', 'NP')
NP_T8_R1.insert(0, 'condition', 'NP')
NP_T8_R3.insert(0, 'condition', 'NP')
NP_T8_R2.insert(1, 'trial', '8')
NP_T8_R1.insert(1, 'trial', '8')
NP_T8_R3.insert(1, 'trial', '8')
NP_T8_R1.insert(2, 'robot', '1')
NP_T8_R2.insert(2, 'robot', '2')
NP_T8_R3.insert(2, 'robot', '3')

NP_T9_R2.insert(0, 'condition', 'NP')
NP_T9_R1.insert(0, 'condition', 'NP')
NP_T9_R3.insert(0, 'condition', 'NP')
NP_T9_R2.insert(1, 'trial', '9')
NP_T9_R1.insert(1, 'trial', '9')
NP_T9_R3.insert(1, 'trial', '9')
NP_T9_R1.insert(2, 'robot', '1')
NP_T9_R2.insert(2, 'robot', '2')
NP_T9_R3.insert(2, 'robot', '3')

NP_T10_R2.insert(0, 'condition', 'NP')
NP_T10_R1.insert(0, 'condition', 'NP')
NP_T10_R3.insert(0, 'condition', 'NP')
NP_T10_R2.insert(1, 'trial', '10')
NP_T10_R1.insert(1, 'trial', '10')
NP_T10_R3.insert(1, 'trial', '10')
NP_T10_R1.insert(2, 'robot', '1')
NP_T10_R2.insert(2, 'robot', '2')
NP_T10_R3.insert(2, 'robot', '3')


# TTH 1 1
TTH_1_1_T1_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T1_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T1_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T1_R2.insert(1, 'trial', '1')
TTH_1_1_T1_R1.insert(1, 'trial', '1')
TTH_1_1_T1_R3.insert(1, 'trial', '1')
TTH_1_1_T1_R1.insert(2, 'robot', '1')
TTH_1_1_T1_R2.insert(2, 'robot', '2')
TTH_1_1_T1_R3.insert(2, 'robot', '3')

TTH_1_1_T2_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T2_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T2_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T2_R2.insert(1, 'trial', '2')
TTH_1_1_T2_R1.insert(1, 'trial', '2')
TTH_1_1_T2_R3.insert(1, 'trial', '2')
TTH_1_1_T2_R1.insert(2, 'robot', '1')
TTH_1_1_T2_R2.insert(2, 'robot', '2')
TTH_1_1_T2_R3.insert(2, 'robot', '3')

TTH_1_1_T3_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T3_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T3_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T3_R2.insert(1, 'trial', '3')
TTH_1_1_T3_R1.insert(1, 'trial', '3')
TTH_1_1_T3_R3.insert(1, 'trial', '3')
TTH_1_1_T3_R1.insert(2, 'robot', '1')
TTH_1_1_T3_R2.insert(2, 'robot', '2')
TTH_1_1_T3_R3.insert(2, 'robot', '3')

TTH_1_1_T4_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T4_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T4_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T4_R2.insert(1, 'trial', '4')
TTH_1_1_T4_R1.insert(1, 'trial', '4')
TTH_1_1_T4_R3.insert(1, 'trial', '4')
TTH_1_1_T4_R1.insert(2, 'robot', '1')
TTH_1_1_T4_R2.insert(2, 'robot', '2')
TTH_1_1_T4_R3.insert(2, 'robot', '3')

# Fixing Trials 5 to 10 for condition 'tth_1_1'
TTH_1_1_T5_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T5_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T5_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T5_R2.insert(1, 'trial', '5')
TTH_1_1_T5_R1.insert(1, 'trial', '5')
TTH_1_1_T5_R3.insert(1, 'trial', '5')
TTH_1_1_T5_R1.insert(2, 'robot', '1')
TTH_1_1_T5_R2.insert(2, 'robot', '2')
TTH_1_1_T5_R3.insert(2, 'robot', '3')

TTH_1_1_T6_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T6_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T6_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T6_R2.insert(1, 'trial', '6')
TTH_1_1_T6_R1.insert(1, 'trial', '6')
TTH_1_1_T6_R3.insert(1, 'trial', '6')
TTH_1_1_T6_R1.insert(2, 'robot', '1')
TTH_1_1_T6_R2.insert(2, 'robot', '2')
TTH_1_1_T6_R3.insert(2, 'robot', '3')

TTH_1_1_T7_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T7_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T7_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T7_R2.insert(1, 'trial', '7')
TTH_1_1_T7_R1.insert(1, 'trial', '7')
TTH_1_1_T7_R3.insert(1, 'trial', '7')
TTH_1_1_T7_R1.insert(2, 'robot', '1')
TTH_1_1_T7_R2.insert(2, 'robot', '2')
TTH_1_1_T7_R3.insert(2, 'robot', '3')

TTH_1_1_T8_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T8_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T8_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T8_R2.insert(1, 'trial', '8')
TTH_1_1_T8_R1.insert(1, 'trial', '8')
TTH_1_1_T8_R3.insert(1, 'trial', '8')
TTH_1_1_T8_R1.insert(2, 'robot', '1')
TTH_1_1_T8_R2.insert(2, 'robot', '2')
TTH_1_1_T8_R3.insert(2, 'robot', '3')

TTH_1_1_T9_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T9_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T9_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T9_R2.insert(1, 'trial', '9')
TTH_1_1_T9_R1.insert(1, 'trial', '9')
TTH_1_1_T9_R3.insert(1, 'trial', '9')
TTH_1_1_T9_R1.insert(2, 'robot', '1')
TTH_1_1_T9_R2.insert(2, 'robot', '2')
TTH_1_1_T9_R3.insert(2, 'robot', '3')

TTH_1_1_T10_R2.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T10_R1.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T10_R3.insert(0, 'condition', 'tth_1_1')
TTH_1_1_T10_R2.insert(1, 'trial', '10')
TTH_1_1_T10_R1.insert(1, 'trial', '10')
TTH_1_1_T10_R3.insert(1, 'trial', '10')
TTH_1_1_T10_R1.insert(2, 'robot', '1')
TTH_1_1_T10_R2.insert(2, 'robot', '2')
TTH_1_1_T10_R3.insert(2, 'robot', '3')



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
all_df["activation"] = pd.to_numeric(all_df["activation"])
all_df["state"] = pd.to_numeric(all_df["state"])
all_df["cumul_tags"] = pd.to_numeric(all_df["cumul_tags"])

# It looks like you have some NaN's?
# print( "Oops, number of NaN's: ", all_df.isnull().sum().sum() )


# Exclude data based on our min max time values above
all_df = all_df[ ( all_df['time'] >= min_time ) ]
all_df = all_df[ ( all_df['time'] <= max_time ) ]

# We can peek inside each file to
# check formatting like this
print( all_df )





# Results 2 6
subset = all_df[ ( all_df['condition'] == 'tth_2_6' ) ]
#print( subset )


# Set y-axis limits
y_limits = (0, 110)

# having trouble with hue colours
colours = dict(zip(subset['robot'].unique(), sns.color_palette(n_colors=len(subset['robot'].unique()))))

lplot = sns.relplot(data=subset, x='time', y='activation', kind='line', hue='robot', palette=colours, height=5, aspect=2, errorbar=('ci',95) )
lplot.fig.suptitle(f'Activity Level {conditions[2]} {title_extension}', fontsize=16)

# Set y-axis limits
plt.ylim(y_limits)
plt.axhline(y=50, linestyle='--', linewidth=0.8, c='hotpink')

plt.xlabel('Time (s)')
plt.ylabel('Activity Level')


plt.savefig(f'{graph_path}/pngs/{experiment}_{conditions[1]}_AL.png')
plt.savefig(f'{graph_path}/pdfs/{experiment}_{conditions[1]}_AL.pdf')
plt.close()




# Results 6 10
subset = all_df[ ( all_df['condition'] == 'tth_6_10' ) ]
#print( subset )


# Set y-axis limits
y_limits = (0, 110)

# having trouble with hue colours
colours = dict(zip(subset['robot'].unique(), sns.color_palette(n_colors=len(subset['robot'].unique()))))

lplot = sns.relplot(data=subset, x='time', y='activation', kind='line', hue='robot', palette=colours, height=5, aspect=2, errorbar=('ci',95) )
lplot.fig.suptitle(f'Activity Level {conditions[1]} {title_extension}', fontsize=16)

# Set y-axis limits
plt.ylim(y_limits)
plt.axhline(y=50, linestyle='--', linewidth=0.8, c='hotpink')

plt.xlabel('Time (s)')
plt.ylabel('Activity Level')

plt.savefig(f'{graph_path}/pngs/{experiment}_{conditions[2]}_AL.png')
plt.savefig(f'{graph_path}/pdfs/{experiment}_{conditions[2]}_AL.pdf')
plt.close()


# Result 100 200

subset = all_df[ ( all_df['condition'] == 'tth_100_200' ) ]
#print( subset )

# Set y-axis limits
y_limits = (0, 110)

# having trouble with hue colours
colours = dict(zip(subset['robot'].unique(), sns.color_palette(n_colors=len(subset['robot'].unique()))))

lplot = sns.relplot(data=subset, x='time', y='activation', kind='line', hue='robot', palette=colours, height=5, aspect=2, errorbar=('ci',95) )
lplot.fig.suptitle(f'Activity Level {conditions[3]} {title_extension}', fontsize=16)

# Set y-axis limits
plt.ylim(y_limits)
plt.axhline(y=50, linestyle='--', linewidth=0.8, c='hotpink')


plt.xlabel('Time (s)')
plt.ylabel('Activity Level')

plt.savefig(f'{graph_path}/pngs/{experiment}_{conditions[3]}_AL.png')
plt.savefig(f'{graph_path}/pdfs/{experiment}_{conditions[3]}_AL.pdf')
plt.close()

# Result 1 1

subset = all_df[ ( all_df['condition'] == 'tth_1_1' ) ]
#print( subset )

# Set y-axis limits
y_limits = (0, 110)

# having trouble with hue colours
colours = dict(zip(subset['robot'].unique(), sns.color_palette(n_colors=len(subset['robot'].unique()))))

lplot = sns.relplot(data=subset, x='time', y='activation', kind='line', hue='robot', palette=colours, height=5, aspect=2, errorbar=('ci',95) )
lplot.fig.suptitle(f'Activity Level {conditions[3]} {title_extension}', fontsize=16)

# Set y-axis limits
plt.ylim(y_limits)
plt.axhline(y=50, linestyle='--', linewidth=0.8, c='hotpink')


plt.xlabel('Time (s)')
plt.ylabel('Activity Level')

plt.savefig(f'{graph_path}/pngs/{experiment}_{conditions[4]}_AL.png')
plt.savefig(f'{graph_path}/pdfs/{experiment}_{conditions[4]}_AL.pdf')
plt.close()




# Results all compared 
subset = all_df[ ( all_df['time'] < max_time ) ]
subset = all_df[ ( all_df['condition'] != 'NP' ) ]
#print( subset )



# having trouble with hue colours
colours = dict(zip(subset['condition'].unique(), sns.color_palette(n_colors=len(subset['condition'].unique()))))

lplot = sns.relplot(data=subset, x='time', y='activation', kind='line', hue='condition', palette=colours, height=5, aspect=2, errorbar=('ci',95) )
lplot.fig.suptitle(f'Activity Level Over Time {title_extension}', fontsize=16)

plt.axhline(y=50, linestyle='--', linewidth=0.8, c='hotpink')


plt.xlabel('Time (s)')
plt.ylabel('Activity Level')

plt.savefig(f'{graph_path}/pngs/{experiment}_TTH_ALL_AL.png')
plt.savefig(f'{graph_path}/pdfs/{experiment}_TTH_ALL._AL.pdf')
plt.close()






# total tags all conditions all trials

# Define the maximum tag limit
max_cumul_tags = 2000

# Max tags added due to a seemingly anomalous reading of over 8000 for NP in one trial

# Filter data and compute final tag counts for each condition
subset_tth_2_6 = all_df[(all_df['time'] < max_time) & (all_df['condition'] == 'tth_2_6') & (all_df['cumul_tags'] < max_cumul_tags)]
tth_2_6_tags = subset_tth_2_6.groupby(['trial'])['cumul_tags'].max()

subset_tth_6_10 = all_df[(all_df['time'] < max_time) & (all_df['condition'] == 'tth_6_10') & (all_df['cumul_tags'] < max_cumul_tags)]
tth_6_10_tags = subset_tth_6_10.groupby(['trial'])['cumul_tags'].max()

subset_tth_100_200 = all_df[(all_df['time'] < max_time) & (all_df['condition'] == 'tth_100_200') & (all_df['cumul_tags'] < max_cumul_tags)]
tth_100_200_tags = subset_tth_100_200.groupby(['trial'])['cumul_tags'].max()


subset_tth_1_1 = all_df[(all_df['time'] < max_time) & (all_df['condition'] == 'tth_1_1') & (all_df['cumul_tags'] < max_cumul_tags)]
tth_1_1_tags = subset_tth_1_1.groupby(['trial'])['cumul_tags'].max()

subset_np = all_df[(all_df['time'] < max_time) & (all_df['condition'] == 'NP') & (all_df['cumul_tags'] < max_cumul_tags)]
np_tags = subset_np.groupby(['trial'])['cumul_tags'].max()


tags = { 'tth_2_6' : tth_2_6_tags,
        'tth_6_10' : tth_6_10_tags,
        'tth_100_200' : tth_100_200_tags,
        'tth_1_1' : tth_1_1_tags,
         'np' : np_tags,
         }

df = pd.DataFrame( tags )
print(df)

#colours = dict(zip(subset['robot'].unique(), sns.color_palette(n_colors=len(subset['robot'].unique()))))
bplot = sns.boxplot( data=df, width=0.4)
bplot.set_xlabel('Experiment Condition')
bplot.set_ylabel('Final Tag Count')
bplot.set_title(f'Final Tag Count {title_extension}')

plt.savefig(f'{graph_path}/pngs/{experiment}_final_tag_count.png')
plt.savefig(f'{graph_path}/pdfs/{experiment}_final_tag_count.pdf')
plt.close()




# Tags over time
subset = all_df[ ( all_df['time'] < max_time ) ]



# having trouble with hue colours
colours = dict(zip(subset['condition'].unique(), sns.color_palette(n_colors=len(subset['condition'].unique()))))

lplot = sns.relplot(data=subset, x='time', y='cumul_tags', kind='line', hue='condition', palette=colours, height=5, aspect=2, errorbar=('ci',95) )
lplot.fig.suptitle(f'Tag Collection over Time {title_extension}', fontsize=16)

plt.xlabel('Time (s)')
plt.ylabel('Tags Detected')

plt.savefig(f'{graph_path}/pngs/{experiment}_Tags_over_time.png')
plt.savefig(f'{graph_path}/pdfs/{experiment}_Tags_over_time.pdf')
plt.close()
