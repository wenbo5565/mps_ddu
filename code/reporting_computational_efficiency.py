import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r'D:\Research\gw_ddu_mps\result\result_for_reporting.xlsx'
sheet_name = 'Comp Efficiency - Individual'

# Read the specific sheet
df = pd.read_excel(file_path, sheet_name=sheet_name)  # Use header=1 to skip the first row

# Filter rows where column B (Instance) contains "Average"
df_average = df[df['Instance'].str.contains('Average', na=False)]

# Extract the "Number of Scenarios" for the x-axis
x_values = df_average['Number of Scenarios'].tolist()

# Extract the values for S-DDP-I and B-DDP-I
s_ddp_i_values = df_average['S-DDI'].tolist()
b_ddp_i_values = df_average['B-DDI'].tolist()

# Create a figure and axis
fig, ax = plt.subplots(figsize =(12, 10))

# Plot column C (S-DDP-I)
ax.plot(x_values, s_ddp_i_values, color='tab:blue', marker='o', label='S-DDP-I')

# Plot column E (B-DDP-I)
ax.plot(x_values, b_ddp_i_values, color='tab:red', marker='x', label='B-DDP-I')

# Add annotations for S-DDP-I values (rounded to 1 decimal place)
for i, (x, y) in enumerate(zip(x_values, s_ddp_i_values)):
    ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=26)

# Add annotations for B-DDP-I values (rounded to 1 decimal place)
for i, (x, y) in enumerate(zip(x_values, b_ddp_i_values)):
    if i != 1: 
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=26)
    else:
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=26)

# Add labels, title, and legend (font size set to 22)
ax.set_xlabel('Number of Scenarios', fontsize=28)
ax.set_ylabel('Solution Time', fontsize=28)
ax.set_title('Average Solution Time of S-DDI and B-DDI', fontsize=28)
ax.legend(fontsize=28)

# Set tick label font size to 22
ax.tick_params(axis='both', labelsize=28)

# Show the plot
plt.tight_layout()
plt.show()

"""
##################################################
# Joint Chance Constraint Model model
##################################################
"""

sheet_name = 'Comp Efficiency - Joint'

# Read the specific sheet
df = pd.read_excel(file_path, sheet_name=sheet_name)  # Use header=1 to skip the first row

# Filter rows where column B (Instance) contains "Average"
df_average = df[(df['Number of Scenarios'] <= 1000) & (df['Instance'].str.contains('Average', na=False))]

# Extract the "Number of Scenarios" for the x-axis
x_values = df_average['Number of Scenarios'].tolist()

# Extract the values for S-DDP-I and B-DDP-I
s_ddp_i_values = df_average['S-DDJ'].tolist()
b_ddp_i_values = df_average['B-DDJ'].tolist()

# Create a figure and axis
fig, ax = plt.subplots(figsize =(12, 10))

# Plot column C (S-DDP-I)
ax.plot(x_values, s_ddp_i_values, color='tab:blue', marker='o', label='S-DDJ')

# Plot column E (B-DDP-I)
ax.plot(x_values, b_ddp_i_values, color='tab:red', marker='x', label='B-DDJ')

# Add annotations for S-DDP-I values (rounded to 1 decimal place)
for i, (x, y) in enumerate(zip(x_values, s_ddp_i_values)):
    if i != 1:
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=26)
    else:
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=26)

# Add annotations for B-DDP-I values (rounded to 1 decimal place)
for i, (x, y) in enumerate(zip(x_values, b_ddp_i_values)):
    if i != 1: 
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=26)
    else:
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=26)
        
# Add labels, title, and legend (font size set to 22)
ax.set_xlabel('Number of Scenarios', fontsize=28)
ax.set_ylabel('Solution Time', fontsize=28)
ax.set_title('Average Solution Time of S-DDJ and B-DDJ', fontsize=28)
ax.legend(fontsize=28)

# Set tick label font size to 22
ax.tick_params(axis='both', labelsize=28)

# Show the plot
plt.tight_layout()
plt.show()

"""
#################################################
# B-DDP
#################################################
"""



# Load the Excel file

sheet_name = 'Comp Efficiency - Joint'

# Read the specific sheet
df = pd.read_excel(file_path, sheet_name=sheet_name)  # Use header=1 to skip the first row

# Filter rows where column B (Instance) contains "Average"
df_average = df[(df['Number of Scenarios'] >= 1000) & (df['Instance'].str.contains('Average', na=False))]

# Extract the "Number of Scenarios" for the x-axis
x_values = df_average['Number of Scenarios'].tolist()

# Extract the values for S-DDP-I and B-DDP-I
s_ddp_i_values = df_average['B-DDJ'].tolist()
b_ddp_i_values = df_average['B-DDI'].tolist()

# Create a figure and axis
fig, ax = plt.subplots(figsize =(12, 10))

# Plot column C (S-DDP-I)
ax.plot(x_values, s_ddp_i_values, color='tab:blue', marker='o', label='B-DDJ')

# Plot column E (B-DDP-I)
ax.plot(x_values, b_ddp_i_values, color='tab:red', marker='x', label='B-DDI')

# Add annotations for S-DDP-I values (rounded to 1 decimal place)
for i, (x, y) in enumerate(zip(x_values, s_ddp_i_values)):
    if i != 1:
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=26)
    else:
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=26)

# Add annotations for B-DDP-I values (rounded to 1 decimal place)
for i, (x, y) in enumerate(zip(x_values, b_ddp_i_values)):
    if i != 1: 
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, -20), ha='center', fontsize=26)
    else:
        ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=26)
        
# Add labels, title, and legend (font size set to 22)
ax.set_xlabel('Number of Scenarios', fontsize=28)
ax.set_ylabel('Solution Time', fontsize=28)
ax.set_title('Solution Time of B-DDI and B-DDJ', fontsize=28)
ax.legend(fontsize=28)
ax.set_ylim(0, 10)

# Set tick label font size to 22
ax.tick_params(axis='both', labelsize=28)

# Show the plot
plt.tight_layout()
plt.show()

























