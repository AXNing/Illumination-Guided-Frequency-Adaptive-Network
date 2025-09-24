# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# # Sample data
# data = {
#     'Method': ['Ours', 'SNRNet', 'NeRco', 'SMG-LLIE', 'MBLLVEN', 'SMOID', 'SMID', 'StableLLVE','SDSDNet', 'LLVE-SEG', 'DP3DF', 'FASTLLVE'],
#     'PSNR': [31.17, 25.65, 23, 26.04, 24.82, 22.57, 22.97, 21.64, 21.88, 24.85, 25.39, 29.34],
#     'Inference Time': [0.081, 0.201, 2.904, 0.362, 0.366, 0.197, 0.131, 0.037, 0.401, 0.165, 0.182, 0.078],
#     'MarkerSize': [200, 150, 150, 100, 120, 100, 80, 60, 90, 110, 130, 140],  # Adjusted size
#     'Color': sns.color_palette("husl", 12)  # Use a color palette
# }

# df = pd.DataFrame(data)

# # Create scatter plot
# plt.figure(figsize=(12, 8))
# scatter = plt.scatter(
#     x=df['Inference Time'],
#     y=df['PSNR'],
#     s=df['MarkerSize'],  # Marker size
#     c=[color for color in df['Color']],  # Ensure correct color assignment
#     alpha=0.7,           # Transparency
#     edgecolor='black'    # Edge color
# )

# # Add text labels
# for i, txt in enumerate(df['Method']):
#     plt.annotate(txt, (df['Inference Time'][i], df['PSNR'][i]), fontsize=15, ha='right', va='bottom')

# # Log scale for x-axis
# plt.xscale('log')

# # Customize plot aesthetics
# plt.title('Performance vs. Efficiency', fontsize=20)
# plt.xlabel('Inference Time (s)', fontsize=16)
# plt.ylabel('PSNR (dB)', fontsize=16)
# plt.grid(True, which="both", ls="--")
# plt.tight_layout()
# plt.savefig('performance_vs_efficiency.png')
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data
data = {
    'Method': [ 'Zhou(2023)', 'BracketFlare(2023)', 'Flare7k(2022)', 'Flare7k++(2024)','Kotp(2024)', 'Ours'],
    'G-PSNR': [23.779,24.573,23.981,24.724,24.879,25.44],
    'S-PSNR': [22.237, 23.682, 24.365, 24.188,24.458,25.018],
    'MarkerSize': [400, 400, 400, 400, 400,800],  # Adjusted size
    'Color': sns.color_palette("Set3", 6),  # Use a color palette
    'Marker': ['o', 'D', 'p', 'h', 'H','*']  # Circles, diamonds, and polygons
}






df = pd.DataFrame(data)

# Create scatter plot
plt.figure(figsize=(12, 6))

# Plot scatter points
for i in range(len(df)):
    plt.scatter(
        x=df['S-PSNR'][i],
        y=df['G-PSNR'][i],
        s=df['MarkerSize'][i],  # Marker size
        c=[df['Color'][i]],     # Marker color
        alpha=0.7,             # Transparency
        edgecolor='black',     # Edge color
        marker=df['Marker'][i] # Marker shape
    )

# Add text labels with some offset
for i, txt in enumerate(df['Method']):
    if txt=='1' or txt=='Zhou(2023)':
        plt.annotate(txt, 
                    (df['S-PSNR'][i] + 0.005, df['G-PSNR'][i] + 0.1),  # Offset the label by 0.005 on x-axis and 0.1 on y-axis
                    fontsize=16, 
                    ha='left',  # Horizontal alignment: left
                    va='bottom' # Vertical alignment: bottom
                    )
    
    elif txt == 'Ours':
        plt.annotate(txt, 
                    (df['S-PSNR'][i] - 0.001, df['G-PSNR'][i] - 0.05),  # Offset the label by 0.005 on x-axis and 0.1 on y-axis
                    fontsize=16, 
                    ha='right',  # Horizontal alignment: left
                    va='top', # Vertical alignment: bottom
                    color='black'
                    )
            
    
    
    else:
        plt.annotate(txt, 
                    (df['S-PSNR'][i] + 0.005, df['G-PSNR'][i] + 0.1),  # Offset the label by 0.005 on x-axis and 0.1 on y-axis
                    fontsize=16, 
                    ha='right',  # Horizontal alignment: left
                    va='bottom' # Vertical alignment: bottom
                    )

# Customize plot aesthetics
plt.xlabel('S-PSNR (dB)', fontsize=16,fontname='Serif')
plt.ylabel('G-PSNR (dB)', fontsize=16,fontname='Serif')

# Set x-axis ticks at 0.1 increments
#plt.xticks([i/100 for i in range(0, 46, 5)], fontsize=16)  # From 0 to 0.45 with increments of 0.05
plt.yticks(fontsize=16)  # Ensure y-axis ticks are also properly sized

# Add gridlines with dashed lines
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# Customize the appearance of major and minor ticks
plt.tick_params(axis='both', which='both', direction='in', length=6, width=1)

# Save and show plot
plt.tight_layout()
plt.savefig('/home/ubuntu/axproject/Flare7K/comparsionmetrics_new.png', dpi=300)  # Save with high resolution
plt.show()
