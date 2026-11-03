import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration for academic/research publication
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5.5), dpi=300)

stages = ['CVMI-1', 'CVMI-2', 'CVMI-3', 'CVMI-4', 'CVMI-5', 'CVMI-6']
counts = [4, 8, 8, 30, 59, 21]

# Professional medical/academic color palette (Deep Slate Blue with soft gradient feel)
colors = ['#34495e', '#34495e', '#2980b9', '#2980b9', '#1b4f72', '#2c3e50']

bars = plt.bar(stages, counts, color=colors, width=0.55, edgecolor='black', linewidth=0.8, alpha=0.9)

# Labels and Title styling
plt.xlabel('Cervical Vertebral Maturation (CVM) Stages', fontsize=13, fontweight='bold', labelpad=10)
plt.ylabel('Number of Samples (Support)', fontsize=13, fontweight='bold', labelpad=10)
plt.title('Dataset Sample Distribution Across CVM Stages', fontsize=15, fontweight='bold', pad=15)

# Axis ticks styling
plt.xticks(fontsize=11, fontweight='semibold')
plt.yticks(fontsize=11)

# Adding data value annotations cleanly on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='#111111')

# Clean grid lines behind the bars
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.ylim(0, 68) # Extra headroom for bar numbers

plt.tight_layout()
plt.savefig('distribution_polished.png', dpi=300)
plt.show()