import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Simulate the data
num_species = 10  # Number of species in the forest

# Randomly generate counts for each species using Poisson distribution
np.random.seed(42)
species_counts = np.random.poisson(10, num_species)

# Total number of animals
total_animals = species_counts.sum()

# Step 2: Calculate the Simpson Index
simpson_index = 1 - sum((species_counts / total_animals) ** 2)

# Step 3: Data Setup for Visualization
species_names = [f"Species {i+1}" for i in range(num_species)]

# Creating a DataFrame for species counts
data = pd.DataFrame({
    'Species': species_names,
    'Count': species_counts
})

# Step 4: Enhanced Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Stacked Bar Chart to show species distribution
sns.barplot(x='Species', y='Count', data=data, palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Species Distribution (Stacked Bar Chart)')
axes[0, 0].set_ylabel('Number of Individuals')
axes[0, 0].set_xticklabels(species_names, rotation=45)

# Pie Chart to show proportions of each species
axes[0, 1].pie(species_counts, labels=species_names, autopct='%1.1f%%', colors=sns.color_palette("viridis", num_species), startangle=90)
axes[0, 1].set_title('Proportional Distribution of Species')

# Diversity Gradient: Visualizing Simpson Index by varying evenness
evenness_values = np.linspace(0.1, 1, 100)  # Simulating diversity gradients
simpson_diversities = 1 - np.array([sum((np.random.poisson(e, num_species) / e.sum()) ** 2) for e in np.random.poisson(evenness_values[:, None], (100, num_species))])

sns.lineplot(x=evenness_values, y=simpson_diversities, ax=axes[1, 0], color='b')
axes[1, 0].set_title('Simpson Index by Evenness')
axes[1, 0].set_xlabel('Evenness of Species Distribution')
axes[1, 0].set_ylabel('Simpson Diversity Index')

# Display Simpson Index Text
axes[1, 1].text(0.5, 0.5, f"Simpson Index: {simpson_index:.4f}", fontsize=15, ha='center', va='center', color='black')
axes[1, 1].set_title('Simpson Index Value')
axes[1, 1].axis('off')

# Add Simpson Index equation in the bottom-right plot
equation_text = r"$D = 1 - \sum \left( \frac{n_i (n_i - 1)}{N (N - 1)} \right)$"
axes[1, 1].text(0.5, 0.75, equation_text, fontsize=14, ha='center', va='center', color='black')

# Adjust layout and show plot
plt.tight_layout()

# Save the figure as a PNG file
fig.savefig('C:/Users/75MAMIRUZZAM/Dropbox/StefAmir/Amir/Fall 2024/adventofcode/simpson_diversity1.png', format='png')

# Show the plot
plt.show()

# Output the Simpson Index
print(f"Simpson Index of Diversity: {simpson_index:.4f}")
