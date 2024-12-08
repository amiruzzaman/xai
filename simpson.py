import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Simulate the data
# Number of species in the forest
num_species = 10

# Randomly generate counts for each species
np.random.seed(42)
species_counts = np.random.poisson(10, num_species)

# Total number of animals
total_animals = species_counts.sum()

# Step 2: Calculate the Simpson Index
simpson_index = 1 - sum((species_counts / total_animals) ** 2)

# Step 3: Visualization
# Creating a bar plot to show the counts of each species
species_names = [f"Species {i+1}" for i in range(num_species)]

# Create a dataframe for plotting
data = pd.DataFrame({
    'Species': species_names,
    'Count': species_counts
})

# Plotting the distribution of animals among species
plt.figure(figsize=(10, 6))
sns.barplot(x='Species', y='Count', data=data, palette='viridis')
plt.title('Animal Distribution Among Species in the Forest')
plt.xlabel('Species')
plt.ylabel('Number of Individuals')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Output the Simpson Index
print(f"Simpson Index of Diversity: {simpson_index:.4f}")
