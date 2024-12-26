from Preprocessing import vulnerability_df, malware_df
from Vectorization_BERT import malware_embeddings, vulnerability_embeddings
from torch.nn.functional import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Compute cosine similarity matrix
similarity_matrix = cosine_similarity(
    malware_embeddings.unsqueeze(1),
    vulnerability_embeddings.unsqueeze(0),
    dim=-1
)

# Convert similarity matrix to DataFrame
malware_classes = malware_df['malware_family'].to_list()
malware_families = malware_df['malware_family'].unique().tolist()
vulnerability_classes = vulnerability_df['vulnerability'].to_list()
similarity_df = pd.DataFrame(similarity_matrix.numpy(), index=malware_classes, columns=vulnerability_classes)

# Step 2: Aggregate similarity scores by malware family
aggregated_similarity = similarity_df.groupby(similarity_df.index).mean()

# Step 3: Assign each vulnerability to the most similar malware family
unique_mapping = aggregated_similarity.idxmax(axis=0)  # Find the family with the highest similarity
mapped_similarity = aggregated_similarity.max(axis=0)  # Maximum similarity value

# Create a DataFrame for unique mappings
mapping_df = pd.DataFrame({
    'Vulnerability': vulnerability_classes,
    'Mapped Malware Family': unique_mapping.values,
    'Similarity': mapped_similarity.values
})

# Step 4: Display unique mappings
print(mapping_df)

# Step 5: Plot the heatmap for aggregated similarities
plt.figure(figsize=(10, 8))
sns.heatmap(
    aggregated_similarity,
    cmap='viridis',
    annot=False,
    cbar=True,
    square=True,
    cbar_kws={'label': 'Cosine Similarity'}
)

plt.title("Malware Family to Vulnerability Mapping")
plt.xlabel("Vulnerability Classes")
plt.ylabel("Malware Families")
plt.xticks(rotation=45, ha='right', fontsize = 6)
plt.yticks(rotation=0, fontsize = 8)
plt.show()





