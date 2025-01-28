import numpy as np
import pandas as pd

def load_data(representation_path, phenotype_labels, phenotype_path):
    latent_data = pd.read_csv(representation_path, sep='\t')
    phenotype_data = pd.read_csv(phenotype_path)

    merged_data = pd.merge(latent_data, phenotype_data, left_on='sample_id', right_on='fpath', how='inner')
    merged_data = merged_data.dropna(subset=phenotype_labels).reset_index(drop=True)

    representations = merged_data.filter(regex='^latent_').values
    phenotypes = merged_data[phenotype_labels]

    return representations, phenotypes

def downsample_data(representations, phenotypes, max_samples=1000):
    if representations.shape[0] > max_samples:
        indices = np.random.choice(representations.shape[0], max_samples, replace=False)
        return representations[indices], phenotypes.iloc[indices]
    return representations, phenotypes

