from Bio import motifs
from Bio.motifs import jaspar
import xgboost as xgb

with open("data/JASPAR2024_CORE_redundant_pfms_jaspar.txt") as handle:
    motif_list = motifs.parse(handle, "jaspar")

all_consensus = []
for motif in motif_list:
    # print("Consensus sequence:", motif.consensus)
    all_consensus.append(str(motif.consensus))

# Create an instance of the XGBCassifier (or your specific model type)
loaded_model = xgb.XGBClassifier()

# Load the model from the file
loaded_model.load_model('xgbclass_submission.json')

from betamark import ocr
import pandas as pd
import numpy as np

def placeholder(x):
    """
    Params:
    -------
    x: string representing a genomic sequence

    Returns:
    --------
    y_pred: int where 0 is negative (not an OCR) or 1 (is an OCR)
    """
    
    # extract as many features as we can think of
    df = pd.DataFrame(index=[0])

    # number of nucleotides
    df['length'] = len(x)

    # % of A, C, T and G's (and N's)

    #df['n_N'] = x.count("N") / df['length'] # this is not needed, it is implied by the other columns

    df['cpg_ratio'] = np.where(
        (x.count('C') * x.count('G')) > 0,
        (x.count('CG') * df['length']) / (x.count('C') * x.count('G')),
        0
    )
    
    # Check if any motif is in x
    df['contains_motif'] = [int(any(substring in x for substring in all_consensus))]

    # Check for stop codons
    df['contains_stop'] = [int(any(substring in x for substring in ['TAG', 'TAA', 'TGA']))]
    
    # GC content
    df['GC_content'] = [(x.count("G") + x.count("C")) / df['length'][0]]
    
    # Count of motifs in x
    df['motif_count'] = [sum(x.count(motif) for motif in all_consensus)]

    # Nucleotide frequencies
    df['n_A'] = [x.count("A") / df['length'][0]]
    df['n_C'] = [x.count("C") / df['length'][0]]
    df['n_T'] = [x.count("T") / df['length'][0]]
    df['n_G'] = [x.count("G") / df['length'][0]]

    # motifs between positive sequences? (k-mers?)
    print(df.head(1))
    
    result = loaded_model.predict(df)
    return(result[0])

print(ocr.run_validation(user_func=placeholder))