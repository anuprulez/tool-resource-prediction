# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create a dataframe
rf_scores = [0.96, 0.76, 0.09, -0.18]
rf_scores_hpo = [0.98, 0.79, 0.15, 0.04]
df = pd.DataFrame({'rf_scores': rf_scores, 'rf_scores_hpo': rf_scores_hpo})

my_range = range(1, len(df.index) + 1)

# The horizontal plot is made using the hline function
plt.hlines(y=my_range, xmin=df['rf_scores'], xmax=df['rf_scores_hpo'], color='grey', alpha=0.4)
plt.scatter(df['rf_scores'], my_range, color='skyblue', alpha=1, label='RF Scores')
plt.scatter(df['rf_scores_hpo'], my_range, color='green', alpha=0.4, label='RF Scores with HPO')
plt.legend()

# Add title and axis names
plt.yticks(my_range, ["lofreq_indelqual/2.1.4", "bamleftalign/1.3.1", "vcf2tsv/1.0.0_rc3", "rna_star/2.7.2b"])
plt.title("Comparison of the value 1 and the value 2", loc='left')
plt.xlabel('Value of the variables')
plt.ylabel('Group')

# Show the graph
plt.show()