import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

df = pd.read_csv("benchmark_results_bruck3D.csv")
df.columns = df.columns.str.strip() # Strip leading spaces from columns

# Set up the matplotlib figure
data_types = df['Data_Type'].unique()
mpi_ranks = df['MPI_Ranks'].unique()


fig, axes = plt.subplots(len(data_types), len(mpi_ranks), figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle('Bruck Algorithm vs MPI_Alltoall Performance Comparison (3D), on Apple M3 Max', fontsize=16)

for i, dtype in enumerate(data_types):
  for j, ranks in enumerate(mpi_ranks):
      ax = axes[i, j] if len(data_types) > 1 and len(mpi_ranks) > 1 else axes[j]
      
      subset = df[(df['Data_Type'] == dtype) & (df['MPI_Ranks'] == ranks)].sort_values('Grid_Dim')
      
      ax.plot(
          np.array(subset['Grid_Dim']), 
          np.array(subset['Bruck_Time_Seconds']), 
          marker='o', label='Bruck', linestyle='-')
      ax.plot(
          np.array(subset['Grid_Dim']), 
          np.array(subset['MPI_Time_Seconds']), 
          marker='s', label='MPI_Alltoall', linestyle='--')
      
      ax.set_title(f'Type: {dtype}, Ranks: {ranks}')
      ax.set_xscale('log', base=2)
      ax.set_yscale('log', base=10)
      
      ax.grid(True, which="both", ls="--", alpha=0.5)
      
      if i == len(data_types) - 1:
          ax.set_xlabel('Grid Dimension')
      if j == 0:
          ax.set_ylabel('Time (Seconds)')
          
      ax.legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('figure/bruck_vs_mpi_3d.png', dpi=300, facecolor='white')
plt.show()