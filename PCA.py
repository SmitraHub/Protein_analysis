import MDAnalysis as mda
from MDAnalysis.analysis import pca, align
import numpy as np
import pandas as pd
import optparse

parser = optparse.OptionParser("Usage: PCA.py [Options]")
parser.add_option("-f", dest="xtc_file", type='string', help="Input fitted trajectory file contains backbone atoms only in .xtc extension")
parser.add_option("-s", dest="tpr_file", type='string', help="Input File containing topology info in .tpr extension")
parser.add_option("-n", dest="n_comp", type='int', help="No. of Principle Components to save from analysis")
parser.add_option("-c", dest="output_file1", type='string', help="Output File contains cumulative varience in .txt extension")
parser.add_option("-o", dest="output_file2", type='string', help="Output File contains PCs in .csv extension")
(options, args) = parser.parse_args()

###################################################### Create Universe #########################################################

universe = mda.Universe(options.tpr_file ,options.xtc_file)
backbone = universe.select_atoms('all')
n_bb = len(backbone)
#print('There are {} backbone atoms in the analysis'.format(n_bb))

################################################### Principle Component Analysis ###############################################

analysis = pca.PCA(universe, select='all',align=False,n_components=None).run()
#print('Colums are the eigenvectors', analysis.p_components.shape)

np.savetxt(options.output_file1, np.around(analysis.cumulated_variance[:20], 3), delimiter=",") # Save Cumulative varience data

transformed = analysis.transform(backbone, n_components=options.n_comp)
#print(transformed.shape) # Columns are the PCs.
df = pd.DataFrame(np.around(transformed,3),
              columns=['PC{}'.format(k+1) for k in range(options.n_comp)])
df['Time (ns)'] = df.index * universe.trajectory.dt/1000

#################################################### Save results ############################################################

df.to_csv(options.output_file2,header=True, index=False)


########################################### Save motion along principle components ############################################
for i in range(options.n_comp):
    eigen = analysis.p_components[:, i]  # First Column
    pc = transformed[:, i]         # First Column
    projected = np.outer(pc, eigen) + analysis.mean.flatten()
    coordinates = projected.reshape(len(pc), -1, 3)
    proj = mda.Merge(backbone)
    proj.load_new(coordinates, order="fac")

    motion_PC=proj.select_atoms("all")
    motion_PC.write("PC_{}.pdb".format(i+1))

    motion_PC.write("PC_{}.xtc".format(i+1), frames='all')


################################################### Cosine Contents ##########################################################
for j in range(options.n_comp):
    cc = pca.cosine_content(transformed, j)
    print(f"Cosine content for PC{j+1} = {cc:.3f}")

