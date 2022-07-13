#Import necessary modules from packages
import os
import matplotlib.pyplot as plt
import nilearn
from nilearn.reporting import get_clusters_table
from nilearn.image import math_img
from nilearn.plotting import plot_stat_map
from nimare import io, utils
from nimare.correct import FDRCorrector, FWECorrector
from nimare.dataset import Dataset
from nimare.meta import ALE

#Set paths for input and output folders
inData = "Input_Data/"
outData = "Output/"

#Import data from sleuth text files
spatial_dset = io.convert_sleuth_to_dataset(os.path.join(inData+'Spatial_PT_all.txt'))
spatial2_dset = io.convert_sleuth_to_dataset(os.path.join(inData+"Spatial_PT_lvl2.txt"))
cognitive_dset = io.convert_sleuth_to_dataset(os.path.join(inData+"Cognitive_PT.txt"))
affective_dset = io.convert_sleuth_to_dataset(os.path.join(inData+"Affective_PT.txt"))
attention_dset = io.convert_sleuth_to_dataset(os.path.join(inData+"Attention.txt"))

#Set parameters for ALE
ale = ALE(null_method="approximate")

#Perform individual domain ALEs (this will take a while to run)
spatial_results = ale.fit(spatial_dset)
spatial2_results = ale.fit(spatial2_dset)
cognitive_results = ale.fit(cognitive_dset)
affective_results = ale.fit(affective_dset)
attention_results = ale.fit(attention_dset)

#Set parameters for FWE correction
corr = FWECorrector(method='montecarlo', voxel_thresh=0.001, n_iters=10000, n_cores=7)

#Correct ALE for multiple comparisons
spatial_corrected_results = corr.transform(spatial_results)
spatial2_corrected_results = corr.transform(spatial2_results)
cognitive_corrected_results = corr.transform(cognitive_results)
affective_corrected_results = corr.transform(affective_results)
attention_corrected_results = corr.transform(attention_results)

#Save the resulting maps to output folder
spatial_corrected_results.save_maps(prefix="all_spatial_005", prefix_sep="_", output_dir= outData)
spatial2_corrected_results.save_maps(prefix="lvl2_spatial_005", prefix_sep="_", output_dir= outData)
cognitive_corrected_results.save_maps(prefix="all_cognitive_005", prefix_sep="_", output_dir= outData)
affective_corrected_results.save_maps(prefix="all_affective_005", prefix_sep="_", output_dir= outData)
attention_corrected_results.save_maps(prefix="all_attention_005", prefix_sep="_", output_dir= outData)

#Generate cluster tables
clust_all_Spatial = nilearn.reporting.get_clusters_table("all_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", 1.645)
clust_lv2_Spatial = nilearn.reporting.get_clusters_table("lvl2_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", 1.645)
clust_all_Cognitive = nilearn.reporting.get_clusters_table("all_cognitive_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", 1.645)
clust_all_Affective = nilearn.reporting.get_clusters_table("all_affective_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", 1.645)
clust_all_Attention = nilearn.reporting.get_clusters_table("all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", 1.645)

#Write cluster tables to .csv
clust_all_Spatial.to_csv(path_or_buf= os.path.join(outData+"cluster_table_all_Spatial.csv"), sep=',')
clust_lv2_Spatial.to_csv(path_or_buf= os.path.join(outData+"cluster_table_lv2_Spatial.csv"), sep=',')
clust_all_Cognitive.to_csv(path_or_buf= os.path.join(outData+"cluster_table_all_Cognitive.csv"), sep=',')
clust_all_Affective.to_csv(path_or_buf= os.path.join(outData+"cluster_table_all_Affective.csv"), sep=',')
clust_all_Attention.to_csv(path_or_buf= os.path.join(outData+"cluster_table_all_Attention.csv"), sep=',') 


#Initialize the counter function
counter = FocusCounter(
    target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
    voxel_thresh=None,
)

#Focus-count analysis - characterize relative contributions of experiments in ALE
spatial_count_table, _ = counter.transform(spatial_corrected_results)
spatial2_count_table, _ = counter.transform(spatial2_corrected_results)
cognitive_count_table, _ = counter.transform(cognitive_corrected_results)
affective_count_table, _ = counter.transform(affective_corrected_results)
attention_count_table, _ = counter.transform(attention_corrected_results)

#Write count tables to .csv
spatial_count_table.to_csv(path_or_buf= os.path.join(outData+"focusCount_table_all_Spatial.csv"), sep=',')
spatial2_count_table.to_csv(path_or_buf= os.path.join(outData+"focusCount_table_lv2_Spatial.csv"), sep=',')
cognitive_count_table.to_csv(path_or_buf= os.path.join(outData+"focusCount_table_all_Cognitive.csv"), sep=',') 
affective_count_table.to_csv(path_or_buf= os.path.join(outData+"focusCount_table_all_Affective.csv"), sep=',') 
attention_count_table.to_csv(path_or_buf= os.path.join(outData+"focusCount_table_all_Attention.csv"), sep=',')

#Initialize formula for conjunction function
formula = "np.where(img * img2 > 0, np.minimum(img, img2), 0)"

#Conjunctions: spatial PT, cognitive PT, affective PT
conj_allSpatial_x_Cognitive = math_img(formula, img= "all_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "all_cognitive_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")
conj_allSpatial_x_Affective = math_img(formula, img="all_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2="all_affective_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")
conj_lvl2Spatial_x_Cognitive = math_img(formula, img="lvl2_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2="all_cognitive_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")
conj_lvl2Spatial_x_Affective = math_img(formula, img="lvl2_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2="all_affective_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")
conj_Cognitive_x_Affective = math_img(formula, img="all_cognitive_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2="all_affective_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")

#Conjunctions: attention x PT
conj_Attention_x_allSpatial = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "all_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")
conj_Attention_x_lvl2Spatial = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "lvl2_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")
conj_Attention_x_Cognitive = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "all_cognitive_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")
conj_Attention_x_Affective = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "all_affective_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")

#Conjunctions: ATTN x [COG x AFF]
conj_Attention_x_Cognitive_x_Affective = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "conj_Cognitive_x_Affective.nii.gz")

#Conjunctions: ATTN x [COG x SPAT]
conj_Attention_x_Cognitive_x_Spatial = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "conj_Cognitive_x_Spatial.nii.gz")
conj_Attention_x_Cognitive_x_lvl2Spatial = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "conj_Cognitive_x_lvl2Spatial.nii.gz")

#Conjunctions: ATTN x [AFF x SPAT]
conj_Attention_x_Affective_x_Spatial = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "conj_Affective_x_Spatial.nii.gz")
conj_Attention_x_Affective_x_lvl2Spatial = math_img(formula, img= "all_attention_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz", img2= "conj_Affective_x_lvl2Spatial.nii.gz")

#Conjunctions: [ATTN x COG x AFF] x SPAT
conj_Attention_x_Cognitive_x_Affective_x_allSpatial = math_img(formula, img= "conj_Attention_x_Cognitive_x_Affective.nii.gz", img2= "all_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")
conj_Attention_x_Cognitive_x_Affective_x_lvl2Spatial = math_img(formula, img= "conj_Attention_x_Cognitive_x_Affective.nii.gz", img2= "lvl2_spatial_005_z_desc-size_level-cluster_corr-FWE_method-montecarlo.nii.gz")

#Conjunctions: [ATTN x SPAT] x [COG x AFF] 
conj_Attention_x_allSpatial_x_Affective_x_Cognitive = math_img(formula, img= "conj_Attention_x_allSpatial.nii.gz", img2= "conj_Cognitive_x_Affective.nii.gz")
conj_Attention_x_lvl2Spatial_x_Affective_x_Cognitive = math_img(formula, img= "conj_Attention_x_lvl2Spatial.nii.gz", img2= "conj_Cognitive_x_Affective.nii.gz")

#Save images to file
conj_allSpatial_x_Cognitive.to_filename('conj_allSpatial_x_Cognitive.nii.gz')
conj_allSpatial_x_Affective.to_filename('conj_allSpatial_x_Affective.nii.gz')
conj_Cognitive_x_Affective.to_filename('conj_Cognitive_x_Affective.nii.gz')
conj_lvl2Spatial_x_Cognitive.to_filename('conj_lvl2Spatial_x_Cognitive.nii.gz')
conj_lvl2Spatial_x_Affective.to_filename('conj_lvl2Spatial_x_Affective.nii.gz')
conj_Attention_x_allSpatial.to_filename('conj_Attention_x_allSpatial.nii.gz')
conj_Attention_x_lvl2Spatial.to_filename('conj_Attention_x_lvl2Spatial.nii.gz')
conj_Attention_x_Cognitive.to_filename('conj_Attention_x_Cognitive.nii.gz')
conj_Attention_x_Affective.to_filename('conj_Attention_x_Affective.nii.gz')
conj_Attention_x_Cognitive_x_Spatial.to_filename('conj_Attention_x_Cognitive_x_Spatial.nii.gz')
conj_Attention_x_Cognitive_x_lvl2Spatial.to_filename('conj_Attention_x_Cognitive_x_lvl2Spatial.nii.gz')



conj_Attention_x_Cognitive_x_Affective.to_filename('conj_Attention_x_Cognitive_x_Affective.nii.gz')
conj_Attention_x_Cognitive_x_Affective_x_allSpatial.to_filename('conj_Attention_x_Cognitive_x_Affective_x_allSpatial.nii.gz')
conj_Attention_x_Cognitive_x_Affective_x_lvl2Spatial.to_filename('conj_Attention_x_Cognitive_x_Affective_x_lvl2Spatial.nii.gz')
conj_Attention_x_allSpatial_x_Affective_x_Cognitive.to_filename('conj_Attention_x_allSpatial_x_Affective_x_Cognitive.nii.gz')
conj_Attention_x_lvl2Spatial_x_Affective_x_Cognitive.to_filename('conj_Attention_x_lvl2Spatial_x_Affective_x_Cognitive.nii.gz')