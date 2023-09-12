# Imaging pipeline for DHM
 Master Thesis at the Stocker Lab with Dieter Baumgartner

data folder contains DeLTA assets --> All DeLTA relevant data, and Ilastik --> Ilastik files and training data.
graphes_images contrains scripts for graphs, data for graphs and graphs themself.
src contains all code
tests contain data and graphs produced for tests.

Workflow process holograms:
Run Imaging-pipeline-for-DHM//src//run_dhm_pipeline.py

Workflow results interpretation:
The output files are extended Delta files they can be handeled as such. All possibilities and good documentation can be found https://gitlab.com/dunloplab/delta/-/tree/main/docs.
Cell cycles can be processed using function showed in Imaging-pipeline-for-DHM//graphes_images//src//cell_cycle_plots

Workflow produce training images:
1. Load time-lapse images into fiji.
2. Save them as .h5 in Imaging-pipeline-for-DHM//data//ilastik//'experiment_name'//'experiment_name'_'position_name'.h5
3. Repeat 1 and 2 for each position
4. Do an ilastik pixel classification workflow of this experiment. Save both the Ilastik workflow and probabilities in this folder.
5. Do an ilastik tracking workflow of this experiment. Save the Ilastik workflow, object identities and csv files with tracking results in this folder.
6. Copy csv files into Imaging-pipeline-for-DHM\data\ilastik\all
7. Run Imaging-pipeline-for-DHM//scr//ilastik//visualize_ilastik_images too look at the Results and determine at which frame the Ilastik workflow stopped working properly. Save images until this frame (explained in the last block of the code). Now the relevant images are saved in Imaging-pipeline-for-DHM\data\ilastik\all
8. Produce DeLTA training images. This can be done with one of two scripts. One script erods the outermost 2 layers (Imaging-pipeline-for-DHM//src//Ilastik//create_delta_train_images) the other erodes all but a core(Imaging-pipeline-for-DHM//src//Ilastik//create_delta_train_images_core). They do have very different weighting. The one producing the core can separete the bacteria better but has the overall worse performance. So Imaging-pipeline-for-DHM//src//Ilastik//create_delta_train_images is recommended. This saves all relevant images into Imaging-pipeline-for-DHM//data//ilastik//delta.
9. Copy the two folder in Imaging-pipeline-for-DHM//data//ilastik//delta into Imaging-pipeline-for-DHM\data\delta_assets\trainingsets\2D\training replacing the old training images.
10. Delete the old validation folders in Imaging-pipeline-for-DHM//data//delta_assets//trainingsets//2D//validation
11. Run Imaging-pipeline-for-DHM//src//delta//rename_to_val_data to make a validation split. New training images are done.

Workflow DeLTA U-Net training:
Segmentation and tracking U-Nets are trained induvidually.
Run Imaging-pipeline-for-DHM//src//delta//train_seg for training the segmentation net.
Run Run Imaging-pipeline-for-DHM//src//delta//train_track for the training of the tracking net.













 