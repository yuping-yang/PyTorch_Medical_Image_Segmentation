# These codes are used for generating brain MRI segmentation models based on the pretrained model from MindGlide
(https://github.com/MS-PINPOINT/mindGlide)
#==========================================================================
# Reference: 
# MONAI: https://github.com/Project-MONAI/MONAI)
# MindGlide: https://github.com/MS-PINPOINT/mindGlide)
# Contributor:
# Yuping Yang, UoM, Manchester, yuping.yang@postgrad.manchester.ac.uk
# Arman Eshaghi, UCL, London, a.eshaghi@ucl.ac.uk
# Nils Muhlert, UoM, Manchester, nils.muhlert@manchester.ac.uk
#==========================================================================

Step 1: Prepare your toolbox, code and data
        Toolbox: MindGlide, MONAI dynunet package, Singularity (Apptainer)
        Code: in /your_container_directory/dynunet/modules/dynunet_pipeline
                update 'transfer_train.py', 'transfer_create_network.py' from the attached files
                update 'create_datalist.py', 'task_params.py' from MindGlide
        Data: in /your_container_directory/Data_Segmentation/Task12_brain
                update 'imagesTr', 'imagesTs', 'labelsTr' according to your data
                update 'dataset.json' according to your data

Step 2: Create the 'dataset_task12.json' file
        Run the following codes in your bash terminal:
        python /your_container_directory/dynunet/modules/dynunet_pipeline/create_datalist.py \
            -input_dir '/your_container_directory/Data_Segmentation' \
            -output_dir '/your_container_directory/Data_Segmentation/MONAI_Output' \
            -task_id 12 -num_folds 5 -seed 12345

Step 3: Train your specific model based on the pretrained model from MindGlide
        Run the following codes in your bash terminal:
        cd /your_container_directory
        singularity exec --nv --bind $PWD:/mnt_Singularity /e/mind-glide_latest.sif bash
            python /mnt_Singularity/dynunet/modules/dynunet_pipeline/transfer_train.py \
            -train_num_workers 4 -interval 1 -num_samples 3 \
            -task_id 12 -root_dir '/mnt_Singularity/Data_Segmentation' \
            -learning_rate 0.01 \
            -max_epochs 10 \
            -pos_sample_num 2 -expr_name 'MindGlide_Transfer1_PretrainedLabel20_OutputLabel25' \
            -tta_val False -datalist_path '/mnt_Singularity/Data_Segmentation/MONAI_Output' \
            -fold 0 \
            -checkpoint '/e/mindGlide/models/model_5_net_key_metric=0.7866.pt' \
            -mode train \
            -transfer 1


