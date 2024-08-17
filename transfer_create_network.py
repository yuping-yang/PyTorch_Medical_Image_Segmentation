# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#==========================================================================
# Reference: 
# MONAI: https://github.com/Project-MONAI/MONAI)
# MindGlide: https://github.com/MS-PINPOINT/mindGlide)
# Contributor:
# Yuping Yang, UoM, Manchester, yuping.yang@postgrad.manchester.ac.uk
# Arman Eshaghi, UCL, London, a.eshaghi@ucl.ac.uk
# Nils Muhlert, UoM, Manchester, nils.muhlert@manchester.ac.uk
#==========================================================================

import os

import torch
from monai.networks.nets import DynUNet
from task_params import deep_supr_num, patch_size, spacing


def get_kernels_strides(task_id):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = patch_size[task_id], spacing[task_id]
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides


def get_network(properties, task_id, pretrain_path, checkpoint=None, transfer=0):
    n_class = len(properties["labels"])
    in_channels = len(properties["modality"])
    kernels, strides = get_kernels_strides(task_id)

    net = DynUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=deep_supr_num[task_id],
    )

    if checkpoint is not None:
        pretrain_path = os.path.join(pretrain_path, checkpoint)
        if os.path.exists(pretrain_path):

            # transfer == 0 (default):
            # Load the entire pretrained model
            # Fine-tune parameters for all layers
            if transfer == 0:
                net.load_state_dict(torch.load(pretrain_path))

            # transfer == 1 or 2 or 3:
            # Load the entire pretrained model except for the output layers
            if transfer in (1, 2, 3):
                pretrained_net = torch.load(checkpoint)

                # List of the specified keys (output layers) not being loaded from the pretrained model
                unload_pretrained_keys_list = ['output_block', 'deep_supervision_heads', 'super_head']

                # Filter the pretrained state dictionary to exclude the specified layers (output layers)
                load_pretrained_state_dict = {}
                for key, value in pretrained_net.items():
                    if not any(unload_pretrained_key in key for unload_pretrained_key in unload_pretrained_keys_list):
                        load_pretrained_state_dict[key] = value
                # print('transfer_mode = ' + str(transfer) + '; loaded_net_keys: \n', \
                #      list(load_pretrained_state_dict.keys()))

                # Partially load the filtered state dictionary into the model
                net.load_state_dict(load_pretrained_state_dict, strict=False)

                # transfer == 1:
                # Fine-tune parameters for all the loaded layers while training parameters for the output layers
                if transfer == 1:
                    pass  # 'requires_grad = True' by default

                # transfer == 2:
                # Freeze parameters for all the loaded layers while training parameters for the output layers
                if transfer == 2:
                    # Freeze all layers
                    for param in net.parameters():
                        param.requires_grad = False

                    # Unfreeze the specified layers (output layers)
                    for param in net.output_block.parameters():
                        param.requires_grad = True
                    for param in net.deep_supervision_heads.parameters():
                        param.requires_grad = True
                    for param in net.skip_layers.next_layer.next_layer.next_layer.super_head.conv.conv.parameters():
                        param.requires_grad = True
                    for param in net.skip_layers.next_layer.next_layer.super_head.conv.conv.parameters():
                        param.requires_grad = True
                    for param in net.skip_layers.next_layer.super_head.conv.conv.parameters():
                        param.requires_grad = True

                # transfer == 3:
                # Freeze parameters for the input & down-sampling layers while fine-tuning parameters for the ...
                # bottleneck & up-sampling layers and training parameters for the output layers
                if transfer == 3:
                    # List of the specified keys being frozen from the pretrained model
                    freeze_pretrained_keys_list = ['input_block', 'downsamples']

                    # Freeze the specified layers
                    for freeze_pretrained_layer in freeze_pretrained_keys_list:
                        # Dynamically access the model's attributes using getattr
                        layer = getattr(net, freeze_pretrained_layer)
                        for param in layer.parameters():
                            param.requires_grad = False

            # transfer == 4:
            # Load only the input & down-sampling layers
            # Freeze parameters for the loaded input & down-sampling layers while training parameters for all the ...
            # other layers
            if transfer == 4:
                pretrained_net = torch.load(checkpoint)

                # List of the specified keys being loaded from the pretrained model
                load_pretrained_keys_list = ['input_block', 'downsamples']

                # Filter the pretrained state dictionary to only include the specified layers
                load_pretrained_state_dict = {}
                for key, value in pretrained_net.items():
                    if any(load_pretrained_key in key for load_pretrained_key in load_pretrained_keys_list):
                        load_pretrained_state_dict[key] = value
                # print('transfer_mode = ' + str(transfer) + '; loaded_net_keys: \n', \
                #      list(load_pretrained_state_dict.keys()))

                # Partially load the filtered state dictionary into the model
                net.load_state_dict(load_pretrained_state_dict, strict=False)

                # List of the specified keys being frozen from the pretrained model
                freeze_pretrained_keys_list = ['input_block', 'downsamples']

                # Freeze the specified layers
                for freeze_pretrained_layer in freeze_pretrained_keys_list:
                    # Dynamically access the model's attributes using getattr
                    layer = getattr(net, freeze_pretrained_layer)
                    for param in layer.parameters():
                        param.requires_grad = False

            print("pretrained checkpoint: {} loaded".format(pretrain_path))

        else:
            print("no pretrained checkpoint")

    return net
