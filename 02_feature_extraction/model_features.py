import torch
import numpy as np
import pandas as pd
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    SpatialPadd,
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
import os
import json
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class MultiLevelUNetFeatureExtractor(torch.nn.Module):
    """Extract features from multiple UNet encoder levels for 1-channel model"""
    
    def __init__(self, checkpoint_path, extract_levels=[0, 1, 2, 3, 4], extract_bottleneck=False, debug=False):
        super().__init__()
        self.extract_levels = extract_levels
        self.extract_bottleneck = extract_bottleneck
        self.debug = debug
        
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(32, 64, 128, 256, 320, 320),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.15
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model_state_dict = {k.replace('_model.', ''): v 
                              for k, v in state_dict.items() 
                              if k.startswith('_model.')}
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        self.multi_level_features = {level: None for level in extract_levels}
        if extract_bottleneck:
            self.multi_level_features['bottleneck'] = None
        self.pooling_type = 'max'
        
        self._register_hooks()
        
    def _register_hooks(self):
        print(f"\n=== FINDING ALL ENCODER OUTPUTS ===")
        
        all_outputs = []
        all_hooks = []
        
        def make_capture_hook(name):
            def hook(m, inp, out):
                if isinstance(out, torch.Tensor) and out.dim() == 5:
                    all_outputs.append({
                        'name': name,
                        'shape': tuple(out.shape),
                        'channels': out.shape[1]
                    })
            return hook
        
        def register_recursive(module, path=""):
            h = module.register_forward_hook(make_capture_hook(path))
            all_hooks.append(h)
            
            if hasattr(module, 'submodule'):
                if hasattr(module.submodule, '__len__'):
                    for i in range(len(module.submodule)):
                        register_recursive(module.submodule[i], f"{path}.submodule[{i}]")
                else:
                    register_recursive(module.submodule, f"{path}.submodule")
            
            if hasattr(module, '_modules'):
                for name, child in module._modules.items():
                    if child is not None and name != 'submodule':
                        register_recursive(child, f"{path}.{name}")
        
        for i in range(3):
            register_recursive(self.model.model[i], f"model[{i}]")
        
        print("Running test forward pass to capture all layer outputs...")
        dummy_input = torch.randn(1, 1, 64, 64, 32).to(next(self.model.parameters()).device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        for h in all_hooks:
            h.remove()
        
        # Deduplicate by path name to preserve all unique layers
        print("\n=== ALL UNIQUE OUTPUTS ===")
        seen_paths = set()
        encoder_outputs = []
        
        for output in all_outputs:
            if output['name'] not in seen_paths and output['channels'] in [32, 64, 128, 256, 320]:
                seen_paths.add(output['name'])
                encoder_outputs.append(output)
                print(f"{output['name']}: channels={output['channels']}, spatial={output['shape'][2:]}")
        
        # Sort by channels desc, then path length asc (shorter = earlier in encoder)
        encoder_outputs.sort(key=lambda x: (-x['channels'], len(x['name'])))
        
        print("\n=== ENCODER PROGRESSION (sorted by channels) ===")
        channel_to_path = {}
        for output in encoder_outputs:
            ch = output['channels']
            if ch not in channel_to_path:
                channel_to_path[ch] = []
            channel_to_path[ch].append(output['name'])
            print(f"{ch}ch (index {len(channel_to_path[ch])-1}): {output['name']}")

        print("\nDEBUG channel_to_path:")
        for ch, paths in channel_to_path.items():
            print(f"  {ch}ch -> {len(paths)} path(s):")
            for p in paths:
                print(f"    {p}")

        print("\n=== REGISTERING HOOKS FOR FEATURE EXTRACTION ===")

        # Map each logical encoder level to its channel count and which index
        # within that channel group to use.
        #
        # channels=(32, 64, 128, 256, 320, 320) — 6 stages total.
        # Levels 4 and 5 both use 320 channels; they are distinguished by
        # ch_index:
        #   index 0 → submodule[0]  = last encoder block  (level 4)
        #   index 3 → submodule[1].submodule = bottleneck block (level 5)
        level_to_channels      = {0: 32, 1: 64, 2: 128, 3: 256, 4: 320, 5: 320}
        level_to_channel_index = {0: 0,  1: 0,  2: 0,   3: 0,   4: 0,   5: 3}

        for level in self.extract_levels:
            expected_channels = level_to_channels.get(level)
            ch_index          = level_to_channel_index.get(level, 0)

            if expected_channels is None:
                print(f"⚠ Level {level} has no channel mapping!")
                continue

            paths = channel_to_path.get(expected_channels, [])
            if ch_index >= len(paths):
                print(f"⚠ Level {level} ({expected_channels}ch, index {ch_index}) not found! "
                      f"Only {len(paths)} path(s) available.")
                continue

            path   = paths[ch_index]
            module = self.model

            for part in path.split('.'):
                if '[' in part:
                    name, idx = part.replace(']', '').split('[')
                    if name:
                        module = getattr(module, name)
                    module = module[int(idx)]
                else:
                    module = getattr(module, part)

            def make_hook(lv):
                def hook_fn(module, input, output):
                    if isinstance(output, torch.Tensor):
                        feat = output.detach()
                        # Collapse sliding-window batch dim → [1, C, H, W, D]
                        if self.pooling_type == 'max':
                            feat = torch.max(feat, dim=0, keepdim=True)[0]
                        else:
                            feat = torch.mean(feat, dim=0, keepdim=True)

                        if self.multi_level_features[lv] is None:
                            self.multi_level_features[lv] = feat
                        else:
                            if self.pooling_type == 'max':
                                self.multi_level_features[lv] = torch.maximum(
                                    self.multi_level_features[lv], feat
                                )
                            else:
                                self.multi_level_features[lv] = (
                                    self.multi_level_features[lv] + feat
                                ) / 2

                        if self.debug:
                            print(f"  Level {lv}: captured shape {feat.shape}, "
                                  f"channels {feat.shape[1]}")
                return hook_fn

            module.register_forward_hook(make_hook(level))
            print(f"✓ Level {level} ({expected_channels}ch, index {ch_index}): {path}")

        if self.extract_bottleneck:
            paths = channel_to_path.get(320, [])
            # index 3 is model[1].submodule[1].submodule[1].submodule[1]
            #           .submodule[1].submodule  — the true bottleneck ResBlock
            bottleneck_index = 3
            if len(paths) > bottleneck_index:
                path   = paths[bottleneck_index]
                module = self.model

                for part in path.split('.'):
                    if '[' in part:
                        name, idx = part.replace(']', '').split('[')
                        if name:
                            module = getattr(module, name)
                        module = module[int(idx)]
                    else:
                        module = getattr(module, part)

                def bottleneck_hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        feat = output.detach()
                        if self.pooling_type == 'max':
                            feat = torch.max(feat, dim=0, keepdim=True)[0]
                        else:
                            feat = torch.mean(feat, dim=0, keepdim=True)

                        if self.multi_level_features['bottleneck'] is None:
                            self.multi_level_features['bottleneck'] = feat
                        else:
                            if self.pooling_type == 'max':
                                self.multi_level_features['bottleneck'] = torch.maximum(
                                    self.multi_level_features['bottleneck'], feat
                                )
                            else:
                                self.multi_level_features['bottleneck'] = (
                                    self.multi_level_features['bottleneck'] + feat
                                ) / 2

                module.register_forward_hook(bottleneck_hook)
                print(f"✓ Bottleneck (320ch, index {bottleneck_index}): {path}")
            else:
                print(f"⚠ No 320ch path at index {bottleneck_index} for bottleneck!")

        print("=== Hook registration complete ===\n")

    def forward(self, x):
        return self.model(x)

    def reset_features(self):
        for level in self.extract_levels:
            self.multi_level_features[level] = None
        if self.extract_bottleneck:
            self.multi_level_features['bottleneck'] = None
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Main extraction function                                                    #
# --------------------------------------------------------------------------- #

def extract_multilevel_features(
    checkpoint_path,
    data_list,
    save_dir,
    extract_levels=[0, 1, 2, 3, 4],
    extract_bottleneck=False,
    pool_features=True,
    pooling_type='max',
    debug=False
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = MultiLevelUNetFeatureExtractor(
        checkpoint_path,
        extract_levels=extract_levels,
        extract_bottleneck=extract_bottleneck,
        debug=debug
    ).to(device)
    feature_extractor.pooling_type = pooling_type

    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.644, 0.644, 1.2), mode="bilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        SpatialPadd(keys=["image"], spatial_size=[192, 192, 64], mode="constant"),
    ])

    dataset    = Dataset(data=data_list, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=False)

    all_features_by_level = {level: [] for level in extract_levels}
    if extract_bottleneck:
        all_features_by_level['bottleneck'] = []
    all_filenames = []

    print(f"\nExtracting features from {len(dataloader)} images...")
    print(f"Encoder levels : {extract_levels}")
    print(f"Bottleneck     : {extract_bottleneck}")
    print(f"Pooling        : {pool_features} ({pooling_type})")

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            torch.cuda.empty_cache()
            gc.collect()

            image = batch["image"].to(device)
            feature_extractor.reset_features()

            roi_size      = [192, 192, 64]
            sw_batch_size = 4

            try:
                _ = sliding_window_inference(
                    image, roi_size, sw_batch_size, feature_extractor, overlap=0.25
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM on image {idx}, retrying with batch_size=1")
                    torch.cuda.empty_cache()
                    feature_extractor.reset_features()
                    _ = sliding_window_inference(image, roi_size, 1, feature_extractor)
                else:
                    raise e

            levels_to_process = list(extract_levels)
            if extract_bottleneck:
                levels_to_process.append('bottleneck')

            for level in levels_to_process:
                feature_map = feature_extractor.multi_level_features[level]

                if feature_map is None:
                    if idx < 3:
                        print(f"Warning: No features captured for level {level}, image {idx}")
                    all_features_by_level[level].append(None)
                    continue

                if pool_features:
                    if pooling_type == 'max':
                        feature_vector = torch.nn.functional.adaptive_max_pool3d(feature_map, 1)
                    else:
                        feature_vector = torch.nn.functional.adaptive_avg_pool3d(feature_map, 1)

                    feature_vector = feature_vector.squeeze().cpu().numpy()

                    if feature_vector.ndim == 0:
                        feature_vector = np.array([feature_vector])
                    elif feature_vector.ndim > 1:
                        feature_vector = feature_vector.flatten()

                    all_features_by_level[level].append(feature_vector)

                    if idx < 3:
                        print(f"Image {idx}, Level {level}: "
                              f"map shape {feature_map.shape}, "
                              f"vector shape {feature_vector.shape}")
                else:
                    feature_vector = feature_map.squeeze(0).cpu().numpy()
                    all_features_by_level[level].append(feature_vector)

            filename = data_list[idx]["image"]
            if isinstance(filename, list):
                filename = filename[0]
            basename = os.path.basename(filename).replace('.nii.gz', '').replace('.nii', '')
            all_filenames.append(basename)

            del image
            if (idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(dataloader)} images")

    # ----------------------------------------------------------------------- #
    #  Save per-level features                                                 #
    # ----------------------------------------------------------------------- #
    metadata = {"filenames": all_filenames, "levels": {}, "bottleneck": None}

    levels_to_save = list(extract_levels)
    if extract_bottleneck:
        levels_to_save.append('bottleneck')

    for level in levels_to_save:
        if pool_features and all_features_by_level[level]:
            valid_features = [f for f in all_features_by_level[level] if f is not None]

            if not valid_features:
                print(f"\n{level}: No valid features captured")
                continue

            features_array = np.stack(valid_features)
            level_name     = f"level_{level}" if isinstance(level, int) else level
            level_dir      = os.path.join(save_dir, level_name)
            os.makedirs(level_dir, exist_ok=True)

            np.save(os.path.join(level_dir, "all_features.npy"), features_array)

            feature_cols = [f"feature_{i}" for i in range(features_array.shape[1])]
            df = pd.DataFrame(features_array, columns=feature_cols)
            valid_filenames = [
                all_filenames[i]
                for i, f in enumerate(all_features_by_level[level])
                if f is not None
            ]
            df.insert(0, 'filename', valid_filenames)
            df.to_csv(os.path.join(level_dir, "all_features.csv"), index=False)

            if level == 'bottleneck':
                metadata["bottleneck"] = {
                    "channels": features_array.shape[1],
                    "shape": list(features_array.shape)
                }
            else:
                metadata["levels"][level] = {
                    "channels": features_array.shape[1],
                    "shape": list(features_array.shape)
                }

            print(f"\n{level_name}: {features_array.shape}")

    # ----------------------------------------------------------------------- #
    #  Concatenate all levels into one feature matrix                          #
    # ----------------------------------------------------------------------- #
    if pool_features:
        available_levels = [
            l for l in extract_levels
            if all_features_by_level[l] and any(f is not None for f in all_features_by_level[l])
        ]
        has_bottleneck = (
            extract_bottleneck
            and all_features_by_level.get('bottleneck')
            and any(f is not None for f in all_features_by_level['bottleneck'])
        )

        if available_levels or has_bottleneck:
            print("\n=== Creating concatenated features ===")

            valid_indices = list(range(len(all_filenames)))

            for level in available_levels:
                valid_indices = [
                    i for i in valid_indices
                    if i < len(all_features_by_level[level])
                    and all_features_by_level[level][i] is not None
                ]

            if has_bottleneck:
                valid_indices = [
                    i for i in valid_indices
                    if i < len(all_features_by_level['bottleneck'])
                    and all_features_by_level['bottleneck'][i] is not None
                ]

            if valid_indices:
                features_to_concat = []

                for level in available_levels:
                    level_features = [all_features_by_level[level][i] for i in valid_indices]
                    features_to_concat.append(np.stack(level_features))

                if has_bottleneck:
                    bottleneck_features = [
                        all_features_by_level['bottleneck'][i] for i in valid_indices
                    ]
                    features_to_concat.append(np.stack(bottleneck_features))

                concatenated_features = np.concatenate(features_to_concat, axis=1)

                np.save(
                    os.path.join(save_dir, "all_levels_concatenated.npy"),
                    concatenated_features
                )

                feature_cols = [f"feature_{i}" for i in range(concatenated_features.shape[1])]
                df = pd.DataFrame(concatenated_features, columns=feature_cols)
                valid_filenames = [all_filenames[i] for i in valid_indices]
                df.insert(0, 'filename', valid_filenames)
                df.to_csv(
                    os.path.join(save_dir, "all_levels_concatenated.csv"),
                    index=False
                )

                print(f"Concatenated features shape : {concatenated_features.shape}")
                print(f"Total dimensions            : {concatenated_features.shape[1]}")

    with open(os.path.join(save_dir, "feature_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Feature extraction complete! ===")
    print(f"Features saved to: {save_dir}")
    return all_features_by_level, all_filenames


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    checkpoint_path = "/container_workspace/Code/CL/MAMA_MIA_Again/step_one/step_one_scanner/mtybh8y3/checkpoints/unet-fold0-best.ckpt"
    with open("/container_workspace/Code/CL/MAMA_MIA_Again/step_three/selected_patients.json") as f:
        json_data = json.load(f)

    selected_fold = 0

    #train_files = [{"image": d["image"]} for d in json_data["training"] if d["fold"] != selected_fold]
    #train_files = [{"image": d["image"]} for d in json_data["training"]]
    train_files = [{"image": d["image"]} for d in json_data["test"]]
    print(f"Total images (fold == {selected_fold}): {len(train_files)}")

    features, filenames = extract_multilevel_features(
        checkpoint_path=checkpoint_path,
        data_list=train_files,
        save_dir="./extracted_features_step_three_238_avg",
        # levels 0-4 = encoder stages, level 5 = bottleneck ResBlock
        extract_levels=[0, 1, 2, 3, 4, 5],
        extract_bottleneck=False,   # set True to also save it separately
        pool_features=True,
        pooling_type='avg', #set 'max' to save maxpooling 
        debug=False
    )

    print("\n=== Done! ===")
    print(f"Total samples: {len(filenames)}")