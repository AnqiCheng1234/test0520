# Resolved-Config Experiment Matrix

First formal batch uses single-variable changes around the STF resolved-config surface. All scripts must keep unchanged dimensions explicit and must not use `--input-type`.

| ID | Purpose | Changed dimension | Required config |
| --- | --- | --- | --- |
| A | RGB baseline | RGB domain/front-end | `input_domain=rgb`, `front_end=dav2_rgb`, `dataset_family=stf_rgb`, `dataset_input_mode=rgb`, `model_input_tensor=image`, `bridge=none`, `decoder_feature_adapter=none`, `lora=none` |
| B | Raw4 to 3ch RAM | raw 3ch RAM front-end | `front_end=raw_to_base_rgb_ram3`, `bridge=none`, `decoder_feature_adapter=none`, `lora=none` |
| C | Raw4 to 4ch RAM | raw 4ch RAM front-end | `front_end=raw_ram4`, `bridge=none`, `decoder_feature_adapter=none`, `lora=none` |
| D | 3ch RAM + bridge | bridge only | B + `bridge=raw_feature_bridge`, `bridge_feature_source_channels=x3`, `bridge_feature_keys=x_cat,ffm_mid,x3` |
| E | 3ch RAM + decoder adaptor | decoder feature adaptor only | B + `decoder_feature_adapter=raw_feature_adapter`, `adapter_feature_source_channels=x3`, `feature_adapter_keys=x_cat,ffm_mid,x3` |
| F | 3ch RAM + bridge + adaptor | bridge and adaptor | B + D + E |
| G | RGB + LoRA | LoRA only | A + `lora=dav2_lora`, explicit rank/alpha/lr/tap layers |

For all raw rows, keep these explicit unless a script documents a deliberate change:

- `input_domain=raw4`
- `dataset_family=stf_raw`
- `dataset_input_mode=raw_ram`
- `model_input_tensor=raw`
- `raw_storage_format=legacy_bggR_decomp16`
- `loss_type=ssi`
- `dav2_train_mode=none` for front-end/bridge/adaptor ablations

The two maintained 0522 queue scripts currently instantiate F for `x4/raw_ram4` and F for `x3/raw_to_base_rgb_ram3`.
