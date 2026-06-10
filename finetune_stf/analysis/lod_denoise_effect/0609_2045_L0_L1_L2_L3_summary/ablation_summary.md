# LOD RAW Noise-aware Feature Distillation Summary

| group | local branch | feature loss | lora | D1 | delta_d1_vs_M_DD | recover ratio | L5 dist | L8 dist | L11 dist | run |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | none | none | dav2_lora | 0.845600 | 0.000321 | 0.005981 |  |  |  | 0609_2045_L0_lod_raw_pair_lora_decoder_e10_featnone_localnone |
| L1 | none | dav2_middeep_cosine | dav2_lora | 0.848400 | 0.003121 | 0.058155 |  |  |  | 0609_2045_L1_lod_raw_pair_lora_decoder_e10_featmiddeep_lam005_localnone |
| L2 | noiseaware_v1 | none | dav2_lora | 0.846000 | 0.000721 | 0.013435 |  |  |  | 0609_2045_L2_lod_raw_pair_lora_decoder_e10_featnone_noiseawarev1_gate003_s01 |
| L3 | noiseaware_v1 | dav2_middeep_cosine | dav2_lora | 0.848500 | 0.003221 | 0.060018 |  |  |  | 0609_2045_L3_lod_raw_pair_lora_decoder_e10_featmiddeep_lam005_noiseawarev1_gate003_s01 |
