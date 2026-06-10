# LOD RAW Noise-aware Feature Distillation Summary

| group | local branch | feature loss | lora | D1 | delta_d1_vs_M_DD | recover ratio | L5 dist | L8 dist | L11 dist | run |
|---|---|---|---|---|---|---|---|---|---|---|
| W0 | none | none | none | 0.831200 | 0.001506 | 0.026247 | 0.117962 | 0.098118 | 0.123409 | 0609_2153_W0_lod_raw_pair_decoder_e10_featnone_localnone |
| W0_seed123 | none | none | none | 0.830800 | 0.001106 | 0.019275 | 0.117161 | 0.096932 | 0.121765 | 0609_2153_W0_seed123_lod_raw_pair_decoder_e10_featnone_localnone |
| W3 | noiseaware_v1 | dav2_middeep_cosine | none | 0.831000 | 0.001306 | 0.022761 | 0.116468 | 0.096780 | 0.121787 | 0609_2153_W3_lod_raw_pair_decoder_e10_featmiddeep_lam005_noiseawarev1_gate003_s01 |
