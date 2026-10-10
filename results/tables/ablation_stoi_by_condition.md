|                           |   noise |   reverb |   noise_reverb |   severe |
|:--------------------------|--------:|---------:|---------------:|---------:|
| Unprocessed               |   0.867 |    0.874 |          0.782 |    0.667 |
| CRN (noise data)          |   0.890 |    0.871 |          0.799 |    0.699 |
| CRN (reverb data)         |   0.866 |    0.889 |          0.792 |    0.673 |
| CRN (joint data)          |   0.881 |    0.884 |          0.803 |    0.694 |
| CRN joint, SI-SDR only    |   0.878 |    0.872 |          0.787 |    0.684 |
| DA-CRN multitask, no FiLM |   0.890 |    0.887 |          0.811 |    0.712 |
| DA-CRN (proposed)         |   0.880 |    0.884 |          0.801 |    0.686 |
| DA-CRN (clean re-run)     |   0.890 |    0.889 |          0.810 |    0.711 |
| DA-CRN oracle labels      |   0.881 |    0.895 |          0.809 |    0.708 |