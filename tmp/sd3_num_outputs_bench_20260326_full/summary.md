# SD3 num_outputs_per_prompt Benchmark

- Model: `stabilityai/stable-diffusion-3-medium-diffusers`
- Backend: `sglang`
- Attention backend: `torch_sdpa`
- Prompt: `A red ceramic mug on a wooden table, soft natural light, high detail`
- Steps: `20`
- Guidance scale: `7.0`
- Seed: `123`
- Repeats per setting: `3`

| Resolution | num_outputs_per_prompt | Median total time (ms) | Median peak reserved memory (MB) | Median time per image (ms) | Speedup vs n=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 1 | 3770.44 | 6828.00 | 3770.44 | 1.00x |
| 512x512 | 2 | 3668.51 | 8744.00 | 1834.26 | 2.06x |
| 512x512 | 4 | 4333.81 | 12972.00 | 1083.45 | 3.48x |
| 512x512 | 8 | 5566.59 | 21324.00 | 695.82 | 5.42x |
| 512x512 | 12 | 6929.45 | 29666.00 | 577.45 | 6.53x |
| 512x512 | 16 | 8323.93 | 38020.00 | 520.25 | 7.25x |
| 1024x1024 | 1 | 4460.34 | 13344.00 | 4460.34 | 1.00x |
| 1024x1024 | 2 | 5890.59 | 21208.00 | 2945.29 | 1.51x |
| 1024x1024 | 4 | 8699.43 | 37840.00 | 2174.86 | 2.05x |
| 1024x1024 | 8 | 14270.71 | 70976.00 | 1783.84 | 2.50x |
| 1024x1024 | 12 | 20020.55 | 97998.00 | 1668.38 | 2.67x |
| 1024x1024 | 16 | 25499.45 | 129088.00 | 1593.72 | 2.80x |

## Conclusion

- `512x512`: `n=16` completed successfully. Median time per image improved from `3770.44 ms` at `n=1` to `520.25 ms` at `n=16` (86.2% lower), while median peak reserved memory increased from `6828.00 MB` to `38020.00 MB` (5.57x).
- `1024x1024`: `n=16` completed successfully. Median time per image improved from `4460.34 ms` at `n=1` to `1593.72 ms` at `n=16` (64.3% lower), while median peak reserved memory increased from `13344.00 MB` to `129088.00 MB` (9.67x).
- The SD3 runs scaled with `num_outputs_per_prompt` via the full batched path: a single request reused the pre-denoising pipeline and denoised the latent batch timestep-by-timestep.
