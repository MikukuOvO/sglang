I did a follow-up benchmark for `stable-diffusion-3-medium-diffusers` with the `sglang` backend to validate `num_outputs_per_prompt` scaling.

The benchmark was run with:
- fixed prompt and seed
- `num_inference_steps=20`
- `guidance_scale=7.0`
- `num_outputs_per_prompt in {1, 2, 4, 8, 12, 16}`
- both `512x512` and `1024x1024`
- `3` repeats per setting, reporting the median

The results are consistent with SD3 going through the full batched optimization path rather than generating outputs one by one outside the denoising loop.

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

Summary:
- `n=16` completed successfully at both resolutions.
- At `512x512`, median time per image improved from `3770.44 ms` to `520.25 ms` (`7.25x`).
- At `1024x1024`, median time per image improved from `4460.34 ms` to `1593.72 ms` (`2.80x`).
- Peak reserved memory scales roughly monotonically with batch size, as expected for batched generation.

Combined plots:
- attach `sd3_num_outputs_benchmark_3x2.svg`
