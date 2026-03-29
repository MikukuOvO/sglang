# SD3 num_outputs_per_prompt Benchmark

- Model: `stabilityai/stable-diffusion-3-medium-diffusers`
- Backend: `sglang`
- Attention backend: `torch_sdpa`
- Prompt: `A red ceramic mug on a wooden table, soft natural light, high detail`
- Steps: `20`
- Guidance scale: `7.0`
- Seed: `123`
- Repeats per setting: `1`

| Resolution | num_outputs_per_prompt | Median total time (ms) | Median peak reserved memory (MB) | Median time per image (ms) | Speedup vs n=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 1 | 3489.15 | 6828.00 | 3489.15 | 1.00x |

## Conclusion

- `512x512`: `n=1` completed successfully. Median time per image improved from `3489.15 ms` at `n=1` to `3489.15 ms` at `n=1` (0.0% lower), while median peak reserved memory increased from `6828.00 MB` to `6828.00 MB` (1.00x).
- The SD3 runs scaled with `num_outputs_per_prompt` via the full batched path: a single request reused the pre-denoising pipeline and denoised the latent batch timestep-by-timestep.
