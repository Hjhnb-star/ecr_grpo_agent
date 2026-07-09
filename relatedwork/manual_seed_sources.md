# Manual related-work seed sources

CoPaper's Semantic Scholar search step returned HTTP 429 on 2026-07-08, so the initial related-work cache was seeded manually from primary paper pages instead of S2 metadata.

## Seed papers

- DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Demystifying GRPO: https://arxiv.org/abs/2603.01162
- PPO: https://arxiv.org/abs/1707.06347
- RUDDER: https://arxiv.org/abs/1806.07857
- Let's Verify Step by Step: https://arxiv.org/abs/2305.20050
- Training Verifiers to Solve Math Word Problems: https://arxiv.org/abs/2110.14168
- Agent Lightning: https://arxiv.org/abs/2508.03680
- ALFWorld: https://arxiv.org/abs/2010.03768
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- WebShop: https://arxiv.org/abs/2207.01206
- LATS: https://arxiv.org/abs/2310.04406

These are seed references, not the final related-work set. Before submission, rerun the S2 search with `S2_API_KEY` or `SEMANTIC_SCHOLAR_API_KEY` configured and prune/extend this list.
