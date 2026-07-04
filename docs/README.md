# Documentation

Companion guides for [**Practical LLMOps Deep Dive**](https://houmuza.gumroad.com/l/practical-llmops-deep-dive).

| Guide | What you will run |
|-------|-------------------|
| [Local GPU, Docker & streaming](local-gpu-docker-streaming.md) | `make serve-gpu`, `make smoke-stream`, `make docker-build` |
| [Azure ML deployment](azure-deployment.md) | YAML pipelines, CPU vs GPU profiles, GitHub Actions |
| [Training & evaluation](training-and-eval.md) | LoRA/QLoRA locally and on Azure, LLM-as-judge |
| [YAML reference](yaml-reference.md) | Field-by-field explanation of every deploy file |

## Quick decision tree

```text
Just exploring on a laptop?
  → make install && python test_model.py && make serve

Have a local NVIDIA GPU?
  → make serve-gpu && make smoke-stream

Want a container identical to Azure?
  → make docker-build && make docker-run-gpu

Fresh Azure subscription (no GPU quota yet)?
  → make deploy-all          # CPU: Standard_E4s_v3

GPU quota approved (T4 in region)?
  → make deploy-all-gpu      # GPU: Standard_NC4as_T4_v3

Fine-tune on Azure?
  → make compute-gpu-create && make train-lora-azure

Judge model quality with an external LLM?
  → export JUDGE_API_KEY=... && make eval && make eval-judge
```
