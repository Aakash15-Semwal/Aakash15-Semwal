```python
import torch
import torch.nn as nn

class Aakash(nn.Module):
    """
    AI/ML Engineer — currently fine-tuning.
    Architecture: Human
    """

    def __init__(self):
        super().__init__()

        # Identity
        self.name        = "Aakash Semwal"
        self.institute   = "IIIT Ranchi  |  CS '28"
        self.role        = "AI/ML Engineer"
        self.status      = "fine-tuning"   # always

        # Optimization config
        self.optimizer   = torch.optim.AdamW(lr=3e-4)
        self.loss_fn     = nn.CrossEntropyLoss()   # current_self vs target_self

        # Stack
        self.stack = nn.ModuleDict({
            "ml"      : ["PyTorch", "HuggingFace", "FAISS"],
            "backend" : ["FastAPI", "Python"],
            "infra"   : ["Git", "Linux"],
        })

        # Interests
        self.interests = [
            "LLMs", "Semantic Search",
            "Information Retrieval", "Agentic AI",
        ]

    def forward(self, problem: str) -> "Solution":
        hypothesis  = self.read_papers(problem)
        build       = self.implement(hypothesis)
        metrics     = self.evaluate(build)

        if metrics.not_good_enough():
            return self.forward(problem)   # recursive improvement

        self.save_checkpoint()
        return build

    def extra_repr(self) -> str:
      return (
          f"epoch=2/4  |  loss=↓  |  "
          f"grad_norm=stable  |  eta=2028"
      )
```
