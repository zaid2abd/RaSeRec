# Meta-α Fusion Gate for RaSeRec

## Overview
This branch introduces the **Meta-α Fusion Gate** mechanism to enhance the **RaSeRec (Retrieval-Augmented Sequential Recommendation)** framework. The proposed mechanism improves recommendation quality by implementing:

- **Learnable fusion parameters**: Replacing static fusion coefficients (`α=0.5`, `β=0.5`) with learnable parameters that adapt during training.
- **Memory filtering and weighting**: A novel technique to prioritize relevant retrieved memories.
- **Progressive training strategy**: A warmup phase followed by targeted parameter optimization.

## Key Innovations

### 1. Meta-α Fusion Gate
The Meta-α Fusion Gate introduces learnable parameters for controlling the fusion of user representations and retrieved memories:

```python
fused_representation = α * user_repr + (1 - α) * (β * channel1_repr + (1 - β) * channel2_repr)
```

Where:

- `α` controls the balance between original user representation and augmented representations.
- `β` controls the balance between two augmentation channels.
- Both parameters are learned during training with guidance toward optimal values (`α ≈ 0.7`, `β ≈ 0.3`).

### 2. Memory Filtering and Weighting
We introduce a `filter_and_weight_memories` function that:

- Calculates relevance scores between user representations and retrieved memories.
- Applies softmax to obtain attention weights.
- Weights memories based on their relevance to the current user.

This technique reduces the influence of irrelevant memories and enhances the impact of relevant ones.

### 3. Progressive Training Strategy
Our training approach consists of two phases:

- **Warmup phase**: Uses the original model with fixed parameters (`α=0.5`, `β=0.5`).
- **Fine-tuning phase**: Gradually updates parameters toward optimal values while monitoring performance.

## Implementation Details
The implementation includes:

- `MetaAlphaFusionGate` class for parameter management.
- `filter_and_weight_memories` function for memory enhancement.
- Modified training process with warmup and targeted optimization.
- Comprehensive statistics tracking and analysis.

## Experimental Results
Our approach shows improvements in recommendation quality, particularly for:

- Long-tail items (less popular items).
- Users with diverse interests.
- Cold-start scenarios.

Detailed performance metrics and comparisons are available in the experimental results section.

## Usage
To use this enhanced version of RaSeRec:

```bash
# Clone this repository
git clone https://github.com/zaid2abd/RaSeRec.git
cd RaSeRec

# Install dependencies
pip install -r requirements.txt

# Run the training script
bash raserec.sh
```

## Branch Information

- **Branch name**: `feature/meta-alpha-fusion-gate`
- **Base branch**: `main`
- **Key modified files**:
  - `meta_alpha_gate.py`: Implementation of the Meta-α Fusion Gate.
  - `raserec.py`: Integration with RaSeRec model.
  - `trainer.py`: Enhanced training process.
  - `seq.yaml`: Configuration parameters.

## Citation
If you use this code in your research, please cite:

```bibtex
@misc{abd2025meta,
    title={Meta-α Fusion Gate: Enhancing Retrieval-Augmented Sequential Recommendation through Learnable Fusion Parameters},
    author={Zaid Abd and [Your Name]},
    year={2025},
    howpublished={GitHub Repository},
    url={https://github.com/zaid2abd/RaSeRec}
}
```

## Acknowledgments
This work builds upon the RaSeRec framework by Zhao et al. (2024). We thank the original authors for their valuable contribution to the field of retrieval-augmented recommendation.



# Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole) and [DuoRec](https://github.com/RuihongQiu/DuoRec).
