# MCTS Preference Learning Implementation

This repository contains an implementation of the paper "Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning" using the Qwen language model.

Repository: [https://github.com/artaasd95/mcts-preference-learning](https://github.com/artaasd95/mcts-preference-learning)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Create a `data` directory
   - Add your training data in JSON format with the following structure:
   ```json
   [
     {
       "prompt": "Your prompt here",
       "answer": "Expected answer"  // Optional
     },
     ...
   ]
   ```

## Training

To start training:

```bash
python src/train.py
```

The training process will:
1. Load the Qwen-1.5B model
2. Use MCTS to collect step-level preferences
3. Train the model using DPO
4. Save checkpoints after each iteration

## Model Architecture

- Base Model: Qwen-1.5B (smallest Qwen model)
- Training Method: Direct Preference Optimization (DPO)
- Search Algorithm: Monte Carlo Tree Search (MCTS)

## Configuration

Key parameters can be adjusted in `src/train.py`:
- `num_iterations`: Number of training iterations
- `num_simulations`: MCTS simulations per step
- `max_depth`: Maximum search depth
- `batch_size`: Training batch size
- `learning_rate`: DPO learning rate

## Citation

If you use this implementation, please cite both the original paper and this repository:

```bibtex
@misc{xie2024montecarlotreesearch,
      title={Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning}, 
      author={Yuxi Xie and Anirudh Goyal and Wenyue Zheng and Min-Yen Kan and Timothy P. Lillicrap and Kenji Kawaguchi and Michael Shieh},
      year={2024},
      eprint={2405.00451},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.00451}, 
}

@software{mcts_preference_learning,
  author = {Arta Asd},
  title = {MCTS Preference Learning Implementation},
  year = {2024},
  url = {https://github.com/artaasd95/mcts-preference-learning},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```

## License

This implementation is for educational purposes only. Please respect the original paper's intellectual property.