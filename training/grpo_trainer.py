from __future__ import annotations
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import wandb
from envs.factor_env import FactorImproveEnv
from agent.prompt import build_prompt

class GRPOTrainer:
    """GRPO (Generative Reinforcement Learning with Policy Optimization) trainer for factor improvement."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_wandb: bool = True
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.env = None
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf
        
        if use_wandb:
            wandb.init(project="qr-agent-grpo", name="factor-improvement")
    
    def setup_model(self):
        """Initialize the Qwen model with LoRA fine-tuning."""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Initialize environment
        self.env = FactorImproveEnv()
        
        print("Model setup complete!")
    
    def generate_action(self, prompt: str) -> Dict:
        """Generate action using the fine-tuned model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract action from response
        try:
            # Find the last JSON object in the response
            lines = response.split('\n')
            for line in reversed(lines):
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    action = json.loads(line.strip())
                    return action
        except:
            pass
        
        # Fallback to default action if parsing fails
        return response
    
    def collect_episode(self) -> Tuple[List[Dict], List[float], float]:
        """Collect a single episode using current policy."""
        obs, _ = self.env.reset()
        done = False
        episode_actions = []
        episode_rewards = []
        total_reward = 0.0
        
        while not done:
            prompt = build_prompt("Improve OOS Sharpe >= 0.2 with turnover <= 1.5", obs)
            action = self.generate_action(prompt)
            
            episode_actions.append(action)
            obs, reward, done, _, _ = self.env.step(action)
            episode_rewards.append(reward)
            total_reward += reward
        
        return episode_actions, episode_rewards, total_reward
    
    def create_training_data(self, episodes: List[Tuple[List[Dict], List[float], float]]) -> Dataset:
        """Create training dataset from collected episodes."""
        training_data = []
        
        for episode_actions, episode_rewards, total_reward in episodes:
            # Create training examples for each action in the episode
            for i, (action, reward) in enumerate(zip(episode_actions, episode_rewards)):
                # Create prompt for this action
                if i == 0:
                    # First action - use initial state
                    obs = {"budget_left": 4, "last_eval": {"oos_sharpe": 0.0, "turnover": 0.0, "tests_pass": True, "leak": False}, "params": {"top_q": 0.2, "turnover_cap": 1.5, "delay_days": 1}}
                else:
                    # Subsequent actions - reconstruct state
                    obs = {"budget_left": 4 - i, "last_eval": {"oos_sharpe": 0.0, "turnover": 0.0, "tests_pass": True, "leak": False}, "params": {"top_q": 0.2, "turnover_cap": 1.5, "delay_days": 1}}
                
                prompt = build_prompt("Improve OOS Sharpe >= 0.2 with turnover <= 1.5", obs)
                
                # Create target response
                target_response = json.dumps(action)
                
                # Calculate advantage (simple reward-to-go)
                advantage = sum(episode_rewards[i:])
                
                training_data.append({
                    "prompt": prompt,
                    "response": target_response,
                    "reward": reward,
                    "advantage": advantage,
                    "total_reward": total_reward
                })
        
        return Dataset.from_list(training_data)
    
    def train_step(self, training_data: Dataset):
        """Perform one training step using PPO."""
        # Configure PPO
        grpo_config = GRPOConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            target_kl=0.1,
            ppo_epochs=4,
            seed=42
        )
        
        # Initialize PPO trainer
        self.trainer = GRPOTrainer(
            config=grpo_config,
            model=self.model,
            ref_model=self.model,
            tokenizer=self.tokenizer,
            dataset=training_data,
            peft_config=None  # Already applied
        )
        
        # Training step
        self.trainer.step()
    
    def train(
        self,
        num_episodes: int = 100,
        episodes_per_update: int = 10,
        save_every: int = 20
    ):
        """Main training loop."""
        if self.model is None:
            self.setup_model()
        
        print(f"Starting GRPO training for {num_episodes} episodes")
        
        for update in range(0, num_episodes, episodes_per_update):
            print(f"\n--- Update {update//episodes_per_update + 1} ---")
            
            # Collect episodes
            episodes = []
            for _ in range(min(episodes_per_update, num_episodes - update)):
                episode_data = self.collect_episode()
                episodes.append(episode_data)
                
                # Log episode stats
                _, _, total_reward = episode_data
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(len(episode_data[0]))
                
                if total_reward > self.best_reward:
                    self.best_reward = total_reward
                    self.save_model("best_model")
                
                if self.use_wandb:
                    wandb.log({
                        "episode_reward": total_reward,
                        "episode_length": len(episode_data[0]),
                        "best_reward": self.best_reward
                    })
            
            # Create training data
            training_data = self.create_training_data(episodes)
            
            # Training step
            self.train_step(training_data)
            
            # Log update stats
            avg_reward = np.mean([ep[2] for ep in episodes])
            print(f"Average reward: {avg_reward:.3f}")
            print(f"Best reward so far: {self.best_reward:.3f}")
            
            # Save checkpoint
            if (update + episodes_per_update) % save_every == 0:
                self.save_model(f"checkpoint_{update + episodes_per_update}")
        
        print("Training complete!")
        if self.use_wandb:
            wandb.finish()
    
    def save_model(self, name: str):
        """Save the fine-tuned model."""
        save_path = Path(f"models/{name}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, path: str):
        """Load a saved model."""
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    trainer = GRPOTrainer()
    trainer.train(num_episodes=50, episodes_per_update=5)
