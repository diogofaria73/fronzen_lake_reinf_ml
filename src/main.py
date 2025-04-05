import argparse
import os
import sys
from pathlib import Path
import random

# Add current directory to the path
sys.path.append(str(Path(__file__).resolve().parent))

# Import our modules
from training_code.q_learning import QLearning
from visualization.pygame_visualizer import FrozenLakeVisualizer
from visualization.statistics_visualizer import StatisticsVisualizer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Frozen Lake Q-Learning')
    
    # Main operation modes
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'visualize', 'stats', 'all'],
                       help='Operation mode: train, visualize, stats, or all')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=100000,
                       help='Number of training episodes')
    parser.add_argument('--slippery', action='store_true', default=True,
                       help='Whether the frozen lake is slippery')
    parser.add_argument('--map-size', type=int, default=4, choices=[4, 8],
                       help='Size of the frozen lake map (4x4 or 8x8)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate (alpha)')
    parser.add_argument('--discount-factor', type=float, default=0.99,
                       help='Discount factor (gamma)')
    parser.add_argument('--exploration-decay', type=float, default=0.0001,
                       help='Exploration rate decay factor')
    parser.add_argument('--max-steps', type=int, default=250,
                       help='Maximum steps per episode')
    
    # Visualization parameters
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for visualization')
    parser.add_argument('--vis-episodes', type=int, default=10,
                       help='Number of episodes to visualize')
    
    # File handling
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model (.pkl file) for loading')
    parser.add_argument('--save-path', type=str, default='../results',
                       help='Directory to save model and results')
    
    return parser.parse_args()

def main():
    """Main function to run the application"""
    # Set random seed for reproducibility
    random.seed(42)
    
    args = parse_arguments()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize agent
    agent = QLearning(is_slippery=args.slippery, map_size=args.map_size)
    
    # Load pre-trained model if specified
    if args.model_path:
        print(f"Carregando modelo de {args.model_path}")
        agent.load_model(args.model_path)
    
    # Mode: Train a new agent
    if args.mode in ['train', 'all'] and not args.model_path:
        print(f"Treinando por {args.episodes} episódios...")
        agent.train(
            num_episodes=args.episodes,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            exploration_decay=args.exploration_decay,
            max_steps_per_episode=args.max_steps
        )
        
        # Save the trained model
        saved_path = agent.save_model(args.save_path)
        print(f"Modelo salvo em {saved_path}")
    
    # Initialize statistics visualizer
    stats_visualizer = StatisticsVisualizer(agent, results_dir=args.save_path)
    
    # Integrated visualization and statistics workflow
    if args.mode in ['visualize', 'all']:
        print("Iniciando visualização do agente treinado...")
        
        # Run Pygame visualization first
        visualizer = FrozenLakeVisualizer(agent, fps=args.fps)
        success_rate = visualizer.visualize_multiple_episodes(num_episodes=args.vis_episodes)
        
        # Pass simulation results to statistics visualizer
        stats_visualizer.set_simulation_results(success_rate)
        
        # Then show statistics
        print("Gerando estatísticas...")
        stats_visualizer.visualize_all_stats()
    
    # Mode: Show only training statistics
    elif args.mode == 'stats':
        print("Gerando estatísticas de treinamento...")
        stats_visualizer.visualize_all_stats()
    
    print("Concluído!")

if __name__ == "__main__":
    main() 