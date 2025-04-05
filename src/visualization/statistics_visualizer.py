import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.gridspec as gridspec
import pygame
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

# Add the parent directory to the path to import training_code
sys.path.append(str(Path(__file__).resolve().parent.parent))
from training_code.q_learning import QLearning

class StatisticsVisualizer:
    """Visualizer for Q-learning training statistics"""
    
    def __init__(self, agent=None, results_dir="../results"):
        """
        Initialize the statistics visualizer
        
        Args:
            agent (QLearning, optional): Trained Q-learning agent
            results_dir (str): Directory to save result plots
        """
        self.agent = agent
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Default window size
        self.window_size = 100
        
        # Simulation results (Pygame visualization success rate)
        self.simulation_success_rate = None
        
        # Pygame colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (240, 240, 240)
        self.DARK_GRAY = (200, 200, 200)
        self.BLUE = (50, 150, 255)
        self.GREEN = (60, 200, 60)
        self.RED = (220, 50, 50)
        self.YELLOW = (240, 240, 50)
    
    def set_simulation_results(self, success_rate):
        """Set the simulation success rate from Pygame visualization"""
        self.simulation_success_rate = success_rate
    
    def create_summary_stats(self):
        """Create a summary of important statistics"""
        if not self.agent or not self.agent.episode_rewards:
            return "Nenhum dado de treinamento disponível"
        
        # Calculate overall statistics
        total_episodes = len(self.agent.episode_rewards)
        success_episodes = sum(self.agent.episode_success)
        success_rate = success_episodes / total_episodes if total_episodes > 0 else 0
        
        # Calculate final performance (last 10% of episodes)
        last_episodes = max(1, total_episodes // 10)
        final_success_rate = np.mean(self.agent.episode_success[-last_episodes:])
        
        # Calculate average episode length for successful episodes
        successful_lengths = [length for length, success in 
                             zip(self.agent.episode_lengths, self.agent.episode_success) 
                             if success]
        avg_success_length = np.mean(successful_lengths) if successful_lengths else 0
        
        # Create summary text
        summary = [
            f"Resumo do Treinamento e Avaliação:",
            f"",
            f"Total de episódios: {total_episodes}",
            f"Episódios com sucesso: {success_episodes} ({success_rate*100:.1f}%)",
            f"Taxa de sucesso final (últimos {last_episodes} episódios): {final_success_rate*100:.1f}%",
            f"Passos médios para sucesso: {avg_success_length:.1f}",
            f"",
        ]
        
        # Add simulation results if available
        if self.simulation_success_rate is not None:
            summary.append(f"Resultados da Simulação:")
            summary.append(f"Taxa de sucesso na simulação visual: {self.simulation_success_rate*100:.1f}%")
        
        return '\n'.join(summary)
    
    def visualize_all_stats(self):
        """Generate all statistics visualizations using matplotlib and save files"""
        print("Gerando visualizações de estatísticas de treinamento...")
        
        # Save figures to files
        self._save_all_figures()
        
        # Export to PDF
        self._export_to_pdf()
        
        # Display the stats in Pygame
        self._show_stats_pygame()
        
        print(f"Todas as visualizações salvas em {self.results_dir}/")
    
    def _show_stats_pygame(self):
        """Show statistics in a Pygame window"""
        # Initialize Pygame
        pygame.init()
        
        # Set up the display - larger window
        WIDTH, HEIGHT = 1400, 900
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Estatísticas de Treinamento - Frozen Lake Q-Learning')
        
        # Load saved images
        metrics_img = pygame.image.load(os.path.join(self.results_dir, 'training_metrics.png'))
        q_table_img = pygame.image.load(os.path.join(self.results_dir, 'q_table_heatmap.png'))
        policy_img = pygame.image.load(os.path.join(self.results_dir, 'optimal_policy.png'))
        
        # Scale images to be larger and better fit the screen
        metrics_img = pygame.transform.scale(metrics_img, (WIDTH//2, HEIGHT//2))
        q_table_img = pygame.transform.scale(q_table_img, (WIDTH//2, HEIGHT//2))
        policy_img = pygame.transform.scale(policy_img, (WIDTH//2, HEIGHT//2))
        
        # Font setup - larger fonts
        font_title = pygame.font.SysFont('Arial', 32, bold=True)
        font_text = pygame.font.SysFont('Arial', 20)
        
        # Get summary text
        summary_text = self.create_summary_stats().split('\n')
        
        # Main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Clear screen
            screen.fill(self.WHITE)
            
            # Draw title
            title = font_title.render("Estatísticas de Aprendizado por Reforço - Frozen Lake", True, self.BLACK)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
            
            # Draw images in grid layout with proper spacing
            screen.blit(metrics_img, (20, 80))
            screen.blit(q_table_img, (WIDTH//2 + 20, 80))
            screen.blit(policy_img, (20, HEIGHT//2 + 40))
            
            # Draw summary text in bottom right with better formatting
            summary_box = pygame.Rect(WIDTH//2 + 20, HEIGHT//2 + 40, WIDTH//2 - 40, HEIGHT//2 - 60)
            pygame.draw.rect(screen, self.GRAY, summary_box)
            pygame.draw.rect(screen, self.DARK_GRAY, summary_box, 2)
            
            # Render text lines with better spacing
            for i, line in enumerate(summary_text):
                text_surface = font_text.render(line, True, self.BLACK)
                screen.blit(text_surface, (WIDTH//2 + 30, HEIGHT//2 + 50 + i * 28))
            
            # Add instruction to exit
            exit_text = font_text.render("Pressione ESC para sair", True, self.BLACK)
            screen.blit(exit_text, (WIDTH - exit_text.get_width() - 30, HEIGHT - 40))
            
            pygame.display.flip()
        
        pygame.quit()
    
    def _plot_success_rate_on_axis(self, ax, window_size=100):
        """Plot success rate on the given matplotlib axis"""
        success = self.agent.episode_success
        episodes = np.arange(1, len(success) + 1)
        
        # Calculate moving average of success rate
        if window_size < len(success):
            success_rate = np.convolve(success, np.ones(window_size)/window_size, mode='valid')
            sr_episodes = episodes[window_size-1:]
        else:
            success_rate = success
            sr_episodes = episodes
        
        ax.plot(sr_episodes, success_rate, 'g-', label=f'Taxa de Sucesso (janela={window_size})')
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Taxa de Sucesso')
        ax.set_title('Taxa de Sucesso Durante o Treinamento')
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_rewards_on_axis(self, ax, window_size=100):
        """Plot episode rewards on the given matplotlib axis"""
        rewards = self.agent.episode_rewards
        episodes = np.arange(1, len(rewards) + 1)
        
        # Calculate moving average
        if window_size < len(rewards):
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ma_episodes = episodes[window_size-1:]
        else:
            moving_avg = rewards
            ma_episodes = episodes
        
        ax.plot(episodes, rewards, 'b-', alpha=0.2, label='Recompensa por Episódio')
        ax.plot(ma_episodes, moving_avg, 'r-', label=f'Média Móvel (janela={window_size})')
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Recompensa')
        ax.set_title('Recompensas Durante o Treinamento')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_lengths_on_axis(self, ax, window_size=100):
        """Plot episode lengths on the given matplotlib axis"""
        lengths = self.agent.episode_lengths
        episodes = np.arange(1, len(lengths) + 1)
        
        # Calculate moving average
        if window_size < len(lengths):
            moving_avg = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
            ma_episodes = episodes[window_size-1:]
        else:
            moving_avg = lengths
            ma_episodes = episodes
        
        ax.plot(episodes, lengths, 'b-', alpha=0.2, label='Comprimento do Episódio')
        ax.plot(ma_episodes, moving_avg, 'r-', label=f'Média Móvel (janela={window_size})')
        ax.set_xlabel('Episódio')
        ax.set_ylabel('Número de Passos')
        ax.set_title('Comprimento dos Episódios Durante o Treinamento')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_optimal_policy_on_axis(self, ax):
        """Plot the optimal policy on the given matplotlib axis"""
        q_table = self.agent.q_table
        map_size = self.agent.map_size
        
        # Get the environment description to know where holes and goal are
        env_desc = self.agent.env.unwrapped.desc
        
        # Get optimal action for each state
        optimal_actions = np.argmax(q_table, axis=1)
        
        # Define action symbols and directions
        action_symbols = ['←', '↓', '→', '↑']
        dx = [0, 0, 0.3, -0.3]  # x adjustment for arrow (L,D,R,U)
        dy = [0.3, -0.3, 0, 0]  # y adjustment for arrow (L,D,R,U)
        
        # Plot grid
        for i in range(map_size + 1):
            ax.axhline(i, color='black', lw=2)
            ax.axvline(i, color='black', lw=2)
        
        # Fill grid with colors based on environment
        for i in range(map_size):
            for j in range(map_size):
                cell_type = env_desc[i][j]
                if cell_type == b'H':  # Hole
                    rect = plt.Rectangle((j, map_size - i - 1), 1, 1, facecolor='royalblue')
                    ax.add_patch(rect)
                elif cell_type == b'G':  # Goal
                    rect = plt.Rectangle((j, map_size - i - 1), 1, 1, facecolor='green')
                    ax.add_patch(rect)
                elif cell_type == b'S':  # Start
                    rect = plt.Rectangle((j, map_size - i - 1), 1, 1, facecolor='khaki')
                    ax.add_patch(rect)
                else:  # Frozen
                    rect = plt.Rectangle((j, map_size - i - 1), 1, 1, facecolor='lightcyan')
                    ax.add_patch(rect)
        
        # Plot arrows representing the policy
        for i in range(map_size):
            for j in range(map_size):
                state = i * map_size + j
                action = optimal_actions[state]
                
                # Skip holes and goal for clarity
                cell_type = env_desc[i][j]
                if cell_type in [b'H', b'G']:
                    continue
                
                # Add state number in small font at corner
                ax.text(j + 0.1, map_size - i - 0.9, f"{state}", fontsize=10, ha='left', va='top')
                
                # Plot the arrow representing the optimal action
                ax.text(j + 0.5 + dx[action], map_size - i - 0.5 + dy[action],
                       action_symbols[action], fontsize=30, ha='center', va='center', fontweight='bold')
        
        # Set axis limits and remove ticks
        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='lightcyan', label='Gelo'),
            plt.Rectangle((0, 0), 1, 1, facecolor='royalblue', label='Buraco'),
            plt.Rectangle((0, 0), 1, 1, facecolor='green', label='Objetivo'),
            plt.Rectangle((0, 0), 1, 1, facecolor='khaki', label='Início')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=14)
        
        ax.set_title('Política Ótima Aprendida')
    
    def _save_all_figures(self):
        """Save all figures to image files"""
        # Save combined training metrics
        metrics_fig = plt.figure(figsize=(12, 15))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)
        
        # Plot 1: Success Rate
        ax1 = metrics_fig.add_subplot(gs[0])
        self._plot_success_rate_on_axis(ax1, self.window_size)
        ax1.tick_params(labelsize=12)
        ax1.title.set_fontsize(16)
        ax1.xaxis.label.set_fontsize(14)
        ax1.yaxis.label.set_fontsize(14)
        ax1.legend(fontsize=12)
        
        # Plot 2: Episode Rewards
        ax2 = metrics_fig.add_subplot(gs[1])
        self._plot_rewards_on_axis(ax2, self.window_size)
        ax2.tick_params(labelsize=12)
        ax2.title.set_fontsize(16)
        ax2.xaxis.label.set_fontsize(14)
        ax2.yaxis.label.set_fontsize(14)
        ax2.legend(fontsize=12)
        
        # Plot 3: Episode Lengths
        ax3 = metrics_fig.add_subplot(gs[2])
        self._plot_lengths_on_axis(ax3, self.window_size)
        ax3.tick_params(labelsize=12)
        ax3.title.set_fontsize(16)
        ax3.xaxis.label.set_fontsize(14)
        ax3.yaxis.label.set_fontsize(14)
        ax3.legend(fontsize=12)
        
        metrics_fig.tight_layout(pad=2.0)
        metrics_fig.savefig(os.path.join(self.results_dir, 'training_metrics.png'), dpi=300)
        
        # Save Q-table visualization
        q_fig = plt.figure(figsize=(16, 14))
        q_gs = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.5)
        
        q_table = self.agent.q_table
        map_size = self.agent.map_size
        state_count = map_size * map_size
        action_names = ['Esquerda', 'Baixo', 'Direita', 'Cima']
        
        for action in range(4):
            ax = q_fig.add_subplot(q_gs[action])
            q_values_grid = q_table[:state_count, action].reshape(map_size, map_size)
            im = ax.imshow(q_values_grid, cmap='hot')
            ax.set_title(f'Valores Q para: {action_names[action]}', fontsize=16)
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=12)
            ax.grid(False)
            ax.tick_params(labelsize=12)
            
            # Add state numbers
            for i in range(map_size):
                for j in range(map_size):
                    state = i * map_size + j
                    ax.text(j, i, f"{state}", ha="center", va="center", color="w", fontsize=14, fontweight='bold')
        
        q_fig.tight_layout(pad=2.0)
        q_fig.savefig(os.path.join(self.results_dir, 'q_table_heatmap.png'), dpi=300)
        
        # Save optimal policy visualization
        policy_fig = plt.figure(figsize=(12, 12))
        ax = policy_fig.add_subplot(111)
        self._plot_optimal_policy_on_axis(ax)
        ax.title.set_fontsize(18)
        policy_fig.tight_layout(pad=2.0)
        policy_fig.savefig(os.path.join(self.results_dir, 'optimal_policy.png'), dpi=300)
        
        # Save summary to text file
        summary_text = self.create_summary_stats()
        with open(os.path.join(self.results_dir, 'training_summary.txt'), 'w') as f:
            f.write(summary_text)
        
        plt.close('all')
    
    def _export_to_pdf(self):
        """Export all statistics to a single PDF file with date-based name"""
        # Generate filename based on date
        today = datetime.now()
        filename = f"{today.day:02d}.{today.month:02d}.{today.year}_statistics_and_results.pdf"
        filepath = os.path.join(self.results_dir, filename)
        
        print(f"Exportando estatísticas para PDF: {filename}")
        
        # Create PDF with PdfPages
        with PdfPages(filepath) as pdf:
            # Create a figure for the title page
            fig_title = plt.figure(figsize=(11.7, 8.3))  # A4 size
            fig_title.suptitle("Frozen Lake Q-Learning - Relatório de Resultados", fontsize=20, y=0.98)
            
            # Add timestamp and summary
            summary_text = self.create_summary_stats()
            plt.figtext(0.1, 0.8, f"Gerado em: {today.strftime('%d/%m/%Y %H:%M:%S')}", fontsize=12)
            plt.figtext(0.1, 0.75, "Resumo:", fontsize=16, weight='bold')
            
            # Add each line of the summary
            y_pos = 0.7
            for line in summary_text.split('\n'):
                plt.figtext(0.1, y_pos, line, fontsize=13)
                y_pos -= 0.03
            
            # Add to PDF
            pdf.savefig(fig_title)
            plt.close(fig_title)
            
            # Training metrics page
            metrics_fig = plt.figure(figsize=(11.7, 8.3))
            gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)
            
            # Plot 1: Success Rate
            ax1 = metrics_fig.add_subplot(gs[0])
            self._plot_success_rate_on_axis(ax1, self.window_size)
            ax1.tick_params(labelsize=12)
            ax1.title.set_fontsize(14)
            ax1.xaxis.label.set_fontsize(12)
            ax1.yaxis.label.set_fontsize(12)
            ax1.legend(fontsize=10)
            
            # Plot 2: Episode Rewards
            ax2 = metrics_fig.add_subplot(gs[1])
            self._plot_rewards_on_axis(ax2, self.window_size)
            ax2.tick_params(labelsize=12)
            ax2.title.set_fontsize(14)
            ax2.xaxis.label.set_fontsize(12)
            ax2.yaxis.label.set_fontsize(12)
            ax2.legend(fontsize=10)
            
            # Plot 3: Episode Lengths
            ax3 = metrics_fig.add_subplot(gs[2])
            self._plot_lengths_on_axis(ax3, self.window_size)
            ax3.tick_params(labelsize=12)
            ax3.title.set_fontsize(14)
            ax3.xaxis.label.set_fontsize(12)
            ax3.yaxis.label.set_fontsize(12)
            ax3.legend(fontsize=10)
            
            metrics_fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.0)
            metrics_fig.suptitle("Métricas de Treinamento", fontsize=18, y=0.98)
            pdf.savefig(metrics_fig)
            plt.close(metrics_fig)
            
            # Q-table visualization page
            q_fig = plt.figure(figsize=(11.7, 8.3))
            q_fig.suptitle("Visualização da Tabela Q", fontsize=18, y=0.98)
            q_gs = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.5)
            
            q_table = self.agent.q_table
            map_size = self.agent.map_size
            state_count = map_size * map_size
            action_names = ['Esquerda', 'Baixo', 'Direita', 'Cima']
            
            for action in range(4):
                ax = q_fig.add_subplot(q_gs[action])
                q_values_grid = q_table[:state_count, action].reshape(map_size, map_size)
                im = ax.imshow(q_values_grid, cmap='hot')
                ax.set_title(f'Valores Q para: {action_names[action]}', fontsize=14)
                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.tick_params(labelsize=10)
                ax.grid(False)
                ax.tick_params(labelsize=10)
                
                # Add state numbers
                for i in range(map_size):
                    for j in range(map_size):
                        state = i * map_size + j
                        ax.text(j, i, f"{state}", ha="center", va="center", color="w", fontsize=12, fontweight='bold')
            
            q_fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.0)
            pdf.savefig(q_fig)
            plt.close(q_fig)
            
            # Optimal policy visualization page - make it fill most of the page
            policy_fig = plt.figure(figsize=(11.7, 8.3))
            policy_fig.suptitle("Política Ótima Aprendida", fontsize=18, y=0.98)
            
            # Use a slightly smaller subplot to leave room for the title
            ax = policy_fig.add_subplot(111)
            self._plot_optimal_policy_on_axis(ax)
            ax.title.set_fontsize(16)
            
            policy_fig.tight_layout(rect=[0, 0, 1, 0.95], pad=1.0)
            pdf.savefig(policy_fig)
            plt.close(policy_fig)
            
            # Set PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Frozen Lake Q-Learning - Relatório de Resultados'
            d['Author'] = 'Sistema de Aprendizado por Reforço'
            d['Subject'] = 'Visualização de Estatísticas e Resultados de Q-Learning'
            d['Keywords'] = 'Q-Learning, Reinforcement Learning, Frozen Lake'
            d['CreationDate'] = datetime.now()
            d['ModDate'] = datetime.now()
        
        print(f"PDF exportado com sucesso para: {filepath}")

if __name__ == "__main__":
    # Example usage
    # Option 1: Train a new agent and visualize results
    agent = QLearning(is_slippery=True, map_size=4)
    agent.train(num_episodes=5000)
    agent.save_model()
    
    stats = StatisticsVisualizer(agent)
    stats.visualize_all_stats()
    
    # Option 2: Load a pre-trained agent and visualize
    # agent = QLearning()
    # agent.load_model("../results/q_model_YYYYMMDD_HHMMSS.pkl")
    # stats = StatisticsVisualizer(agent)
    # stats.visualize_all_stats() 