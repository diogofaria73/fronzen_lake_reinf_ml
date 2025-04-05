import pygame
import numpy as np
import sys
import time
import os
import random
from pathlib import Path
import math

# Add the parent directory to the path to import training_code
sys.path.append(str(Path(__file__).resolve().parent.parent))
from training_code.q_learning import QLearning

class FrozenLakeVisualizer:
    """Pygame visualizer for the Frozen Lake environment"""
    
    # Colors - Using colors closer to the original Frozen Lake
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    ICE_BLUE = (220, 240, 255)     # Frozen lake (safe)
    HOLE_BLUE = (10, 70, 150)      # Hole (danger)
    START_GREEN = (40, 180, 40)    # Start position
    GOAL_RED = (200, 60, 60)       # Goal position
    AGENT_COLOR = (250, 150, 50)   # Agent (orange)
    GRID_LINE = (180, 200, 220)    # Grid lines
    HIGHLIGHT = (255, 255, 100)    # Highlight color
    TEXT_COLOR = (30, 30, 30)      # Dark text
    INFO_BG = (240, 240, 240, 200) # Semi-transparent info background
    
    # Direction mappings
    ACTIONS = {
        0: "←",  # Usando símbolos de seta em vez de texto
        1: "↓",
        2: "→",
        3: "↑"
    }
    
    # Textures and symbols for cells
    TEXTURES = {
        b'S': "S",     # Start
        b'F': "🧊",    # Frozen (usando emoji de gelo)
        b'H': "🌊",    # Hole (usando emoji de água)
        b'G': "🏆"     # Goal (usando emoji de troféu)
    }
    
    def __init__(self, agent, window_size=800, fps=5):
        """
        Initialize the visualizer
        
        Args:
            agent (QLearning): Trained Q-learning agent
            window_size (int): Size of the window in pixels
            fps (int): Frames per second for visualization
        """
        self.agent = agent
        self.env = agent.env
        self.map_size = agent.map_size
        
        # Window layout with sidebar for info
        self.main_size = window_size
        self.sidebar_width = 300
        self.window_width = window_size + self.sidebar_width
        self.window_height = window_size
        self.cell_size = (window_size - 20) // self.map_size  # Smaller grid with padding
        self.grid_offset_x = 10  # Padding from left
        self.grid_offset_y = 10  # Padding from top
        self.fps = fps
        
        # Animation state
        self.animations = []
        self.render_q_values = True
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Frozen Lake Q-Learning Visualization')
        self.clock = pygame.time.Clock()
        
        # Load fonts
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.info_font = pygame.font.SysFont('Arial', 18)
        self.cell_font = pygame.font.SysFont('Arial', 22, bold=True)
        self.q_font = pygame.font.SysFont('Arial', 14)
        
        # Get the map description (positions of holes, start, goal)
        self.desc = self.env.unwrapped.desc
        
        # Create a cleaner background with ice pattern
        self.background = self._create_background()
        
        # Load or create character image for agent
        self.agent_img = self._create_agent_image()
    
    def _create_background(self):
        """Create a textured background for the ice"""
        bg = pygame.Surface((self.main_size, self.main_size))
        bg.fill(self.ICE_BLUE)
        
        # Add subtle ice pattern
        for _ in range(200):
            x = random.randint(0, self.main_size - 1)
            y = random.randint(0, self.main_size - 1)
            radius = random.randint(1, 3)
            alpha = random.randint(10, 40)
            
            # Create a small surface for the circle with alpha
            circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (255, 255, 255, alpha), (radius, radius), radius)
            bg.blit(circle_surf, (x, y))
            
        return bg
    
    def _create_agent_image(self):
        """Create a simple character image for the agent"""
        size = self.cell_size // 2
        agent_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw a simple person-like shape
        # Head
        pygame.draw.circle(agent_surf, self.AGENT_COLOR, (size//2, size//3), size//4)
        
        # Body
        pygame.draw.rect(agent_surf, self.AGENT_COLOR, 
                         pygame.Rect(size//3, size//3, size//3, size//2))
        
        # Legs
        pygame.draw.rect(agent_surf, self.AGENT_COLOR, 
                         pygame.Rect(size//3, size*2//3, size//6, size//3))
        pygame.draw.rect(agent_surf, self.AGENT_COLOR, 
                         pygame.Rect(size//2, size*2//3, size//6, size//3))
        
        return agent_surf
    
    def get_cell_rect(self, row, col):
        """Get the rectangle for a cell position with offset"""
        return pygame.Rect(
            self.grid_offset_x + col * self.cell_size,
            self.grid_offset_y + row * self.cell_size,
            self.cell_size,
            self.cell_size
        )
    
    def draw_grid(self):
        """Draw the Frozen Lake grid with improved visuals"""
        # Draw background first
        self.screen.blit(self.background, (0, 0))
        
        # Draw each cell
        for row in range(self.map_size):
            for col in range(self.map_size):
                cell_type = self.desc[row][col]
                rect = self.get_cell_rect(row, col)
                
                # Draw cell based on type
                if cell_type == b'F':  # Ice - transparent with texture
                    pygame.draw.rect(self.screen, self.ICE_BLUE, rect, 0, 3)
                    # Add sparkle effect
                    if random.random() < 0.05:  # Occasionally add sparkle
                        sparkle_pos = (rect.x + random.randint(5, rect.width-5),
                                      rect.y + random.randint(5, rect.height-5))
                        pygame.draw.circle(self.screen, self.WHITE, sparkle_pos, 1)
                
                elif cell_type == b'H':  # Hole - water
                    pygame.draw.rect(self.screen, self.HOLE_BLUE, rect, 0, 3)
                    # Add wave effect
                    for i in range(3):
                        wave_y = rect.y + rect.height//4 + i*rect.height//4
                        wave_width = rect.width - 10
                        pygame.draw.line(self.screen, (100, 150, 255), 
                                        (rect.x + 5, wave_y),
                                        (rect.x + 5 + wave_width, wave_y), 2)
                
                elif cell_type == b'G':  # Goal
                    pygame.draw.rect(self.screen, self.GOAL_RED, rect, 0, 3)
                    # Add goal marker
                    pygame.draw.circle(self.screen, self.WHITE, rect.center, rect.width//4)
                    text = self.cell_font.render('G', True, self.GOAL_RED)
                    self.screen.blit(text, text.get_rect(center=rect.center))
                
                elif cell_type == b'S':  # Start
                    pygame.draw.rect(self.screen, self.START_GREEN, rect, 0, 3)
                    text = self.cell_font.render('S', True, self.WHITE)
                    self.screen.blit(text, text.get_rect(center=rect.center))
                
                # Add grid border
                pygame.draw.rect(self.screen, self.GRID_LINE, rect, 1, 3)
    
    def draw_agent(self, state, animation_offset=(0,0)):
        """Draw the agent at the current state with optional animation"""
        # Convert state to row and column
        row = state // self.map_size
        col = state % self.map_size
        
        rect = self.get_cell_rect(row, col)
        center_x = rect.centerx + animation_offset[0]
        center_y = rect.centery + animation_offset[1]
        
        # Draw the agent
        agent_rect = self.agent_img.get_rect(center=(center_x, center_y))
        self.screen.blit(self.agent_img, agent_rect)
    
    def draw_q_values(self, state):
        """Draw the Q-values for the current state"""
        if not self.render_q_values:
            return
            
        q_values = self.agent.q_table[state]
        
        # Convert state to row and column
        row = state // self.map_size
        col = state % self.map_size
        rect = self.get_cell_rect(row, col)
        
        # Find best action
        best_action = np.argmax(q_values)
        
        # Draw arrows with Q-values
        arrow_colors = [(200, 200, 200) for _ in range(4)]
        arrow_colors[best_action] = self.HIGHLIGHT
        
        # Position offsets for each direction
        offsets = [
            (-self.cell_size//2, 0),  # LEFT
            (0, self.cell_size//2),   # DOWN
            (self.cell_size//2, 0),   # RIGHT
            (0, -self.cell_size//2)   # UP
        ]
        
        for action, q_value in enumerate(q_values):
            # Calculate position for the Q-value
            x = rect.centerx + offsets[action][0]
            y = rect.centery + offsets[action][1]
            
            # Draw small colored circles at arrow positions
            radius = 12
            pygame.draw.circle(self.screen, arrow_colors[action], (x, y), radius)
            
            # Draw arrow symbol in black
            text = self.q_font.render(self.ACTIONS[action], True, self.BLACK)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
            
            # Draw Q-value below the arrow
            q_text = self.q_font.render(f"{q_value:.2f}", True, self.BLACK)
            q_rect = q_text.get_rect(center=(x, y + radius + 10))
            self.screen.blit(q_text, q_rect)
    
    def draw_sidebar(self, state, steps, total_reward, episode_num, num_episodes):
        """Draw information sidebar"""
        # Draw sidebar background
        sidebar_rect = pygame.Rect(self.main_size, 0, self.sidebar_width, self.window_height)
        pygame.draw.rect(self.screen, (240, 240, 240), sidebar_rect)
        pygame.draw.line(self.screen, (200, 200, 200), 
                         (self.main_size, 0), 
                         (self.main_size, self.window_height), 2)
        
        # Title
        title = self.title_font.render("Frozen Lake Q-Learning", True, self.BLACK)
        self.screen.blit(title, (self.main_size + 20, 20))
        
        # Episode info
        episode_text = self.info_font.render(f"Episode: {episode_num}/{num_episodes}", True, self.BLACK)
        self.screen.blit(episode_text, (self.main_size + 20, 60))
        
        # Current info section
        pygame.draw.rect(self.screen, (230, 230, 230), 
                         pygame.Rect(self.main_size + 10, 100, self.sidebar_width - 20, 150), 0, 5)
        
        info_texts = [
            f"Steps: {steps}",
            f"State: {state}",
            f"Reward: {total_reward}",
            f"Action: {self.ACTIONS[np.argmax(self.agent.q_table[state])]}"
        ]
        
        for i, text in enumerate(info_texts):
            info_surface = self.info_font.render(text, True, self.BLACK)
            self.screen.blit(info_surface, (self.main_size + 20, 110 + i * 30))
        
        # Q-value table section
        pygame.draw.rect(self.screen, (230, 230, 230),
                         pygame.Rect(self.main_size + 10, 280, self.sidebar_width - 20, 200), 0, 5)
        
        q_title = self.title_font.render("Q-Values", True, self.BLACK)
        self.screen.blit(q_title, (self.main_size + 20, 290))
        
        # Display current state Q-values
        q_values = self.agent.q_table[state]
        action_names = ["Left", "Down", "Right", "Up"]
        
        for i, (action, q_value) in enumerate(zip(action_names, q_values)):
            color = self.GOAL_RED if q_value == max(q_values) else self.BLACK
            q_text = self.info_font.render(f"{action}: {q_value:.4f}", True, color)
            self.screen.blit(q_text, (self.main_size + 20, 330 + i * 30))
        
        # Toggle instructions
        toggle_text = self.info_font.render("Press 'Q' to toggle Q-values", True, self.BLACK)
        self.screen.blit(toggle_text, (self.main_size + 20, self.window_height - 60))
    
    def animate_movement(self, state, next_state, action):
        """Create animation data for agent movement"""
        # Calculate source and destination positions
        src_row, src_col = state // self.map_size, state % self.map_size
        dst_row, dst_col = next_state // self.map_size, next_state % self.map_size
        
        # Source and destination coordinates
        src_rect = self.get_cell_rect(src_row, src_col)
        dst_rect = self.get_cell_rect(dst_row, dst_col)
        
        # Create animation frames for smooth movement
        frames = 10
        dx = (dst_rect.centerx - src_rect.centerx) / frames
        dy = (dst_rect.centery - src_rect.centery) / frames
        
        animation_frames = []
        for i in range(frames + 1):
            offset_x = dx * i
            offset_y = dy * i
            animation_frames.append((offset_x, offset_y))
        
        return animation_frames
    
    def visualize_episode(self, episode_num, num_episodes, max_steps=100):
        """Visualize a single episode using the trained policy"""
        state, _ = self.env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not done and not truncated and steps < max_steps:
            # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.render_q_values = not self.render_q_values
            
            # Choose best action
            action = np.argmax(self.agent.q_table[state])
            
            # Take action
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # Create animation for movement
            animation_frames = self.animate_movement(state, next_state, action)
            
            # Draw frames with animation
            for frame, offset in enumerate(animation_frames):
                # Draw base screen
                self.draw_grid()
                
                # Draw agent with animation offset if not last frame
                if frame < len(animation_frames) - 1:
                    self.draw_agent(state, offset)
                else:
                    self.draw_agent(next_state)
                
                # Draw Q-values for current state (only if not animating)
                if frame == 0:
                    self.draw_q_values(state)
                
                # Draw sidebar information
                self.draw_sidebar(state, steps, total_reward, episode_num, num_episodes)
                
                # Update display
                pygame.display.flip()
                self.clock.tick(self.fps * 2)  # Faster animation
            
            # Update state and metrics
            state = next_state
            total_reward += reward
            steps += 1
        
        # Show final state
        self.draw_grid()
        self.draw_agent(state)
        self.draw_q_values(state)
        self.draw_sidebar(state, steps, total_reward, episode_num, num_episodes)
        
        # Display result overlay
        if total_reward > 0:
            self.show_result_overlay("SUCESSO!", self.START_GREEN)
        else:
            self.show_result_overlay("FALHA!", self.GOAL_RED)
        
        pygame.display.flip()
        
        # Wait a bit before next episode
        time.sleep(2)
        
        return total_reward > 0
    
    def show_result_overlay(self, message, color):
        """Show a result overlay with the given message"""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.main_size, self.main_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Draw message
        font = pygame.font.SysFont('Arial', 48, bold=True)
        text = font.render(message, True, color)
        text_rect = text.get_rect(center=(self.main_size // 2, self.main_size // 2))
        
        # Add a background to the text
        padding = 20
        bg_rect = pygame.Rect(text_rect.x - padding, text_rect.y - padding,
                              text_rect.width + padding * 2, text_rect.height + padding * 2)
        pygame.draw.rect(self.screen, (255, 255, 255, 200), bg_rect, 0, 10)
        pygame.draw.rect(self.screen, color, bg_rect, 3, 10)
        
        # Draw the text
        self.screen.blit(text, text_rect)
    
    def visualize_multiple_episodes(self, num_episodes=5):
        """Visualize multiple episodes"""
        success_count = 0
        
        for episode in range(num_episodes):
            # Display episode title screen
            self.screen.fill(self.BLACK)
            episode_text = self.title_font.render(f"Episódio {episode + 1}/{num_episodes}", True, self.WHITE)
            self.screen.blit(episode_text, (self.window_width // 2 - episode_text.get_width() // 2, 
                                           self.window_height // 2 - 50))
            
            # Add instructions
            instr_text = self.info_font.render("Aguarde o início da simulação...", True, self.WHITE)
            self.screen.blit(instr_text, (self.window_width // 2 - instr_text.get_width() // 2, 
                                          self.window_height // 2 + 10))
            
            pygame.display.flip()
            time.sleep(1)
            
            # Run episode
            success = self.visualize_episode(episode + 1, num_episodes)
            if success:
                success_count += 1
        
        # Return success rate for displaying in statistics
        return success_count / num_episodes

if __name__ == "__main__":
    # Example usage
    agent = QLearning(is_slippery=True, map_size=4)
    agent.train(num_episodes=5000)
    agent.save_model()
    
    visualizer = FrozenLakeVisualizer(agent)
    visualizer.visualize_multiple_episodes(num_episodes=5) 