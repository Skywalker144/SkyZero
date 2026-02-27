import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
import sys

# Ensure the script works when run from the root directory, similar to the CLI script
sys.path.append(os.getcwd())

from gomoku import Gomoku

class GomokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Gomoku (Human vs Human)')
        
        # Game initialization
        try:
            self.game = Gomoku()
        except Exception as e:
            messagebox.showerror('Error', f'Failed to initialize game: {e}')
            root.destroy()
            return

        self.board_size = self.game.board_size
        self.cell_size = 40
        self.margin = 30
        self.canvas_size = self.cell_size * (self.board_size - 1) + 2 * self.margin

        # State management
        self.state_history = [] # Stack to store history for undo
        self.init_game()

        # UI Components
        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='#F0D9B5')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_click)

        self.info_label = tk.Label(root, text='Black's Turn', font=('Arial', 14))
        self.info_label.pack(pady=5)

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=5)

        self.btn_undo = tk.Button(self.btn_frame, text='Undo (撤回)', command=self.undo_move, font=('Arial', 12))
        self.btn_undo.pack(side=tk.LEFT, padx=10)

        self.btn_restart = tk.Button(self.btn_frame, text='Restart', command=self.restart_game, font=('Arial', 12))
        self.btn_restart.pack(side=tk.LEFT, padx=10)

        self.draw_board()

    def init_game(self):
        initial_state = self.game.get_initial_state()
        self.state_history = [initial_state]
        self.game_over = False
        self.winner = None

    def get_current_state(self):
        return self.state_history[-1]

    def get_current_player(self):
        # Determine current player based on stone count
        # Black = 1, White = -1
        current_board = self.get_current_state()[-1]
        stone_count = np.sum(current_board != 0)
        return 1 if stone_count % 2 == 0 else -1

    def draw_board(self):
        self.canvas.delete('all')
        
        # Draw grid lines
        for i in range(self.board_size):
            # Horizontal
            start_x = self.margin
            end_x = self.canvas_size - self.margin
            y = self.margin + i * self.cell_size
            self.canvas.create_line(start_x, y, end_x, y)
            
            # Vertical
            start_y = self.margin
            end_y = self.canvas_size - self.margin
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, start_y, x, end_y)

        # Draw star points (standard 15x15 points)
        star_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
        for r, c in star_points:
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            r_dot = 3
            self.canvas.create_oval(x - r_dot, y - r_dot, x + r_dot, y + r_dot, fill='black')

        # Draw stones
        current_board = self.get_current_state()[-1]
        rows, cols = current_board.shape
        for r in range(rows):
            for c in range(cols):
                if current_board[r, c] != 0:
                    self.draw_stone(r, c, current_board[r, c])
        
        # Highlight last move if exists
        if len(self.state_history) > 1:
            prev_board = self.state_history[-2][-1]
            curr_board = self.state_history[-1][-1]
            diff = curr_board - prev_board
            # Find the changed cell
            indices = np.where(diff != 0)
            if len(indices[0]) > 0:
                last_r, last_c = indices[0][0], indices[1][0]
                self.highlight_last_move(last_r, last_c)

    def draw_stone(self, row, col, player):
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        r = self.cell_size // 2 - 2
        color = 'black' if player == 1 else 'white'
        outline = 'white' if player == 1 else 'black' # Contrast outline for visibility
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline=color)

    def highlight_last_move(self, row, col):
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        r = 3
        self.canvas.create_rectangle(x - r, y - r, x + r, y + r, fill='red', outline='red')

    def on_click(self, event):
        if self.game_over:
            return

        # Convert coordinates to grid index
        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)

        # Check bounds
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return

        state = self.get_current_state()
        current_board = state[-1]

        # Check if occupied
        if current_board[row, col] != 0:
            return

        # Check legality
        action = row * self.board_size + col
        legal_moves = self.game.get_is_legal_actions(state)
        
        if not legal_moves[action]:
            # Illegal move (e.g. forbidden for Black)
            # We can give feedback, but usually silent is fine or status bar update
            player = self.get_current_player()
            if player == 1:
                self.info_label.config(text='Illegal Move (Forbidden Rule)')
            return

        # Execute move
        player = self.get_current_player()
        next_state = self.game.get_next_state(state, action, player)
        self.state_history.append(next_state)
        
        # Update UI
        self.draw_board()
        
        # Check Winner
        winner = self.game.get_winner(next_state)
        if winner is not None:
            self.game_over = True
            if winner == 1:
                msg = 'Black Wins!'
            elif winner == -1:
                msg = 'White Wins!'
            else:
                msg = 'Draw!'
            self.info_label.config(text=msg)
            messagebox.showinfo('Game Over', msg)
        else:
            # Update turn info
            next_player = self.get_current_player() # Recalculate based on new state
            p_name = 'Black' if next_player == 1 else 'White'
            self.info_label.config(text=f'{p_name}'s Turn')

    def undo_move(self):
        if len(self.state_history) > 1:
            self.state_history.pop()
            self.game_over = False # Reset game over state if we undo
            self.winner = None
            self.draw_board()
            
            player = self.get_current_player()
            p_name = 'Black' if player == 1 else 'White'
            self.info_label.config(text=f'{p_name}'s Turn')
        else:
            messagebox.showinfo('Info', 'Cannot undo further.')

    def restart_game(self):
        if messagebox.askyesno('Restart', 'Are you sure you want to restart?'):
            self.init_game()
            self.draw_board()
            self.info_label.config(text='Black's Turn')

if __name__ == '__main__':
    root = tk.Tk()
    # Center window
    # app = GomokuGUI(root)
    # root.mainloop()
    
    # Simple centering hack if needed, but let's just run it
    gui = GomokuGUI(root)
    root.mainloop()
