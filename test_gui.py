import tkinter as tk
from tournament_gui import SetupDialog, BracketWindow

root = tk.Tk()
root.withdraw()
setup = {
    'num_players': 8,
    'draw_size': 8,
    'num_byes': 0,
    'surface': 'Hard',
    'level': 'M',
    'name': 'Test'
}
bw = BracketWindow(root, setup)
root.wait_window(bw)
root.destroy()
