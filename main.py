"""App entrypoint for launching the Compression Workbench UI."""

# COMP6049001 - Algorithm Design and Analysis
# Final Project: A Comparative Analysis of Image Compression Algorithms
# Group Members:
# - Harris Ekaputra Suryadi (2802400502)
# - Michael Arianno Chandrarieta (2802499711)
# - Muhammad Ryan Ismail Putra (2802522733)

import tkinter as tk

from app_ui import ImageCompressorApp

def main():
    # Start Tkinter main loop with the compressor UI wired in.
    root = tk.Tk()
    app = ImageCompressorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()