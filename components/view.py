import os

from components import frames, menu
from utils.functions import set_window_center
from PIL import Image, ImageTk


class MainPage:

    def __init__(self, master=None):
        self.root = master
        self.root.resizable(False, False)
        set_window_center(self.root, 1000, 720)
        menu.MainMenu(self)

        # Get the current directory
        _dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(_dir, "..", "img", "heart-logo.ico")
        logo = Image.open(logo_path)
        logo = ImageTk.PhotoImage(logo)
        self.root.iconphoto(False, logo)
        # Frames
        self.current_frame = None
        self.page_frame = {
            # "home": frames.HomeFrame,
            "home": frames.PredictFrame,
            "settings": frames.SettingsFrame,
            "contact": frames.AboutFrame,
        }
        self.open_home()
        self.win_about = None

    def open_page(self, frame_name, title):
        """open page method"""
        self.root.title(title)
        # selct frame
        if self.current_frame is not None and (
            hasattr(self.current_frame.destroy, "__call__")
        ):
            self.current_frame.destroy()

        self.current_frame = self.page_frame[frame_name](self.root)
        self.current_frame.pack()

    def open_home(self):
        """App main interface"""
        self.open_page("home", "CToxPred2")

    def open_settings(self):
        """App settings interface"""
        self.open_page("settings", "CToxPred2")
