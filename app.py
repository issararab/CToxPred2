import warnings
from tkinter import Tk


import utils.global_variable as glv
from components.view import MainPage
from CToxPred2.pairwise_correlation import CorrelationThreshold


class App(Tk):
    """
    This class serves as the main application window.
    """

    def __init__(self):
        """
        This method initializes the main application window.
        """
        Tk.__init__(self)
        MainPage(self)
        self.mainloop()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    glv.init_global_variable()
    App()
