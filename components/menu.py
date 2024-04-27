from tkinter import Menu, messagebox, filedialog
import utils.global_variable as glv
import os


class MainMenu:

    def __init__(self, master):
        self.master = master
        self.root = master.root
        self.init_menu()

    def init_menu(self):
        """Initialize menu"""

        self.menubar = Menu(self.root)
        self.root.config(menu=self.menubar)

        filemenu = Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Home", command=self.master.open_home)
        filemenu.add_command(label="Export", command=self.file_export)
        filemenu.add_command(label="Settings", command=self.master.open_settings)
        filemenu.add_separator()
        filemenu.add_command(label="quit", command=self.root.quit)

        # help dropdown
        helpmenu = Menu(self.menubar, tearoff=0)

        helpmenu.add_command(label="about", command=self.help_about)

        # Add drop down menu to menu bar
        self.menubar.add_cascade(label="App", menu=filemenu)
        self.menubar.add_cascade(label="help", menu=helpmenu)

    def file_open(self):
        messagebox.showinfo("Open", "File-Open!")  # message box

    def file_new(self):
        messagebox.showinfo("New", "File-New!")  # Message prompt box

    def file_export(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            output_table = glv.GLB_SMILES[
                [
                    "ID",
                    "InChI",
                    "MW",
                    "MPSA",
                    "AlogP",
                    "HBA",
                    "ALERTS",
                    "AROMS",
                    "ROTB",
                    "HBD",
                    "hERG",
                    "hERG_confidence",
                    "Nav1.5",
                    "Nav1.5_confidence",
                    "Cav1.2",
                    "Cav1.2_confidence",
                    "SMILES",
                ]
            ].sort_values(by="ID")
            output_table.rename(
                columns={
                    "hERG": "hERG_inhibitor",
                    "Nav1.5": "Nav1.5_inhibitor",
                    "Cav1.2": "Cav1.2_inhibitor",
                },
                inplace=True,
            )
            path = [folder_selected, "CToxPred2_export.csv"]
            output_table.to_csv(os.path.join(*path), index=False)
            messagebox.showinfo("Save", "File saved to: " + os.path.join(*path))
        else:
            messagebox.showinfo(
                "Save", "File not saved! No folder destination " "selected."
            )

    def help_about(self):
        """about"""
        messagebox.showinfo(
            "About",
            "CToxPred2: a comprehensive cardiotoxicity prediction "
            "tool.\n\n"
            " The software computes the following:\n\n"
            "- InChI key: International Chemical Identifier of the "
            "molecule\n"
            "- 2D structure of the molecule\n"
            "- MW: The molecular weight in Daltons - Da\n"
            "- MPSA: The molecular polar surface area in Ångström - Å²\n"
            "- AlogP: The octanol-water partition coefficient\n"
            "- HBA: The number of hydrogen bond acceptors\n"
            "- HBD: The number of hydrogen bond donors\n"
            "- ROTB: The number of rotatable bonds\n"
            "- AROMS: The number of aromatic rings\n"
            "- ALERTS: The count of present unwanted chemical functionalities\n"
            "- Prediction of hERG, Nav1.5, and Cav1.2 inhibition "
            "along with the model confidence score.\n\n"
            "Copyright © 2024 - issar.arab@tum.de. All rights "
            "reserved.",
        )
