import os
from io import BytesIO
from tkinter import (
    Button,
    DISABLED,
    END,
    Entry,
    Frame,
    filedialog,
    HORIZONTAL,
    Label,
    LabelFrame,
    messagebox,
    NORMAL,
    OptionMenu,
    PhotoImage,
    Radiobutton,
    StringVar,
    SUNKEN,
    scrolledtext,
    ttk,
)


from PIL import Image, ImageTk

import utils.global_variable as glv
from utils.functions import treeview_sort_column, process_smiles


class AboutFrame(Frame):
    """
    A class representing an about interface.

    This class represents an interface for displaying information about the application.
    """

    def __init__(self, parent=None):
        """
        Initialize the AboutFrame.

        Parameters:
            parent: The parent widget (usually a Tkinter window) for the AboutFrame.
        """
        Frame.__init__(self, parent)
        self.root = parent
        self.init_page()

    def init_page(self):
        """
        Load controls onto the AboutFrame.

        This method initializes and loads controls onto the AboutFrame, such as labels.
        """
        Label(self, text="For").grid()
        Label(
            self,
            text="Similar to a popup window, with independent window properties.",
            width=150,
        ).grid()


class PredictFrame(Frame):
    """ "
    This class represents an interface for displaying the  physicochemical
    properties of a molecule as long as the cardiotoxicity predictions.
    """

    def __init__(self, parent=None):
        """
        Initialize the PredictFrame.

        Parameters:
            parent: The parent widget (usually a Tkinter window) for the PredictFrame.
        """
        Frame.__init__(self, parent)
        self.root = parent
        self.list = []
        self.selected_item = None
        self.selected_name = StringVar()
        self.selected_id = None
        self.progress_message = StringVar()
        self.selected_first_name = StringVar()
        self.selected_last_name = StringVar()
        self.init_page()

    def init_page(self):
        """
        Initialize the page.

        This method initializes the PredictFrame interface.
        """
        self.list = glv.GLB_SMILES

        sub_foot_frame = LabelFrame(self)
        sub_foot_frame.grid(row=2, column=0, columnspan=2, sticky="nswe", pady=10)
        _dir = os.path.dirname(os.path.abspath(__file__))
        file_icon_path = os.path.join(_dir, "..", "img", "file_2.ico")
        file_icon = Image.open(file_icon_path)
        file_icon = file_icon.resize((16, 16), Image.ANTIALIAS)
        file_icon = ImageTk.PhotoImage(file_icon)
        btn_assign = Button(
            sub_foot_frame,
            text="Select a SMILES File",
            image=file_icon,
            cursor="hand2",
            compound="left",
            command=self.open_and_read_file,
            padx=5,
        )
        btn_assign.image = file_icon
        btn_assign.pack(side="left", fill="both", expand=True)

        ## Progress bar
        progress_frame = LabelFrame(self, relief="flat")
        progress_frame.place(anchor="center")
        progress_frame.grid(row=15, column=0, columnspan=2, sticky="nswe")
        Label(
            progress_frame, textvariable=self.progress_message, foreground="green"
        ).pack(pady=5)

        self.show_progress_message()
        self.hide_progress_message()

        # Selected Row InChI Key
        head_frame = LabelFrame(self, text="InChI Key", font="Arial 10 bold underline")
        head_frame.place(anchor="center")
        head_frame.grid(row=16, column=0, columnspan=2, sticky="nswe")
        Label(head_frame, textvariable=self.selected_name).pack()

        # Configure the style of Heading in Treeview widget
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Treeview.Heading", background="#BAB9B9")
        s.map("Treeview", background=[("selected", "#258096")])  # 45BAE1
        style = ttk.Style()
        style.layout("my.Treeview")
        style.configure(
            "my.Treeview.Heading", background="lightgrey", font=("Arial Bold", 8)
        )
        style.configure(
            "Green.Horizontal.TProgressbar", troughcolor="white", background="green"
        )
        self.tree_view = ttk.Treeview(
            self, selectmode="browse", show="headings", height="6", style="my.Treeview"
        )

        self.tree_view["columns"] = ("ID", "InChI", "SMILES")

        # show header
        self.tree_view.heading("ID", text="ID")
        self.tree_view.heading("InChI", text="InChI")
        self.tree_view.heading("SMILES", text="SMILES")

        # insert data
        num = 1
        for idx, item in self.list.iterrows():
            self.tree_view.insert(
                "", num, text="", values=(item["ID"], item["InChI"], item["SMILES"])
            )

        # select row
        self.tree_view.bind("<<TreeviewSelect>>", self.select)

        # sort
        for col in self.tree_view["columns"]:  # add all headers
            self.tree_view.heading(
                col,
                text=col,
                command=lambda _col=col: treeview_sort_column(
                    self.tree_view, _col, False
                ),
            )
        # Vertical Scrollbar
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.tree_view.yview)
        vbar.grid(row=8, column=1, sticky="ns")

        # Horizontal Scrollbar
        hbar = ttk.Scrollbar(self, orient="horizontal", command=self.tree_view.xview)
        hbar.grid(row=9, column=0, sticky="ew")

        # Configure the Treeview to update the horizontal scrollbar
        self.tree_view.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.tree_view.grid(row=8, column=0, sticky="nsew")

        # Physico-chemical properties frame

        # Image placeholder
        blank_image = Image.new("RGB", (300, 300), "white")
        blank_img_byte_array = BytesIO()
        blank_image.save(blank_img_byte_array, format="PNG")
        blank_img_byte_array = blank_img_byte_array.getvalue()
        blank_photo = ImageTk.PhotoImage(data=blank_img_byte_array)

        # Add a label to display the image
        grid_frame_main = LabelFrame(self, relief="flat")
        grid_frame_main.grid(
            row=23, column=0, sticky="nswe", rowspan=2, padx=10, pady=10
        )
        grid_frame = LabelFrame(grid_frame_main)
        grid_frame.grid(row=23, column=0, sticky="nswe", rowspan=2, padx=10, pady=10)
        grid_frame_img = LabelFrame(grid_frame_main)
        grid_frame_img.grid(
            row=23, column=1, sticky="nswe", rowspan=2, padx=10, pady=10
        )
        self.img_label = Label(grid_frame_img, image=blank_photo)
        self.img_label.image = blank_photo
        self.img_label.pack(side="right")

        # Physico-chemical properties Grid

        self.physico_chemical_props = [
            ["MW", "MPSA", "AlogP", "HBA"],
            ["ALERTS", "AROMS", "ROTB", "HBD"],
        ]
        self.units = [["(Da)", "(Å²)", "", ""], ["", "", "", ""]]

        self.physico_chemical_labels = [[], []]
        # Upper row, split into 2 rows and 4 columns
        for i in range(2):
            grid_frame.grid_rowconfigure(i, weight=1)
            for j in range(4):
                grid_frame.grid_columnconfigure(j, weight=1)
                sub_frame = LabelFrame(
                    grid_frame,
                    text=self.physico_chemical_props[i][j] + " : ",
                    font=("Arial", 11, "bold"),
                    bg="lightgrey",
                )
                sub_frame.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")

                # Add the label with specified formatting
                self.physico_chemical_labels[i].append(
                    Label(
                        sub_frame,
                        text="ABC",
                        font=("Arial", 12, "normal"),
                        anchor="center",
                        bg="lightgrey",
                        pady=30,
                    )
                )

                self.physico_chemical_labels[i][-1].pack(fill="both")

        grid_frame = LabelFrame(self)
        grid_frame.grid(row=25, column=0, sticky="nswe", rowspan=1, padx=5, pady=2)
        self.ion_channels = ["hERG", "Nav1.5", "Cav1.2"]
        self.ion_channel_labels = []

        # Create the lower row, split into 3 columns
        for i in range(3):
            grid_frame.grid_columnconfigure(i, weight=1, minsize=150)
            sub_frame = LabelFrame(
                grid_frame,
                text=self.ion_channels[i] + ": ",
                bg="lightgrey",
                font=("Arial", 10, "bold"),
            )
            sub_frame.grid(row=1, column=i, padx=5, pady=5, sticky="nsew")
            self.ion_channel_labels.append(
                Label(
                    sub_frame, text="ABC", font=("Arial", 10, "normal"), bg="lightgrey"
                )
            )
            self.ion_channel_labels[-1].pack(side="left", fill="both")

        first_row_id = self.tree_view.get_children()[0]
        self.tree_view.selection_set(first_row_id)
        self.select(None)

    def hide_progress_message(self):
        self.progress_message.set("")

    def show_progress_message(self):
        self.progress_message.set("CALCULATING...")
        self.update()

    def select(self, event):
        """Select row and update display fields"""

        try:
            slct = event.widget.selection()[0]
            # Set InChI key
            self.selected_id = self.tree_view.item(slct)["values"][0]
            self.selected_item = self.list.loc[self.list["ID"] == self.selected_id]
            self.selected_name.set(str(self.selected_item["InChI"].values[0]))

            # Set picture
            img = self.selected_item["_2d_structure"].values[0]
            # Convert the image to bytes
            img_byte_array = BytesIO()
            img.save(img_byte_array, format="PNG")
            img_byte_array = img_byte_array.getvalue()

            # Create a Tkinter PhotoImage from the bytes
            new_photo = ImageTk.PhotoImage(data=img_byte_array)

            self.img_label.configure(image=new_photo)
            self.img_label.image = new_photo

            # Assign the properties
            for i in range(2):
                for j in range(4):
                    self.physico_chemical_labels[i][j].config(
                        text=str(
                            self.selected_item[
                                self.physico_chemical_props[i][j]
                            ].values[0]
                        )
                        + "\n"
                        + self.units[i][j]
                    )

            # Assign prediction
            for i in range(3):

                self.ion_channel_labels[i].config(
                    text=self.selected_item[
                        self.ion_channels[i] + "_confidence"
                    ].values[0]
                    + (
                        " Toxic"
                        if self.selected_item[self.ion_channels[i]].values[0]
                        else " Non-Toxic"
                    ),
                    foreground=(
                        "red"
                        if self.selected_item[self.ion_channels[i]].values[0]
                        else "green"
                    ),
                    font=("Arial", 10, "bold"),
                )
        except:
            glv.LOGGER.info("GUI components refreshed!")

    def open_and_read_file(self):
        file_path = (
            filedialog.askopenfilename()
        )  # Open the file dialog to choose a file

        if file_path:  # Check if a file was selected
            if file_path.endswith((".smi", ".smiles")):  # Check the file extension
                try:
                    # Read and display the content of the chosen file
                    with open(file_path, "r") as file:
                        file_content = file.read()
                    smiles_mols = file_content.strip().split("\n")
                    if len(smiles_mols) > glv.GLB_MAXIMUM_COMPOUNDS:
                        messagebox.showinfo(
                            "Error",
                            "Make sure the number of "
                            "SMILES provided is less or "
                            "equal to the max number "
                            "of compounds configured for "
                            "processing! "
                            "\n\nPlease review your settings "
                            "configuration!",
                        )
                    else:
                        self.show_progress_message()
                        glv.GLB_SMILES = process_smiles(
                            [sentence.strip() for sentence in smiles_mols]
                        )
                        # refresh table
                        self.reset()
                except Exception as e:
                    glv.LOGGER.error("Error reading file: %s", str(e))
            else:
                messagebox.showinfo(
                    "File selector",
                    "Wrong file selected!\n"
                    "Make sure your file has one of these "
                    "extension: .smi OR .smiles",
                )
        else:
            messagebox.showinfo("File selector", "No file was selected!")

    def reset(self, *args):
        """Compounds list reset"""
        # Clear tree entries
        for i in self.tree_view.get_children():
            self.tree_view.delete(i)
        # Load new data to list
        self.list = glv.GLB_SMILES
        self.hide_progress_message()
        # Populate the table
        num = 1
        for idx, item in self.list.iterrows():
            self.tree_view.insert(
                "",
                num,
                text="",
                values=(item["ID"], item["InChI"], item["SMILES"]),
            )

        # Set the width of the "SMILES" column
        self.tree_view.heading("ID", text="ID")
        self.tree_view.heading("InChI", text="InChI")
        self.tree_view.heading("SMILES", text="SMILES")


class SettingsFrame(Frame):
    """
    This class represents an interface for managing model settings to use
    for predictions.
    """

    def __init__(self, parent=None):
        """
        Initialize the SettingsFrame.

        Parameters:
            parent: The parent widget (usually a Tkinter window) for the SettingsFrame.
        """
        Frame.__init__(self, parent)
        # Variable to track selected choice
        self.selected_model = None
        self.forward_iterations_options = None
        self.compounds_batch_options = None
        self.dropout_rate_options = None
        self.root = parent
        self.init_page()

    def init_page(self):
        """
        Initialize the page.

        This method initializes the settings interface page.
        """
        page_head_frame = Frame(
            self, highlightbackground="grey", highlightthickness=0.5
        )
        page_head_frame.pack(side="top", fill="x", pady=20)
        headerLabel = Label(
            page_head_frame,
            text="Settings",
            font="Helvetica " "10 bold",
            width=50,
            anchor="center",
        )
        headerLabel.pack(fill="x", padx=220)
        
        # Add button
        foot_frame = Frame(self)
        foot_frame.pack(side="bottom", fill="x", pady=10)
        ##Inner Frame
        inner_frame = Frame(
            foot_frame, highlightbackground="grey", highlightthickness=1
        )
        inner_frame.pack(side="bottom", fill="x", padx=80)
        btn_add = Button(
            inner_frame,
            text="Save",
            cursor="hand2",
            command=lambda: self.save(
                ForwardIterationVariable, DropeoutRateVariable, maxCompoundVariable
            ),
        )
        btn_add.pack(side="bottom", fill="x", ipadx=20)

        ##Fields

        # Divisions
        left_frame = Frame(self)  # ,bg='red')
        left_frame.pack(side="left", fill="y", pady=10, ipady=30)
        right_frame = Frame(self)  # ,bg='green')
        right_frame.pack(side="right", fill="y", pady=10, ipady=30)

        # Create and pack radio buttons
        self.selected_model = StringVar()
        # Radio Buttons
        model_dl_sl = Radiobutton(
            left_frame,
            text="Model: DL - SL",
            variable=self.selected_model,
            value="dl-sl",
            width=30,
            anchor="center",
            command=self.model_settings_config,
        )
        model_dl_sl.pack(fill="x", pady=10, padx=20, anchor="center")
        model_rf_ssl = Radiobutton(
            right_frame,
            text="Model: RF - SSL",
            variable=self.selected_model,
            value="rf-ssl",
            width=30,
            anchor="center",
            command=self.model_settings_config,
        )
        model_rf_ssl.pack(fill="x", pady=20, padx=20, anchor="center")

        # Configure style for default selection
        model_dl_sl.config(activebackground="#8fbc8f")
        model_rf_ssl.config(activebackground="#8fbc8f")
        # Set default choice
        self.selected_model.set(glv.GLB_Model)

        # Fileds and values
        ForwardIterationLabel = Label(
            left_frame,
            text="Forward Iterations*",
            width=30,
            font=("Sylfaen", 10),
            anchor="w",
        )
        ForwardIterationLabel.pack(fill="x", pady=10, padx=20)
        ##drop-down list
        ForwardIterationOptions = ["100", "500", "1000"]

        ForwardIterationVariable = StringVar(right_frame)
        ForwardIterationVariable.set(str(glv.GLB_FORWARD_ITERATIONS))  # default value
        # Set border
        self.forward_iterations_options = OptionMenu(
            right_frame, ForwardIterationVariable, *ForwardIterationOptions
        )
        self.forward_iterations_options.pack(fill="both", pady=10, padx=20)
        ##drop-down

        DropeoutRateLabel = Label(
            left_frame, text="Dropout Rate*", width=30, font=("Sylfaen", 10), anchor="w"
        )
        DropeoutRateLabel.pack(fill="x", pady=15, padx=20)
        ##drop-down list
        DropeoutRateOptions = ["0.1", "0.2", "0.3"]

        DropeoutRateVariable = StringVar(right_frame)
        DropeoutRateVariable.set(str(glv.GLB_DROPOUT_RATE))  # default value
        # Set border
        self.dropout_rate_options = OptionMenu(
            right_frame, DropeoutRateVariable, *DropeoutRateOptions
        )
        self.dropout_rate_options.pack(fill="x", pady=8, padx=20)

        ## Max number of molecular compounds to process
        MaxCompoundsLabel = Label(
            left_frame,
            text="Maximum Compounds*",
            width=30,
            font=("Sylfaen", 10),
            anchor="w",
        )
        MaxCompoundsLabel.pack(fill="x", pady=15, padx=20)
        ##drop-down list
        MaxCompoundsOptions = ["64", "128", "256", "512"]

        maxCompoundVariable = StringVar(right_frame)
        maxCompoundVariable.set(str(glv.GLB_MAXIMUM_COMPOUNDS))  # default value
        # Set border
        self.compounds_batch_options = OptionMenu(
            right_frame, maxCompoundVariable, *MaxCompoundsOptions
        )
        self.compounds_batch_options.pack(fill="x", pady=8, padx=20)
        self.model_settings_config()

    def save(self, ForwardIteration, DropeoutRate, CompoundsBatch):
        """Save model config"""
        if "" in [ForwardIteration.get(), DropeoutRate.get(), CompoundsBatch.get()]:
            # Pop up message box
            messagebox.showerror(
                "Settings error", "Make sure all mandatory fields are selected!"
            )
        else:
            glv.GLB_Model = str(self.selected_model.get())
            glv.GLB_FORWARD_ITERATIONS = int(ForwardIteration.get())
            glv.GLB_DROPOUT_RATE = float(DropeoutRate.get())
            glv.GLB_MAXIMUM_COMPOUNDS = int(CompoundsBatch.get())
            messagebox.showinfo("Settings", "Selected settings are saved!")

    def model_settings_config(self):
        glv.GLB_Model = self.selected_model.get()
        if self.selected_model.get() == "dl-sl":
            self.forward_iterations_options.config(state=NORMAL)
            self.dropout_rate_options.config(state=NORMAL)
            self.compounds_batch_options.config(state=NORMAL)
        else:
            self.forward_iterations_options.config(state=DISABLED)
            self.dropout_rate_options.config(state=DISABLED)
            self.compounds_batch_options.config(state=DISABLED)
