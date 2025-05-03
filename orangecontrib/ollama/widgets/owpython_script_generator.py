from AnyQt.QtWidgets import QTextEdit, QPushButton, QComboBox, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QWidget
from AnyQt.QtCore import QMetaObject, Qt, Q_ARG, QObject, pyqtSignal
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table
from Orange.data.pandas_compat import table_from_frame, table_to_frame
import requests
import json
import threading
import re

class StreamHandler(QObject):
    new_text = pyqtSignal(str)
    error = pyqtSignal(str)

class OWPythonScriptGenerator(OWWidget):
    name = "Python Script Generator with LLM"
    description = "Generates Python scripts using models hosted in Ollama."
    icon = "icons/ollama-seeklogo.svg"
    priority = 100

    class Inputs:
        in_table = Input("Input Data", Table)

    class Outputs:
        out_table = Output("Output Data", Table)

    # Widget settings
    ollama_host = Setting("localhost")
    ollama_port = Setting("11434")
    selected_model = Setting("")
    code_prompt = Setting("add 2 to very numeric column and samples every other row")
    generated_code = Setting("out_df = in_df")
    error_message = Setting("")

    PRE_PROMPT = "generate a python function called, process_dataframe(in_df: Dataframe), that takes a Pandas dataframe and does the following steps, returning the resulting dataframe."
    POST_PROMPT = "return only the code and no markup.  you can include documentation in comments. do not use the __name__ variable."
    PRE_CODE = "from Orange.data.pandas_compat import table_from_frame, table_to_frame\nin_df = table_to_frame(in_table, include_metas=True)"
    POST_CODE = "out_df = process_dataframe(in_df)\nout_table = table_from_frame(out_df)"

    def __init__(self):
        super().__init__()
        self.in_table = None
        self.layout_control_area()
        self.layout_main_area()

    def layout_control_area(self):
        # Layout
        self.control_widget = QWidget()
        layout = QGridLayout()
        layout.setVerticalSpacing(2)

        layout.addWidget(QLabel("Ollama Host:"), 0, 0)
        self.host_input = QLineEdit(self.ollama_host)
        layout.addWidget(self.host_input, 0, 1)
        self.host_input.editingFinished.connect(self.update_model_list)

        layout.addWidget(QLabel("Ollama Port:"), 1, 0)
        self.port_input = QLineEdit(self.ollama_port)
        layout.addWidget(self.port_input, 1, 1)
        self.port_input.editingFinished.connect(self.update_model_list)
        
        self.model_selector = QComboBox()
        layout.addWidget(QLabel("Active Model:"), 2, 0)
        layout.addWidget(self.model_selector, 2, 1, 1, 2)

        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.addLayout(layout)

        self.control_widget.setLayout(control_layout)
        self.controlArea.layout().addWidget(self.control_widget)

    def layout_main_area(self):
        self.coding_widget = QWidget()
        coding_layout = QGridLayout()

        self.query_box = QTextEdit(self.code_prompt)
        coding_layout.addWidget(QLabel("Code Generation Prompt:"), 0, 0, 1, 3)
        coding_layout.addWidget(self.query_box, 1, 0, 1, 3)

        self.send_button = QPushButton("Generate Code")
        self.send_button.clicked.connect(self.generate_code)
        coding_layout.addWidget(self.send_button, 2, 0, 1, 3)

        self.code_box = QTextEdit()
        self.code_box.setPlainText(self.generated_code)
        self.code_box.setMinimumHeight(200)
        coding_layout.addWidget(QLabel("Generated Code (editable):"), 3, 0, 1, 3)
        coding_layout.addWidget(self.code_box, 4, 0, 1, 3)

        self.execute_button = QPushButton("Execute Code")
        self.execute_button.clicked.connect(self.commit)
        coding_layout.addWidget(self.execute_button, 5, 0, 1, 3)

        self.error_box = QTextEdit(self.error_message)
        self.error_box.setReadOnly(True)
        self.error_box.setStyleSheet("color: red;")
        coding_layout.addWidget(QLabel("Execution or Generation Errors:"), 6, 0, 1, 3)
        coding_layout.addWidget(self.error_box, 7, 0, 1, 3)

        self.coding_widget.setLayout(coding_layout)
        self.mainArea.layout().addWidget(self.coding_widget)

        self.update_model_list()

    def update_model_list(self):
        host, port = self.host_input.text(), self.port_input.text()
        try:
            r = requests.get(f"http://{host}:{port}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                self.model_selector.clear()
                for m in models:
                    self.model_selector.addItem(m['name'])
                if self.selected_model:
                    index = self.model_selector.findText(self.selected_model)
                    if index >= 0:
                        self.model_selector.setCurrentIndex(index)
        except Exception as e:
            self.display_error("Failed to fetch models from Ollama server: " + str(e))

    def generate_code(self):
        prompt = self.query_box.toPlainText()
        self.code_prompt = prompt

        prompt = f"{self.PRE_PROMPT}\n{prompt}\n{self.POST_PROMPT}"

        model = self.model_selector.currentText()
        self.selected_model = model
        host = self.host_input.text()
        port = self.port_input.text()
        self.ollama_host = host
        self.ollama_port = port

        self.code_box.clear()
        handler = StreamHandler()
        handler.new_text.connect(self.code_box.insertPlainText)
        handler.error.connect(self.display_error)

        def stream_response():
            try:
                headers = {"Content-Type": "application/json"}
                data = json.dumps({"model": model, "prompt": prompt})
                with requests.post(f"http://{host}:{port}/api/generate", headers=headers, data=data, stream=True, timeout=60) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                msg = json.loads(line.decode("utf-8"))
                                if "response" in msg:
                                    handler.new_text.emit(msg["response"])
                        self.generated_code = self.code_box.toPlainText()
                        handler.error.emit("")
                    else:
                        handler.error.emit(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                handler.error.emit("Code generation error: " + str(e))

        thread = threading.Thread(target=stream_response)
        thread.start()

    def display_error(self, message):
        self.error_message = message
        self.error_box.setPlainText(message)

    def commit(self):
        local_vars = {"in_table": self.in_table, "out_table": None}
        raw_code = self.code_box.toPlainText()

        # Extract code between triple backticks if present
        match = re.findall(r"```(?:python)?\n(.*?)```", raw_code, re.DOTALL)
        code_to_run = match[0] if match else raw_code

        code_to_run = f"{self.PRE_CODE}\n{code_to_run}\n{self.POST_CODE}"

        try:
            exec(code_to_run, {}, local_vars)
            out_table = local_vars.get("out_table", self.in_table)
            self.display_error("")
        except Exception as e:
            self.display_error("Code execution error: " + str(e))
            out_table = self.in_table

        self.Outputs.out_table.send(out_table)

    @Inputs.in_table
    def set_input(self, data):
        self.in_table = data
        self.commit()

if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPythonScriptGenerator).run(Table("iris"))
