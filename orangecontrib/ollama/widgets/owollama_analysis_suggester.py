from AnyQt.QtWidgets import QTextEdit, QPushButton, QComboBox, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QWidget, QProgressBar
from AnyQt.QtCore import QMetaObject, Qt, Q_ARG, QObject, pyqtSignal, QThread
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input
from Orange.data import Table
from Orange.data.pandas_compat import table_from_frame, table_to_frame
import requests
import json
import threading
import pandas as pd

class SuggestionWorker(QThread):
    progress_updated = pyqtSignal(int)
    suggestion_ready = pyqtSignal(str)
    suggestion_error = pyqtSignal(str)

    def __init__(self, table, host, port, model, analysis_type):
        super().__init__()
        self.table = table
        self.host = host
        self.port = port
        self.model = model
        self.analysis_type = analysis_type
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def describe_table(self):
        try:
            df = table_to_frame(self.table, include_metas=True)
            sample = df.sample(min(len(df), 20))
            description = sample.describe(include='all').to_string()
            return f"Table Columns: {list(df.columns)}\nSample Summary:\n{description}"
        except Exception as e:
            return f"Error describing table: {e}"

    def run(self):
        description = self.describe_table()
        prompt = f"{self.analysis_type}\n\n{description}"
        suggestion_text = ""

        try:
            response = requests.post(
                f"http://{self.host}:{self.port}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps({"model": self.model, "prompt": prompt})
            )
            if response.status_code == 200:
                lines = response.text.strip().splitlines()
                results = [json.loads(l)['response'] for l in lines if 'response' in json.loads(l)]
                suggestion_text = "".join(results).strip()
            else:
                suggestion_text = f"[Error in response: {response.status_code}]"
        except Exception as e:
            suggestion_text = f"[Error: {e}]"

        self.suggestion_ready.emit(suggestion_text)

class OWOllamaAnalysisSuggester(OWWidget):
    name = "Ollama Analysis Suggester"
    description = "Suggests types of analysis and preprocessing using Ollama."
    icon = "icons/ollama-seeklogo.svg"
    priority = 104

    class Inputs:
        in_table = Input("Table", Table)

    ollama_host = Setting("localhost")
    ollama_port = Setting("11434")
    selected_model = Setting("")
    selected_analysis_type = Setting("")

    def __init__(self):
        super().__init__()
        self.in_table = None
        self.worker = None
        self.suggested_text = ""
        self.layout_control_area()
        self.layout_main_area()

    def layout_control_area(self):
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
        layout.addWidget(self.model_selector, 2, 1)

        self.run_button = QPushButton("Perform Suggestion")
        self.run_button.clicked.connect(self.run_suggestion_thread)
        layout.addWidget(self.run_button, 3, 0, 1, 2)

        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.addLayout(layout)
        self.control_widget.setLayout(control_layout)
        self.controlArea.layout().addWidget(self.control_widget)
        self.update_model_list()

    def layout_main_area(self):
        layout = QVBoxLayout()

        self.analysis_selector = QComboBox()
        self.analysis_selector.addItems([
            "Suggest types of analysis for the following table.",
            "Suggest additional data sources to combine with this table.",
            "Suggest steps to preprocess and improve the data in this table."
        ])
        self.analysis_selector.currentTextChanged.connect(lambda text: setattr(self, 'selected_analysis_type', text))
        layout.addWidget(QLabel("Type of Analysis:"))
        layout.addWidget(self.analysis_selector)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_bar)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(QLabel("Suggested Analysis:"))
        layout.addWidget(self.result_text)

        container = QWidget()
        container.setLayout(layout)
        self.mainArea.layout().addWidget(container)

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
            print("Failed to fetch models from Ollama server:", e)

    def run_suggestion_thread(self):
        if self.in_table is None:
            return

        host = self.host_input.text()
        port = self.port_input.text()
        model = self.model_selector.currentText()
        analysis_type = self.analysis_selector.currentText()

        self.result_text.clear()
        self.progress_bar.setValue(0)

        self.worker = SuggestionWorker(self.in_table, host, port, model, analysis_type)
        self.worker.suggestion_ready.connect(self.handle_suggestion)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.start()

    def handle_suggestion(self, result):
        self.result_text.setPlainText(result)
        self.progress_bar.setValue(100)

    @Inputs.in_table
    def set_input(self, data):
        self.in_table = data

if __name__ == "__main__":
    from Orange.data import Table
    from orangewidget.utils.widgetpreview import WidgetPreview

    data = Table("iris")
    WidgetPreview(OWOllamaAnalysisSuggester).run(data)
