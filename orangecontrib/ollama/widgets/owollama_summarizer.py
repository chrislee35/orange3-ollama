from AnyQt.QtWidgets import QTextEdit, QPushButton, QComboBox, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QWidget, QProgressBar
from AnyQt.QtCore import QMetaObject, Qt, Q_ARG, QObject, pyqtSignal, QThread
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, Domain, StringVariable
from Orange.data.pandas_compat import table_from_frame, table_to_frame
import requests
import json
import threading
import pandas as pd

class SummarizerWorker(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    summary_error = pyqtSignal(str)

    def __init__(self, corpus, column, host, port, model, context):
        super().__init__()
        self.corpus = corpus
        self.column = column
        self.host = host
        self.port = port
        self.model = model
        self.context = context
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        summaries = []
        total = len(self.corpus)

        for i, text in enumerate(self.corpus.get_column(self.column)):
            if self._cancel:
                return
            prompt = f"{self.context}\n\nPlease provide a very short summary of the following document:\n\n{text}"

            try:
                response = requests.post(
                    f"http://{self.host}:{self.port}/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"model": self.model, "prompt": prompt})
                )
                if response.status_code == 200:
                    lines = response.text.strip().splitlines()
                    results = [json.loads(l)['response'] for l in lines if 'response' in json.loads(l)]
                    summaries.append("".join(results).strip())
                else:
                    summaries.append("[Error in response]")
            except Exception as e:
                summaries.append(f"[Error: {e}]")

            progress = int((i + 1) / total * 100)
            self.progress_updated.emit(progress)

        self.corpus = self.corpus.add_column(StringVariable(name="Summary"), summaries, to_metas=True)
        self.result_ready.emit(self.corpus)

class OWOllamaSummarizer(OWWidget):
    name = "Ollama Summarizer"
    description = "Summarizes documents using models hosted in Ollama."
    icon = "icons/ollama-seeklogo.svg"
    priority = 101

    class Inputs:
        in_corpus = Input("Corpus", Table)

    class Outputs:
        out_table = Output("Summarized Corpus", Table)

    ollama_host = Setting("localhost")
    ollama_port = Setting("11434")
    selected_model = Setting("")
    prompt_context = Setting("")
    selected_column = Setting("")

    def __init__(self):
        super().__init__()
        self.in_corpus = None
        self.worker = None
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
        layout.addWidget(self.model_selector, 2, 1, 1, 2)

        self.summarize_button = QPushButton("Summarize Documents")
        self.summarize_button.clicked.connect(self.run_summary_thread)
        layout.addWidget(self.summarize_button, 3, 0, 1, 2)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_summary)
        layout.addWidget(self.cancel_button, 3, 2)

        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)
        control_layout.addLayout(layout)

        self.control_widget.setLayout(control_layout)
        self.controlArea.layout().addWidget(self.control_widget)
        self.update_model_list()

    def layout_main_area(self):
        self.context_box = QTextEdit(self.prompt_context)
        self.context_box.setPlaceholderText("Add context for summarization prompt...")
        self.context_box.textChanged.connect(lambda: setattr(self, 'prompt_context', self.context_box.toPlainText()))
        self.mainArea.layout().addWidget(QLabel("Prompt Context:"))
        self.mainArea.layout().addWidget(self.context_box)

        self.feature_selector = QComboBox()
        self.feature_selector.currentTextChanged.connect(lambda text: setattr(self, 'selected_column', text))
        self.mainArea.layout().addWidget(QLabel("Text Feature to Summarize:"))
        self.mainArea.layout().addWidget(self.feature_selector)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.mainArea.layout().addWidget(QLabel("Progress:"))
        self.mainArea.layout().addWidget(self.progress_bar)

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

    def run_summary_thread(self):
        if self.in_corpus is None:
            return

        metas = self.text_columns
        if self.selected_column not in metas:
            return

        host = self.host_input.text()
        port = self.port_input.text()
        model = self.model_selector.currentText()
        context = self.prompt_context

        self.worker = SummarizerWorker(self.in_corpus.copy(), self.selected_column, host, port, model, context)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.result_ready.connect(self.handle_result)
        self.worker.start()
        self.cancel_button.setEnabled(True)

    def cancel_summary(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        self.cancel_button.setEnabled(False)

    def handle_result(self, summary_table):
        self.Outputs.out_table.send(summary_table)
        self.progress_bar.setValue(100)
        self.cancel_button.setEnabled(False)

    @Inputs.in_corpus
    def set_input(self, data):
        self.in_corpus = data
        self.text_columns = []
        if data is not None:
            self.feature_selector.clear()
            for col in data.domain.metas:
                if isinstance(col, StringVariable):
                    self.feature_selector.addItem(col.name)
                    self.text_columns.append(col.name)
            if self.selected_column:
                idx = self.feature_selector.findText(self.selected_column)
                if idx >= 0:
                    self.feature_selector.setCurrentIndex(idx)

if __name__ == "__main__":
    from orangecontrib.text.corpus import Corpus
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWOllamaSummarizer).run(Corpus("andersen"))
