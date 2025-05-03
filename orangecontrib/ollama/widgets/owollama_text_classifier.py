from AnyQt.QtWidgets import QTextEdit, QPushButton, QComboBox, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QWidget, QProgressBar, QListWidget, QHBoxLayout
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

class ClassificationWorker(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    summary_error = pyqtSignal(str)

    def __init__(self, corpus, column, host, port, model, labels):
        super().__init__()
        self.corpus = corpus
        self.column = column
        self.host = host
        self.port = port
        self.model = model
        self.labels = labels
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        predictions = []
        total = len(self.corpus)
        label_str = ', '.join(self.labels)
        use_generic_prompt = not self.labels

        for i, text in enumerate(self.corpus.get_column(self.column)):
            if self._cancel:
                return

            if use_generic_prompt:
                prompt = f"Classify the topic of the following text in a simple, one-word label.\n\n{text}"
            else:
                prompt = f"Given the following text, classify it as one of the following: {label_str}. Respond with only one word from the list. Do not add anything else.\n\n{text}"

            try:
                response = requests.post(
                    f"http://{self.host}:{self.port}/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"model": self.model, "prompt": prompt})
                )
                if response.status_code == 200:
                    lines = response.text.strip().splitlines()
                    results = [json.loads(l)['response'] for l in lines if 'response' in json.loads(l)]
                    result = "".join(results).strip()
                    predictions.append(result)
                else:
                    predictions.append("[Error in response]")
            except Exception as e:
                predictions.append(f"[Error: {e}]")

            progress = int((i + 1) / total * 100)
            self.progress_updated.emit(progress)

        self.corpus = self.corpus.add_column(StringVariable(name="Prediction"), predictions, to_metas=True)
        self.result_ready.emit(self.corpus)

class OWOllamaTextClassifier(OWWidget):
    name = "Ollama Text Classification"
    description = "Classifies text using models hosted in Ollama."
    icon = "icons/ollama-seeklogo.svg"
    priority = 103

    class Inputs:
        in_corpus = Input("Corpus", Table)

    class Outputs:
        out_table = Output("Labeled Corpus", Table)

    ollama_host = Setting("localhost")
    ollama_port = Setting("11434")
    selected_model = Setting("")
    selected_column = Setting("")
    classification_labels = Setting([])

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

        self.run_button = QPushButton("Classify Text")
        self.run_button.clicked.connect(self.run_classification_thread)
        layout.addWidget(self.run_button, 3, 0, 1, 2)

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
        container = QWidget()
        layout = QVBoxLayout(container)

        self.feature_selector = QComboBox()
        self.feature_selector.currentTextChanged.connect(lambda text: setattr(self, 'selected_column', text))
        layout.addWidget(QLabel("Text Feature to Classify:"))
        layout.addWidget(self.feature_selector)

        label_layout = QHBoxLayout()
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Add classification label")
        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_label)
        label_layout.addWidget(self.label_input)
        label_layout.addWidget(add_button)
        layout.addLayout(label_layout)

        self.label_list = QListWidget()
        self.label_list.addItems(self.classification_labels)
        layout.addWidget(self.label_list)

        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected_label)
        layout.addWidget(remove_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.progress_bar)

        self.mainArea.layout().addWidget(container)


    def add_label(self):
        label = self.label_input.text().strip()
        if label and label not in self.classification_labels:
            self.classification_labels.append(label)
            self.label_list.addItem(label)
            self.label_input.clear()

    def remove_selected_label(self):
        selected_items = self.label_list.selectedItems()
        for item in selected_items:
            self.classification_labels.remove(item.text())
            self.label_list.takeItem(self.label_list.row(item))

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

    def run_classification_thread(self):
        if self.in_corpus is None or not self.classification_labels:
            return

        metas = self.text_columns
        if self.selected_column not in metas:
            return

        host = self.host_input.text()
        port = self.port_input.text()
        model = self.model_selector.currentText()

        self.worker = ClassificationWorker(self.in_corpus.copy(), self.selected_column, host, port, model, self.classification_labels)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.result_ready.connect(self.handle_result)
        self.worker.start()
        self.cancel_button.setEnabled(True)

    def cancel_summary(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        self.cancel_button.setEnabled(False)

    def handle_result(self, result_table):
        self.Outputs.out_table.send(result_table)
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
    import random

    full_corpus = Corpus("election-tweets-2016")
    indices = random.sample(range(len(full_corpus)), 10)
    sample_corpus = full_corpus[indices]

    WidgetPreview(OWOllamaTextClassifier).run(sample_corpus)
