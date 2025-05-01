from AnyQt.QtWidgets import QTextEdit, QPushButton, QComboBox, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QWidget
from AnyQt.QtCore import QMetaObject, Qt, Q_ARG, QObject, pyqtSignal
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from Orange.data import Table, Domain, StringVariable
from Orange.data.pandas_compat import table_from_frame, table_to_frame
import requests
import json
import threading
import pandas as pd

class StreamHandler(QObject):
    new_text = pyqtSignal(str)
    error = pyqtSignal(str)

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
        self.layout_control_area()
        self.layout_main_area()

    def layout_control_area(self):
        self.control_widget = QWidget()
        layout = QGridLayout()
        layout.setVerticalSpacing(2)

        layout.addWidget(QLabel("Ollama Host:"), 0, 0)
        self.host_input = QLineEdit(self.ollama_host)
        layout.addWidget(self.host_input, 0, 1)

        layout.addWidget(QLabel("Ollama Port:"), 1, 0)
        self.port_input = QLineEdit(self.ollama_port)
        layout.addWidget(self.port_input, 1, 1)

        self.model_selector = QComboBox()
        layout.addWidget(QLabel("Active Model:"), 2, 0)
        layout.addWidget(self.model_selector, 2, 1, 1, 2)

        self.summarize_button = QPushButton("Summarize Documents")
        self.summarize_button.clicked.connect(self.summarize_documents)
        layout.addWidget(self.summarize_button, 3, 0, 1, 3)

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

    def summarize_documents(self):
        if self.in_corpus is None:
            return

        df = table_to_frame(self.in_corpus, include_metas=True)
        if self.selected_column not in df.columns:
            return

        summaries = []

        host = self.host_input.text()
        port = self.port_input.text()
        model = self.model_selector.currentText()
        context = self.prompt_context

        for i, row in df.iterrows():
            text = row[self.selected_column]
            prompt = f"{context}\n\nPlease provide a very short summary of the following document:\n\n{text}"

            try:
                response = requests.post(
                    f"http://{host}:{port}/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"model": model, "prompt": prompt})
                )
                if response.status_code == 200:
                    lines = response.text.strip().splitlines()
                    results = [json.loads(l)['response'] for l in lines if 'response' in json.loads(l)]
                    summaries.append("".join(results).strip())
                else:
                    summaries.append("[Error in response]")
            except Exception as e:
                summaries.append(f"[Error: {e}]")
            break

        df['Summary'] = summaries
        # summary_var = StringVariable("Summary")
        # attrs = list(self.in_corpus.domain.attributes)
        # class_vars = list(self.in_corpus.domain.class_vars)
        # metas = list(self.in_corpus.domain.metas) + [summary_var]
        # domain = Domain(attrs, class_vars, metas)
        #summary_col = pd.Series(summaries, name='Summary')
        summary_table = table_from_frame(df)

        self.Outputs.out_table.send(summary_table)

    @Inputs.in_corpus
    def set_input(self, data):
        self.in_corpus = data
        if data is not None:
            df = table_to_frame(data, include_metas=True)
            self.feature_selector.clear()
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                    self.feature_selector.addItem(col)
            if self.selected_column:
                idx = self.feature_selector.findText(self.selected_column)
                if idx >= 0:
                    self.feature_selector.setCurrentIndex(idx)

if __name__ == "__main__":
    from orangecontrib.text.corpus import Corpus
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWOllamaSummarizer).run(Corpus("andersen"))
