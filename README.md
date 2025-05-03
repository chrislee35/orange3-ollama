# orange3-ollama

This provides a collection of widgets that leverage Ollama to perform various data mining tasks.

## Installation

Within the Add-ons installer, click on "Add more..." and type in orange3-ollama

## Widgets

![All 5 Ollama-based widgets on the canvas](https://github.com/chrislee35/orange3-ollama/blob/main/imgs/ollama-all.png?raw=true)

### Ollama Text Classifier
* Inputs: Corpus
* Outputs: Annotated corpus with predicted categories

![Example Orange3 workflow using the Ollama Text Classification widget to determine if a line from the show Friends is male, female, or neutral.](https://github.com/chrislee35/orange3-ollama/blob/main/imgs/ollama-classification.png?raw=true)


### Ollama Sentiment Analyzer
* Inputs: Corpus
* Outputs: Annotated corpus with sentiment labels

![Example Orange3 workflow using the Ollama Sentiment Analysis widget to determine the sentiment of a line from the show Friends as positive, negative, or neutral.](https://github.com/chrislee35/orange3-ollama/blob/main/imgs/ollama-sentiment.png?raw=true)

### Ollama Summarizer
* Inputs: Corpus
* Outputs: Annotated corpus with summary column

![Example Orange3 workflow using the Ollama Summarizer widget to generate a short summary of stories from the Andersen text corpus.](https://github.com/chrislee35/orange3-ollama/blob/main/imgs/ollama-summarizer.png?raw=true)

### Ollama Analysis Suggester
* Inputs: Table
* Outputs: None, it displays it's suggestion within the widget.

![Example Orange3 workflow using the Ollama Analysis Suggester widget to generate suggestions for types of analysis to apply to the Titantic dataset.](https://github.com/chrislee35/orange3-ollama/blob/main/imgs/ollama-analysis-suggester.png?raw=true)


### Python Script Generator
* Inputs: Table
* Outputs: Table

![Example Orange3 workflow using the Python Script Generator widget to generate a python script based on a prompt to skip every other row of input and to add 2 to every numerical colum.  This is applied to the Iris dataset.](https://github.com/chrislee35/orange3-ollama/blob/main/imgs/ollama-script-generator.png?raw=true)

