import typer

import evaluation.cli as evaluation
import prepocessing.cli as preprocessing
import training.cli as training

app = typer.Typer()
app.add_typer(evaluation.app, name="evaluate")
app.add_typer(preprocessing.app, name="preprocess")
app.add_typer(training.app, name="train")

if __name__ == "__main__":
    app()