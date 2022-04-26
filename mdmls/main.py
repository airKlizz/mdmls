import typer

import evaluation.cli as evaluation
import prepocessing.cli as preprocessing
import training.cli as training
import use.cli as use

app = typer.Typer()
app.add_typer(evaluation.app, name="evaluate")
app.add_typer(preprocessing.app, name="preprocess")
app.add_typer(training.app, name="train")
app.add_typer(use.app, name="use")

if __name__ == "__main__":
    app()