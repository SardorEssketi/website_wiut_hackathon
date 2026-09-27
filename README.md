# Alert intelligence

Our WIUT hackathon report explores historical transaction behavior and escalation prediction. The Streamlit app presents aggregate data, feature analysis, model comparisons, and validation results.

## Run locally

Python 3.12:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy

On Streamlit Community Cloud, select this repository, branch `main`, entry point `app.py`, and Python 3.12. The app includes its dependencies and theme configuration.

All report tables are in `data/`. The website runs independently of the training pipeline and requires no model files or credentials.
