# End to end Text-Summarizer-Project

A production-grade NLP pipeline that uses **Google's Pegasus** model fine-tuned on the **SAMSum** dialogue dataset to generate concise summaries of conversations.

## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. update the conponents
6. update the pipeline
7. update the main.py
8. update the app.py


# How to run locally?
### STEPS:

Clone the repository:

```bash
git clone https://github.com/Ishitv-gigabyte/Text-Summarization-NLP.git
cd Text-Summarization-NLP
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n summary python=3.8 -y
```

```bash
conda activate summary
```


### STEP 02- Install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Run the locally hosted Gradio user interface
```bash
python app_gradio.py
```

Now, open your web browser and navigate to:
```text
http://localhost:8080
```

---

**Author:** Ishitv Sharma  
**GitHub:** [Ishitv-gigabyte](https://github.com/Ishitv-gigabyte)

