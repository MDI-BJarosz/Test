# GenAI Travel Survey Helper

A zero-shot Large Language Model (LLM) agent that pre-populates travel survey trip mode and purpose labels to reduce respondent burden in federal travel surveys. Tested on the National Household Travel Survey (NHTS) and the TSDC 2021–2022 Can Do Colorado E-Bike Pilot Program Study using Claude 3.7 Sonnet via AWS Bedrock.

This project was conducted as part of the **Informing Next-generation Federal Statistics (INFS)** project in support of the **National Secure Data Service Demonstration (NSDS-D)**.

---

## Overview

Traditional travel surveys ask respondents to manually label every detected trip with a transportation mode and trip purpose — a task that drives fatigue and attrition in longitudinal studies. This project evaluates whether a generative AI agent can pre-populate those labels from observable trip attributes, respondent demographics, and prior trip history, shifting respondents from manual data entry to a simpler confirmation and correction task.

The agent operates **zero-shot** — no fine-tuning or training data is required. All reasoning occurs in-context at inference time using Claude 3.7 Sonnet on AWS Bedrock.

---

## Key Results

| Dataset | Mode Top-1 | Mode Top-3 | Purpose Top-1 | Purpose Top-3 |
|---|---|---|---|---|
| NHTS | 90.0% | 98.8% | 53.0% | 76.4% |
| TSDC | 26.5% | 67.7% | 23.1% | 46.8% |

---

## Requirements

- Python 3.11+
- AWS account with Bedrock access enabled for LLM models
- NHTS and/or TSDC data (links below)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data

This repository does not include raw data files. Download from the following public sources:

| Dataset | Source |
|---|---|
| National Household Travel Survey (NHTS) | https://nhts.ornl.gov |
| TSDC Can Do Colorado E-Bike Pilot | https://www.nrel.gov/tsdc |
 Example data is in the inputs/
---

## Usage

### 1. Configure the agent

Copy the example config and fill in your AWS credentials:

Edit `config.yaml`:

```yaml
# Option 1 (Recommended): Use an AWS profile
AWS_PROFILE: "default"

# Option 2: Direct credentials
# AWS_ACCESS_KEY_ID: "your_key"
# AWS_SECRET_ACCESS_KEY: "your_secret"

AWS_REGION: "us-east-1"
BEDROCK_MODEL_ID: "anthropic.claude-3-7-sonnet..."

DATASET_TYPE: "nhts"    # "nhts" or "tdsc"
```

Make sure you have requested Claude model access in the AWS Bedrock Console:
`AWS Bedrock Console → Model access → Manage model access → Enable Claude models`


---

### 2. Run the accuracy evaluation

Open `surveyHelper/tst/test_accuracy.py` and set the configuration at the top of the file:

```python
DATASET_NAME = "nhts"    # "nhts" or "tdsc"
SAMPLE_SIZE = 200        # Number of trips to evaluate
RANDOM_SEED = 42         # Fixed seed for reproducibility
TEST_MODE = False        # Set True for a quick 5-trip test run
```

Then run:

```bash
cd surveyHelper
python -m tst.test_accuracy
```

The agent processes each trip sequentially, printing predictions and running accuracy as it goes. Results are written to CSV incrementally so progress is not lost if the run is interrupted.

---

### 3. Generate metrics

Once the evaluation is complete, run:

```bash
python generate_metrics.py
```

This reads the results CSV, removes any trips with failed predictions, and writes a full metrics report to `tst/results/accuracy_metrics_{dataset}.csv`.

---

### 4. Output files

| File | Description |
|---|---|
| `tst/results/accuracy_results_nhts.csv` | Raw Top-3 predictions for all evaluated NHTS trips |
| `tst/results/accuracy_results_tdsc.csv` | Raw Top-3 predictions for all evaluated TSDC trips |
| `tst/results/accuracy_metrics_nhts.csv` | Aggregated metrics by category and Top-K level (NHTS) |
| `tst/results/accuracy_metrics_tdsc.csv` | Aggregated metrics by category and Top-K level (TSDC) |

Metrics files include:
- Overall Top-1, Top-2, and Top-3 accuracy and weighted F1-score
- Per-category precision, recall, F1-score, TP, FP, TN, FN for every mode and purpose label


---

## How It Works

For each trip, the agent receives a structured prompt containing:

1. **Current trip attributes** — distance, duration, speed, time of day, day of week
2. **Respondent demographics** — age, income, vehicle access, employment status
3. **Prior trip history** — mode and purpose of preceding trips in chronological order

The agent returns **three ranked predictions** per trip, each with a confidence level:
- **Rank 1 (high confidence)** — primary prediction
- **Rank 2 (medium confidence)** — must use a different mode than Rank 1
- **Rank 3 (low confidence)** — must differ from both Rank 1 and Rank 2

Ground truth labels are withheld from the current trip's attributes to prevent data leakage.

---

## Switching Datasets

To switch between NHTS and TSDC, update two settings:

1. In `configs/config.yaml`: set `DATASET_TYPE: "nhts"` or `DATASET_TYPE: "tdsc"`
2. In `tst/test_accuracy.py`: set `DATASET_NAME = "nhts"` or `DATASET_NAME = "tdsc"`

The agent automatically loads the appropriate adapter and dataset-specific category options.

---

## Citation

If you use this code in your research, please cite:

```
Add the paper/project here
```

---

## Funding

The National Secure Data Service (NSDS), a flagship initiative of the federal statistical system,
is envisioned to transform the government’s capacity to use the data it already collects, providing
a secure, scalable service—using AI, innovative tools, and powerful privacy protections—to
further connect people with trusted data and technical solutions to make smarter decisions and to
solve real-world problems. The 5-year NSDS demonstration project was authorized under
section 10375 of the CHIPS and Science Act (136 Stat. 1574). The NSDS demonstration project
informs the viability and scalability of a future NSDS. The NSDS will be a new addition to the
federal statistical system’s suite of shared services, joining the Federal Statistical Research Data
Centers (FSRDCs) and the Standard Application Process (SAP), among others, in meeting
common data needs of statistical agencies and their users.

This project supports NSDS by presenting a roadmap for supporting the data infrastructure of the
federal data ecosystem to improve interoperability and efficiency of data resources used to
generate novel frames and estimates.

Multiple key issues, such as governance, future funding, and scalability, are under discussion
with government leaders.

This project was conducted under Government contract with America’s DataHub Consortium
(ADC), a public-private partnership that supports the strategic objectives of the National Center
for Science and Engineering Statistics within the U.S. National Science Foundation. Strategic
leadership for the development of the NSDS is provided by the Interagency Council on
Statistical Policy’s (ICSP) Subcommittee on the NSDS (S-NSDS).

---

## License

MIT License. See `LICENSE` for details.
