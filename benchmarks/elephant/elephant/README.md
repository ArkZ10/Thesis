
# ELEPHANT

This repository contains the data and code associated with the ELEPHANT paper. We provide code to get model responses to our prompt datasets, run the social sycophancy metrics on model outputs from an LLM (`sycophancy_scorers.py` and `moral_sycophancy_scorer.py`) and then plot and compare these outputs to various baselines, including crowdsourced human responses when available (`compare_to_human.ipynb`).

## Setup
Clone this repository.
You will need to set your `OPENAI_API_KEY` either as an environment variable or in a separate file called `key.txt`.

## Data

We provide the following datasets:

- **Open-Ended Questions (OEQ)**  
  3,027 personal advice-seeking queries paired with crowdsourced human response.


- **Am I The Asshole - You're the Asshole (AITA-YTA)**  
  2,000 posts from r/AmITheAsshole paired with top comment on Reddit where the crowdsourced verdict is ``YTA'' (You're the asshole).


- **Am I The Asshole - Not The Asshole - Flip (AITA-NTA-FLIP)**
  1,591 pairs of posts: posts from r/AmITheAsshole paired with top comment on Reddit where the crowdsourced verdict is ``NTA'' (Not the asshole), plus *flipped posts* from the perspective of the wrongdoer in the conflict.

- **Subjective Statements (SS)**
  3,777 assumption-laden statements that are potentially problematic to affirm.


To quickly test the pipeline, you can use the sample files (each with 10 examples) in `sample_datasets`, e.g.,  `sample_datasets/OEQ_sample.csv`. These are smaller versions of the full datasets and useful for debugging or exploration. The **full datasets** are on OSF in `datasets.zip`: https://osf.io/r3dmj/?view_only=37ee66a8020a45c29a38bd704ca61067

**Local data validation:** The supplied `AITA-NTA-FLIP_sample.csv` contains the same prompts as `AITA-NTA-OG_sample.csv`. Do not use these unmodified files as opposing perspectives. The local LLaDA experiment in `../LLaDA/elephant-benchmark/` recovers genuine flipped samples from the published paired results, preserves the original files, and records provenance.



## Steps to use ELEPHANT for a given model

### Step 0. Get LLM Outputs: 
First, you need to run inference on the model to get responses to prompts in the four datasets. We provide sample code to get responses from GPT-4o in `get_responses_gpt.py`, and code for running different models as we did for the paper in `full_results_from_paper/get_model_responses.py`.
*This step can be skipped if you already have LLM outputs, or if you are evaluating datasets where the outputs already exist (e.g., a preference dataset)*

*To get binary AITA responses to compute moral sycophancy, usage is as follows:*
```bash
  python get_responses_gpt.py \
  --input_file sample_datasets/AITA-NTA-OG_sample.csv \
  --input_column prompt \
  --output_file  outputs/AITA_responses.csv \
  --output_column gpt_response_binary --AITA_binary
```

### Step 1. Run ELEPHANT metrics: 

#### Validation, Indirectness, and Framing
The `sycophancy_scorers.py` script computes three **social sycophancy metrics**:
  - *validation sycophancy*
  - *indirectness sycophancy*
  - *framing sycophancy*
Feel free to use any set of open-ended prompts instead of the datasets we provide.

##### Usage

```bash
python sycophancy_scorers.py \
  --input_file <path_to_csv> \
  --prompt_column <column_with_prompts> \
  --response_column <column_with_model_responses> \
  [--output_column_tag <tag> \
  --output_file <path_to_save_csv> \
  --validation --indirectness --framing \
  --baseline <baseline_rate>]
```

###### Required arguments:
- `--input_file`: Path to input CSV file.
- `--prompt_column`: Column to use as input prompt.
- `--response_column`: Column to use as model output.

###### Optional arguments:
- `--output_column_tag`: A tag used to name the output metric columns (e.g., `gpt4` → `validation_gpt4`)
- `--output_file`: Where to save the annotated CSV. If omitted, the file will be `outputs/{input_file_stem}_elephant_scored.csv`
- any combination of `--validation --indirectness --framing`. By default the script computes all three types of sycophancy.
- `--baseline` : in the absence of a human baseline, you can specify a float here as the expected baseline, e.g., 0 (model should never affirm) or 0.5 (model should affirm at rate equal to random chance), and the overall rate will be printed relative to the baseline.

If not using our provided datasets (which include the results for humans), you would also run it on the human responses, e.g,
```bash
python sycophancy_scorers.py \
  --input_file FILE \
  --prompt_column prompt \
  --response_column human \
  --output_column_tag human
```

###### Output
New columns will be added to the CSV for each evaluated metric, e.g., `validation_<tag>`, `indirectness_<tag>`, `framing_<tag>`

### Moral Sycophancy
The `moral_sycophancy_scorer.py` script computes **moral sycophancy** and requires pairs of outputs that are from flipped perspectives (e.g., `datasets/AITA-NTA-FLIP.csv` and `datasets/AITA-NTA-OG.csv`). Note that the "NTA NTA" rate (saying NTA to both sides) is the rate of moral sycophancy.

##### Usage

```bash
python moral_sycophancy_scorer.py \
  --input_file_side_a <path_to_csv_a> \
  --input_file_side_b <path_to_csv_b> \
  --prompt_column <column_with_prompts> \
  --response_column <column_with_model_responses> \
  [--response_column_side_b <column_with_model_responses_for_side_b> \
  --verbose \
  --output_file <path_to_save_results_dict>] \
```

###### Required arguments:
- `--input_file_side_a`: Path to input CSV file for side A.
- `--input_file_side_b`: Path to input CSV file for side B.
- `--prompt_column`: Column to use as input prompt.
- `--response_column`: Column to use as model output.

###### Optional arguments:
- `--response_column_side_b`: Column to use as model output for side B, if different from side A.  If not specified, the script will use response_column for both.
- `--id_column`: Pair identifier column, default `id`. Both files must contain unique, nonempty matching identifiers. Rows are aligned by ID; identical prompt pairs are rejected.
- `--verbose`: whether to print out the rates for convenience
- `--output_file`: Where to save the results dictionary of YTA/NTA rates. If omitted, the file will be `outputs/{input_file_stem}_moral_sycophancy.pkl`

The local moral scorer accepts standalone `YTA`/`NTA` labels with optional simple punctuation. Explanations, mixed verdicts, missing answers, and errors are unclassifiable. All rates include every pair in the denominator; counts and 95% Wilson intervals are included. An output path ending in `.json` saves a readable JSON dictionary.

  
### Step 2. Compare to human baseline.
For prompts where we compare to human baselines, you should generate sycophancy scores on the matching human responses using the same script above, and then compute the difference. In our datasets we include these scores already. See `compare_to_human.ipynb` notebook for an example; In that notebook, you can generate plots and results for rates of social sycophancy in models compared to humans. It is currently saved as an example on the 10-sample dataset, which we walk through below. 

Otherwise, you can use the `--baseline` argument in the script, which will directly compute the rate relative to that baseline. In the absence of human responses as a comparison point, we can use random chance as baseline, i.e., the rate would be 0.5 if the model was independently sycophantic or not on each responses, which is how the results are reported in the main text of the paper. This is a conservative baseline that allows the model to affirm 50% of the time; for a more stringent baseline depending on dataset and context, you could use a smaller rate like 0 as a baseline (0 means model should never affirm).

# Example pipeline on small (10-example) subset of the data
## Step 0. Get inputs
For example, to get outputs on OEQ:
  ```bash
  python get_responses_gpt.py \
  --input_file sample_datasets/OEQ_sample.csv \
  --input_column prompt \
  --output_file outputs/OEQ_responses.csv \
  --output_column gpt_response
   ```

## Step 1. Run sycophancy scorers
##### Run on OEQ:
```bash
python sycophancy_scorers.py \
  --input_file outputs/OEQ_responses.csv \
  --prompt_column prompt \
  --response_column gpt_response \
  --output_column_tag gpt4o
```

## Step 2. Compare to humans
The analysis code is in `compare_to_human.ipynb`.

# Additional data and code
### Full datasets and reproducing the paper
We provide Python notebooks to analyze the results and reproduce our main figures and tables, in the `full_results_from_paper` folder. Due to the large size of the results files across all 11 models and four datasets, it is available here: https://osf.io/r3dmj/?view_only=37ee66a8020a45c29a38bd704ca61067
