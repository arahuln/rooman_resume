# Resume Screening Pipeline for Rooman AI Role

This pipeline screens PDF resumes against the Rooman AI Product Engineer job description using NVIDIA's meta/llama-3.1-70b-instruct model.

## Setup

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set your NVIDIA API key:**
   ```bash
   export NVIDIA_API_KEY="your_api_key_here"
   ```

3. **Add PDF resumes:**
   - Place all PDF resume files in the `resumes/` folder

## Usage

**Basic screening:**
```bash
python screen_resumes.py --resumes ./resumes --jd ./jd.json --topk 25 --out ranked.csv
```

**Run demo:**
```bash
python screen_resumes.py --demo
```

## Output

The pipeline generates a `ranked.csv` file with:
- **score**: Overall match score (0-100)
- **passes**: Boolean indicating if candidate meets minimum requirements
- **reasons**: Detailed reasoning for the score
- **must**: Must-have skills found
- **nice**: Nice-to-have skills found
- **exp_years**: Years of experience detected
- **edu_match**: Education requirements met
- **red_flags**: Any concerning issues
- **summary**: Brief candidate summary

## Job Description

The `jd.json` is configured for Rooman's AI Product Engineer role:
- **Must-have**: Python (only hard requirement)
- **Nice-to-have**: PyTorch, TensorFlow, LLMs, Vector DBs, Docker, AWS/GCP, React, etc.
- **Experience**: 0+ years (freshers welcome)
- **Education**: B.Tech/B.E./M.Tech/M.S. preferred
- **Institutions**: IIT/NIT/IIIT preference (bonus, not required)

## Algorithm

1. **PDF Extraction**: Reads and normalizes text from PDF files
2. **TF-IDF Prefilter**: Uses cosine similarity to rank resumes by keyword match
3. **LLM Scoring**: Calls NVIDIA API for detailed scoring and reasoning
4. **Ranking**: Sorts by pass/fail status, then score, then pre-similarity

## Performance

- **Complexity**: O(N·D + K) where N=resumes, D=avg doc length, K=top candidates
- **Memory**: ~O(N·D) for TF-IDF vectors
- **API Calls**: Only for top K candidates (default 25)
