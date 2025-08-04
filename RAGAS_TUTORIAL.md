# 🚀 RAGAS Evaluation Tutorial - Beginner Guide

## What is RAGAS? 
RAGAS (Retrieval Augmented Generation Assessment) evaluates how well your RAG system:
- Finds relevant information (Retrieval)
- Generates accurate answers (Generation)

## 📊 Key Metrics Explained

### 1. **Faithfulness** (0-1)
- How truthful is the answer based on retrieved context?
- `1.0` = Perfect truth, `0.0` = Complete hallucination

### 2. **Answer Relevancy** (0-1)  
- How well does the answer address the question?
- `1.0` = Perfect relevance, `0.0` = Off-topic

### 3. **Context Precision** (0-1)
- How relevant are the retrieved documents?
- `1.0` = All contexts relevant, `0.0` = All irrelevant

### 4. **Context Recall** (0-1)
- How well do contexts cover the expected answer?
- `1.0` = Complete coverage, `0.0` = No coverage

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# Already included in your pyproject.toml
uv sync
```

### 2. Set Environment Variables
```bash
# Add to your .env file
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key  # Optional for reranking
```

## 🎯 How to Run Evaluation

### Option 1: Super Simple (Recommended for Beginners)
```bash
python simple_ragas_eval.py
```

### Option 2: Full Featured
```bash
python ragas_evaluation.py
```

## 📋 What Each Script Does

### `simple_ragas_eval.py` - Beginner Friendly
- **5 test questions** based on your data
- **Quick evaluation** (2-3 minutes)
- **Simple output** with clear scores
- **Uses GPT-3.5-turbo** (cheaper)

### `ragas_evaluation.py` - Advanced
- **Auto-generates test questions** from your CSV
- **10+ test samples** with different complexity
- **Detailed analysis** and interpretation
- **Saves results** to CSV files

## 📊 Understanding Your Results

### Good Scores (What to Aim For)
- **Faithfulness**: > 0.8 (answers stick to facts)
- **Answer Relevancy**: > 0.8 (answers the question)
- **Context Precision**: > 0.7 (finds relevant docs)
- **Context Recall**: > 0.7 (comprehensive coverage)

### Example Output
```
📊 Average Scores:
  Faithfulness:      0.857
  Answer Relevancy:  0.823
  Context Precision: 0.751
  Context Recall:    0.789
```

### What This Means
- **Faithfulness 0.857**: 85% truthful (good!)
- **Answer Relevancy 0.823**: 82% relevant (good!)
- **Context Precision 0.751**: 75% relevant contexts (okay)
- **Context Recall 0.789**: 79% coverage (good!)

## 🔧 Improving Your Scores

### Low Faithfulness? 
- Your RAG is hallucinating
- **Fix**: Better prompts, more context

### Low Answer Relevancy?
- Answers don't match questions
- **Fix**: Improve question understanding

### Low Context Precision?
- Retrieving irrelevant documents
- **Fix**: Better search, reranking

### Low Context Recall?
- Missing important information
- **Fix**: Retrieve more documents, better chunks

## 💰 Cost Estimation

### Simple Evaluation (5 questions)
- **~$0.10-0.20** using GPT-3.5-turbo
- **Takes**: 2-3 minutes

### Full Evaluation (10+ questions)
- **~$0.50-1.00** using GPT-3.5-turbo
- **Takes**: 5-10 minutes

## 🚀 Quick Start Commands

1. **Run simple evaluation:**
   ```bash
   python simple_ragas_eval.py
   ```

2. **Check results:**
   ```bash
   cat ragas_simple_results.csv
   ```

3. **Run full evaluation:**
   ```bash
   python ragas_evaluation.py
   ```

## 📁 Output Files

- `ragas_simple_results.csv` - Simple evaluation results
- `ragas_test_dataset.csv` - Generated test questions
- `ragas_results_YYYYMMDD_HHMMSS.csv` - Full results with timestamp

## ⚠️ Troubleshooting

### "OpenAI API key not found"
- Add `OPENAI_API_KEY=your_key` to `.env` file

### "Import error: ragas"
- Run `uv sync` to install dependencies

### "Evaluation taking too long"
- Use `simple_ragas_eval.py` instead
- Reduce number of test questions

### "Low scores across all metrics"
- Check if your RAG system is working
- Test with `python test_claim_search.py` first

## 🎯 Next Steps

1. **Run simple evaluation** first
2. **Analyze your scores** 
3. **Improve weak areas**
4. **Re-run evaluation** to see improvement
5. **Use full evaluation** for comprehensive testing

Remember: This is for learning! Don't worry about perfect scores initially. Focus on understanding what each metric means for your specific use case.