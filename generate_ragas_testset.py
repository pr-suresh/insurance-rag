#!/usr/bin/env python3
"""
RAGAS Test Dataset Generator
Creates auto-generated questions and answers from your CSV data
"""

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# Try different RAGAS imports for test generation
try:
    from ragas.testset.generator import TestsetGenerator
    print("✅ Using ragas.testset.generator")
    RAGAS_AVAILABLE = True
except ImportError:
    try:
        from ragas import TestsetGenerator
        print("✅ Using ragas.TestsetGenerator")
        RAGAS_AVAILABLE = True
    except ImportError:
        print("❌ RAGAS TestsetGenerator not available")
        RAGAS_AVAILABLE = False

def prepare_documents_from_csv(csv_path="data/insurance_claims.csv", max_docs=15):
    """Prepare documents from CSV for test generation."""
    
    print(f"📄 Loading documents from {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"📊 Found {len(df)} records in CSV")
    
    documents = []
    
    for idx, row in df.iterrows():
        # Create rich document content
        content = f"""
Claim Information:
- Claim ID: {row['Claim Number']}
- Claim Type: {row['Claim Feature']}
- Policy Number: {row['Policy Number']}
- Policy Effective Date: {row['Policy Effective Date']}
- Loss Description: {row['Loss Description']}
- Injury Type: {row['Type of injury']}
- Loss State: {row['Loss state']}
- Loss Date: {row['Loss Date']}
- Insured Company: {row['Insured Company Name']}

Financial Details:
- Paid Indemnity: ${row['Paid Indemnity']:,.2f}
- Paid DCC: ${row['Paid DCC']:,.2f}  
- Outstanding Indemnity: ${row['Outstanding Indemnity']:,.2f}
- Outstanding DCC: ${row['Outstanding DCC']:,.2f}
- Total Exposure: ${(row['Paid Indemnity'] + row['Paid DCC'] + row['Outstanding Indemnity'] + row['Outstanding DCC']):,.2f}

Adjuster Notes:
{row['Adjuster Notes'][:800] if pd.notna(row['Adjuster Notes']) else 'No additional notes available.'}
        """
        
        # Create document with metadata
        doc = Document(
            page_content=content.strip(),
            metadata={
                'claim_id': row['Claim Number'],
                'claim_type': row['Claim Feature'],
                'state': row['Loss state'],
                'total_amount': row['Paid Indemnity'] + row['Paid DCC'] + row['Outstanding Indemnity'] + row['Outstanding DCC'],
                'company': row['Insured Company Name']
            }
        )
        
        documents.append(doc)
        
        # Limit documents for faster processing
        if idx >= max_docs - 1:
            break
    
    print(f"✅ Prepared {len(documents)} documents for test generation")
    return documents

def generate_test_questions(num_samples=8):
    """Generate test questions using RAGAS."""
    
    if not RAGAS_AVAILABLE:
        print("❌ RAGAS not available for test generation")
        return create_sample_dataset()
    
    print(f"🧪 Generating {num_samples} test questions with RAGAS...")
    
    # Initialize models (using cheaper ones)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Prepare documents
    documents = prepare_documents_from_csv()
    
    try:
        # Initialize test generator
        generator = TestsetGenerator(
            llm=llm,
            embedding_model=embeddings
        )
        
        print("⏳ Generating questions... (this may take 2-3 minutes)")
        
        # Generate test set
        testset = generator.generate(
            documents=documents,
            test_size=num_samples,
            with_debugging_logs=False
        )
        
        # Convert to DataFrame
        test_df = testset.to_pandas()
        
        # Normalize column names
        if 'question' in test_df.columns and 'user_input' not in test_df.columns:
            test_df['user_input'] = test_df['question']
        if 'ground_truth' in test_df.columns and 'reference' not in test_df.columns:
            test_df['reference'] = test_df['ground_truth']
        
        print(f"✅ Generated {len(test_df)} test questions")
        
        # Save the dataset
        test_df.to_csv("ragas_generated_testset.csv", index=False)
        print("💾 Saved to 'ragas_generated_testset.csv'")
        
        return test_df
        
    except Exception as e:
        print(f"⚠️ RAGAS generation failed: {e}")
        print("📝 Creating sample dataset instead...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset to show the expected format."""
    
    sample_data = [
        {
            "user_input": "What is the total financial exposure for cyber liability claims?",
            "reference": "Cyber liability claims have significant exposure, with GL-2024-0025 showing $350,000 total exposure including $250,000 outstanding indemnity and $55,000 outstanding DCC for a data breach affecting 10,000 credit cards.",
            "evolution_type": "simple",
            "metadata": [{"claim_type": "Cyber Liability", "state": "VA"}]
        },
        {
            "user_input": "How do carbon monoxide poisoning claims typically get resolved and what are the liability factors?",
            "reference": "Carbon monoxide claims like GL-2024-0024 involve high liability (95%) due to installation negligence. Resolution includes immediate medical coverage, neuropsychological testing for cognitive effects, enhanced training programs, and settlements covering future medical monitoring.",
            "evolution_type": "reasoning",
            "metadata": [{"claim_type": "Completed Operations", "state": "MN"}]
        },
        {
            "user_input": "Compare the settlement patterns between slip and fall claims and professional liability claims.",
            "reference": "Slip and fall claims typically involve immediate bodily injury with premises liability, while professional liability claims involve errors and omissions with longer-term financial impacts. Both require liability assessment but have different injury severity scoring systems.",
            "evolution_type": "multi_context",
            "metadata": [{"claim_type": "Multiple", "state": "Various"}]
        }
    ]
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv("ragas_sample_testset.csv", index=False)
    print("💾 Saved sample dataset to 'ragas_sample_testset.csv'")
    
    return sample_df

def display_testset(test_df):
    """Display the generated test dataset in a nice format."""
    
    print("\n" + "="*80)
    print("📋 GENERATED TEST DATASET")
    print("="*80)
    
    for idx, row in test_df.iterrows():
        print(f"\n🔍 TEST QUESTION {idx + 1}:")
        print("-" * 50)
        
        question = row.get('user_input', row.get('question', 'Unknown'))
        answer = row.get('reference', row.get('ground_truth', 'Unknown'))
        evolution = row.get('evolution_type', 'Unknown')
        
        print(f"📝 Question: {question}")
        print(f"✅ Expected Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"🧬 Type: {evolution}")
        
        if 'metadata' in row and row['metadata']:
            print(f"📊 Context: {row['metadata']}")
    
    print("\n" + "="*80)
    print(f"📊 DATASET SUMMARY")
    print("-" * 30)
    print(f"Total Questions: {len(test_df)}")
    
    if 'evolution_type' in test_df.columns:
        type_counts = test_df['evolution_type'].value_counts()
        print("Question Types:")
        for qtype, count in type_counts.items():
            print(f"  {qtype}: {count}")
    
    print(f"\n💡 This dataset tests your RAG system's ability to:")
    print("  • Answer factual questions about specific claims")
    print("  • Perform reasoning across multiple claim contexts") 
    print("  • Compare and analyze patterns in claims data")
    print("  • Extract specific financial and technical details")

def main():
    """Main function to generate and display test dataset."""
    
    print("🚀 RAGAS TEST DATASET GENERATOR")
    print("="*50)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Add your OpenAI API key to .env file")
        return
    
    print("✅ OpenAI API key found")
    
    # Generate test dataset
    num_questions = int(input("\nHow many test questions to generate? (1-12): ") or "6")
    
    test_df = generate_test_questions(num_questions)
    
    # Display the results
    display_testset(test_df)
    
    print(f"\n🎯 Next Steps:")
    print("1. Review the generated questions above")
    print("2. Run evaluation: python ragas_evaluation.py")
    print("3. Check results in the generated CSV files")

if __name__ == "__main__":
    main()