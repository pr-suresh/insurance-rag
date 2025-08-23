# pattern_analysis_agent.py - AI Agent for Claims Pattern Analysis
# Analyzes historical claims to identify patterns, trends, and predictive insights

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from collections import Counter
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json

class PatternAnalysisAgent:
    """AI Agent for analyzing patterns in insurance claims data"""
    
    def __init__(self, claims_data_path: str = "data/insurance_claims.csv"):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.load_claims_data(claims_data_path)
        self.setup_tools()
        
    def load_claims_data(self, path: str):
        """Load and preprocess claims data"""
        self.df = pd.read_csv(path)
        # Convert date columns
        date_columns = ['Loss Date', 'Policy Effective Date']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['Paid Indemnity', 'Paid DCC', 'Outstanding Indemnity', 'Outstanding DCC']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
    def setup_tools(self):
        """Setup pattern analysis tools"""
        
        @tool
        def analyze_claim_patterns_by_type(claim_type: str = None) -> str:
            """Analyze patterns in claims by type of injury or claim feature.
            Provides statistics on frequency, average costs, and common characteristics.
            
            Args:
                claim_type: Optional specific claim type to analyze (e.g., 'Slip and Fall', 'Product Liability')
            """
            try:
                if claim_type:
                    # Filter for specific claim type
                    filtered_df = self.df[
                        (self.df['Claim Feature'].str.contains(claim_type, case=False, na=False)) |
                        (self.df['Type of injury'].str.contains(claim_type, case=False, na=False))
                    ]
                    if filtered_df.empty:
                        return f"No claims found for type: {claim_type}"
                else:
                    filtered_df = self.df
                
                # Calculate statistics
                stats = {
                    'total_claims': len(filtered_df),
                    'by_claim_feature': filtered_df['Claim Feature'].value_counts().to_dict(),
                    'by_injury_type': filtered_df['Type of injury'].value_counts().to_dict(),
                    'by_state': filtered_df['Loss state'].value_counts().head(5).to_dict(),
                    'avg_paid_indemnity': float(filtered_df['Paid Indemnity'].mean()),
                    'avg_outstanding': float(filtered_df['Outstanding Indemnity'].mean()),
                    'total_exposure': float(filtered_df['Paid Indemnity'].sum() + filtered_df['Outstanding Indemnity'].sum())
                }
                
                # Identify high-risk patterns
                high_cost_claims = filtered_df[filtered_df['Paid Indemnity'] > filtered_df['Paid Indemnity'].quantile(0.75)]
                if not high_cost_claims.empty:
                    stats['high_risk_features'] = high_cost_claims['Claim Feature'].value_counts().head(3).to_dict()
                
                return json.dumps(stats, indent=2, default=str)
            except Exception as e:
                return f"Error analyzing patterns: {str(e)}"
        
        @tool
        def identify_risk_indicators(state: str = None, company: str = None) -> str:
            """Identify risk indicators and patterns from historical claims.
            Analyzes adjuster notes for common risk factors and liability assessments.
            
            Args:
                state: Optional state code to filter analysis
                company: Optional company name to filter analysis
            """
            try:
                filtered_df = self.df.copy()
                
                if state:
                    filtered_df = filtered_df[filtered_df['Loss state'] == state.upper()]
                if company:
                    filtered_df = filtered_df[filtered_df['Insured Company Name'].str.contains(company, case=False, na=False)]
                
                if filtered_df.empty:
                    return "No matching claims found for the specified criteria"
                
                # Extract liability assessments from notes
                liability_patterns = []
                risk_factors = []
                
                for notes in filtered_df['Adjuster Notes'].dropna():
                    # Extract liability percentages
                    liability_matches = re.findall(r'LIABILITY[:\s]+(?:ASSESSMENT[:\s]+)?(?:Very\s+)?(?:High|Strong|Clear|Moderate|Low)?\s*\(?(\d+)%?\)?', notes, re.IGNORECASE)
                    liability_patterns.extend(liability_matches)
                    
                    # Extract common risk factors
                    if 'inadequate' in notes.lower():
                        risk_factors.append('Inadequate safety measures')
                    if 'deferred' in notes.lower() or 'delayed' in notes.lower():
                        risk_factors.append('Deferred maintenance')
                    if 'prior' in notes.lower() and ('incident' in notes.lower() or 'claim' in notes.lower()):
                        risk_factors.append('Prior incidents')
                    if 'defect' in notes.lower():
                        risk_factors.append('Product/Design defect')
                    if 'negligence' in notes.lower() or 'negligent' in notes.lower():
                        risk_factors.append('Negligence')
                
                # Calculate risk metrics
                risk_analysis = {
                    'total_claims_analyzed': len(filtered_df),
                    'average_liability_assessment': np.mean([int(x) for x in liability_patterns if x]) if liability_patterns else 0,
                    'high_liability_claims': len([x for x in liability_patterns if x and int(x) >= 75]),
                    'common_risk_factors': dict(Counter(risk_factors).most_common(5)),
                    'avg_total_cost': float(filtered_df['Paid Indemnity'].sum() + filtered_df['Outstanding Indemnity'].sum()) / len(filtered_df),
                    'claims_with_prior_incidents': sum(1 for notes in filtered_df['Adjuster Notes'].dropna() if 'prior' in notes.lower())
                }
                
                # Add state-specific insights if applicable
                if state:
                    risk_analysis['state_specific'] = {
                        'state': state,
                        'total_exposure': float(filtered_df['Paid Indemnity'].sum() + filtered_df['Outstanding Indemnity'].sum()),
                        'most_common_claim_type': filtered_df['Claim Feature'].mode()[0] if not filtered_df['Claim Feature'].empty else 'N/A'
                    }
                
                return json.dumps(risk_analysis, indent=2, default=str)
            except Exception as e:
                return f"Error identifying risk indicators: {str(e)}"
        
        @tool  
        def predict_claim_outcome(claim_feature: str, injury_type: str, state: str) -> str:
            """Predict likely outcome of a claim based on historical similar cases.
            Provides settlement estimates and timeline predictions.
            
            Args:
                claim_feature: Type of claim (e.g., 'Slip and Fall', 'Product Liability')
                injury_type: Type of injury (e.g., 'Bodily Injury', 'Property Damage')
                state: State where the loss occurred
            """
            try:
                # Find similar claims
                similar_claims = self.df[
                    (self.df['Claim Feature'].str.contains(claim_feature, case=False, na=False)) &
                    (self.df['Type of injury'] == injury_type) &
                    (self.df['Loss state'] == state.upper())
                ]
                
                if similar_claims.empty:
                    # Broaden search if no exact matches
                    similar_claims = self.df[
                        (self.df['Claim Feature'].str.contains(claim_feature, case=False, na=False)) |
                        (self.df['Type of injury'] == injury_type)
                    ]
                
                if similar_claims.empty:
                    return "No similar historical claims found for prediction"
                
                # Calculate predictions
                predictions = {
                    'based_on_claims': len(similar_claims),
                    'predicted_indemnity': {
                        'minimum': float(similar_claims['Paid Indemnity'].min()),
                        'average': float(similar_claims['Paid Indemnity'].mean()),
                        'maximum': float(similar_claims['Paid Indemnity'].max()),
                        'median': float(similar_claims['Paid Indemnity'].median())
                    },
                    'predicted_dcc': {
                        'average': float(similar_claims['Paid DCC'].mean()),
                        'median': float(similar_claims['Paid DCC'].median())
                    },
                    'total_expected_cost': float(similar_claims['Paid Indemnity'].median() + similar_claims['Paid DCC'].median()),
                    'settlement_likelihood': f"{len(similar_claims[similar_claims['Paid Indemnity'] > 0]) / len(similar_claims) * 100:.1f}%",
                    'common_complications': []
                }
                
                # Extract common complications from notes
                complications = []
                for notes in similar_claims['Adjuster Notes'].dropna().head(10):
                    if 'litigation' in notes.lower():
                        complications.append('Litigation risk')
                    if 'medical' in notes.lower() and 'ongoing' in notes.lower():
                        complications.append('Ongoing medical treatment')
                    if 'attorney' in notes.lower() or 'counsel' in notes.lower():
                        complications.append('Legal representation')
                
                predictions['common_complications'] = list(set(complications))
                
                # Add confidence score based on sample size
                if len(similar_claims) >= 10:
                    predictions['confidence'] = 'High'
                elif len(similar_claims) >= 5:
                    predictions['confidence'] = 'Medium'
                else:
                    predictions['confidence'] = 'Low'
                
                return json.dumps(predictions, indent=2, default=str)
            except Exception as e:
                return f"Error predicting claim outcome: {str(e)}"
        
        @tool
        def analyze_settlement_trends(time_period: str = "2023") -> str:
            """Analyze settlement trends over time to identify patterns.
            Shows how claim costs and frequencies are changing.
            
            Args:
                time_period: Year or date range to analyze (default: 2023)
            """
            try:
                # Filter by time period
                if '-' in time_period:  # Date range
                    start, end = time_period.split('-')
                    mask = (self.df['Loss Date'] >= pd.to_datetime(start)) & (self.df['Loss Date'] <= pd.to_datetime(end))
                else:  # Single year
                    mask = self.df['Loss Date'].dt.year == int(time_period)
                
                period_claims = self.df[mask]
                
                if period_claims.empty:
                    return f"No claims found for period: {time_period}"
                
                # Calculate trends
                trends = {
                    'time_period': time_period,
                    'total_claims': len(period_claims),
                    'total_paid': float(period_claims['Paid Indemnity'].sum()),
                    'total_outstanding': float(period_claims['Outstanding Indemnity'].sum()),
                    'average_claim_value': float(period_claims['Paid Indemnity'].mean()),
                    'median_claim_value': float(period_claims['Paid Indemnity'].median()),
                    'claim_distribution': {}
                }
                
                # Monthly trend if data permits
                if len(period_claims) > 10:
                    monthly = period_claims.groupby(period_claims['Loss Date'].dt.to_period('M')).agg({
                        'Claim Number': 'count',
                        'Paid Indemnity': 'sum'
                    })
                    trends['monthly_pattern'] = {
                        str(idx): {'count': int(row['Claim Number']), 'total_paid': float(row['Paid Indemnity'])}
                        for idx, row in monthly.iterrows()
                    }
                
                # Top cost drivers
                trends['top_cost_drivers'] = period_claims.nlargest(5, 'Paid Indemnity')[['Claim Number', 'Claim Feature', 'Paid Indemnity']].to_dict('records')
                
                # Settlement speed (if we had settlement date, we'd calculate time to settlement)
                trends['claim_types'] = period_claims['Claim Feature'].value_counts().head(5).to_dict()
                
                return json.dumps(trends, indent=2, default=str)
            except Exception as e:
                return f"Error analyzing settlement trends: {str(e)}"
        
        # Store tools for access
        self.tools = [
            analyze_claim_patterns_by_type,
            identify_risk_indicators,
            predict_claim_outcome,
            analyze_settlement_trends
        ]
        
    def get_tools(self):
        """Return the list of pattern analysis tools"""
        return self.tools
    
    def analyze_with_llm(self, query: str) -> str:
        """Use LLM to provide natural language analysis of patterns"""
        # This method can be used to provide more sophisticated analysis
        # combining multiple tool outputs with LLM reasoning
        prompt = f"""
        As an insurance claims pattern analyst, analyze the following query:
        {query}
        
        Use the available data to identify patterns, trends, and actionable insights.
        Focus on risk factors, cost drivers, and predictive indicators.
        """
        
        response = self.llm.invoke(prompt)
        return response.content