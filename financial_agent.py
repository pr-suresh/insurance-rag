# financial_agent.py - AI Agent for Financial Analytics and Reporting
# Provides financial insights, reserve analysis, and cost analytics for insurance claims

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json

class FinancialAnalysisAgent:
    """AI Agent for financial analysis of insurance claims"""
    
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
        
        # Convert numeric columns and handle NaN
        numeric_columns = ['Paid Indemnity', 'Paid DCC', 'Outstanding Indemnity', 'Outstanding DCC']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Calculate total exposure per claim
        self.df['Total Paid'] = self.df['Paid Indemnity'] + self.df['Paid DCC']
        self.df['Total Outstanding'] = self.df['Outstanding Indemnity'] + self.df['Outstanding DCC']
        self.df['Total Exposure'] = self.df['Total Paid'] + self.df['Total Outstanding']
        
    def setup_tools(self):
        """Setup financial analysis tools"""
        
        @tool
        def calculate_total_exposure(state: str = None, claim_type: str = None, company: str = None) -> str:
            """Calculate total financial exposure across claims with various filters.
            Shows paid amounts, outstanding reserves, and total exposure.
            
            Args:
                state: Optional state code to filter
                claim_type: Optional claim type to filter
                company: Optional company name to filter
            """
            try:
                filtered_df = self.df.copy()
                
                # Apply filters
                if state:
                    filtered_df = filtered_df[filtered_df['Loss state'] == state.upper()]
                if claim_type:
                    filtered_df = filtered_df[filtered_df['Claim Feature'].str.contains(claim_type, case=False, na=False)]
                if company:
                    filtered_df = filtered_df[filtered_df['Insured Company Name'].str.contains(company, case=False, na=False)]
                
                if filtered_df.empty:
                    return "No claims found matching the specified criteria"
                
                # Calculate financial metrics
                exposure = {
                    'filter_criteria': {
                        'state': state or 'All',
                        'claim_type': claim_type or 'All',
                        'company': company or 'All'
                    },
                    'total_claims': len(filtered_df),
                    'financial_summary': {
                        'total_paid_indemnity': float(filtered_df['Paid Indemnity'].sum()),
                        'total_paid_dcc': float(filtered_df['Paid DCC'].sum()),
                        'total_paid': float(filtered_df['Total Paid'].sum()),
                        'total_outstanding_indemnity': float(filtered_df['Outstanding Indemnity'].sum()),
                        'total_outstanding_dcc': float(filtered_df['Outstanding DCC'].sum()),
                        'total_outstanding': float(filtered_df['Total Outstanding'].sum()),
                        'total_exposure': float(filtered_df['Total Exposure'].sum())
                    },
                    'averages': {
                        'avg_paid_per_claim': float(filtered_df['Total Paid'].mean()),
                        'avg_outstanding_per_claim': float(filtered_df['Total Outstanding'].mean()),
                        'avg_total_per_claim': float(filtered_df['Total Exposure'].mean())
                    },
                    'claim_status': {
                        'closed_claims': len(filtered_df[filtered_df['Total Outstanding'] == 0]),
                        'open_claims': len(filtered_df[filtered_df['Total Outstanding'] > 0]),
                        'closure_rate': f"{len(filtered_df[filtered_df['Total Outstanding'] == 0]) / len(filtered_df) * 100:.1f}%"
                    }
                }
                
                # Add top exposures
                top_exposures = filtered_df.nlargest(5, 'Total Exposure')[['Claim Number', 'Claim Feature', 'Total Exposure']]
                exposure['top_5_exposures'] = top_exposures.to_dict('records')
                
                return json.dumps(exposure, indent=2, default=str)
            except Exception as e:
                return f"Error calculating exposure: {str(e)}"
        
        @tool
        def analyze_reserve_adequacy(claim_type: str = None) -> str:
            """Analyze reserve adequacy by comparing initial reserves to actual payments.
            Identifies under-reserved or over-reserved claims.
            
            Args:
                claim_type: Optional claim type to filter analysis
            """
            try:
                filtered_df = self.df.copy()
                
                if claim_type:
                    filtered_df = filtered_df[filtered_df['Claim Feature'].str.contains(claim_type, case=False, na=False)]
                
                # Analyze closed claims (where outstanding = 0)
                closed_claims = filtered_df[filtered_df['Total Outstanding'] == 0]
                open_claims = filtered_df[filtered_df['Total Outstanding'] > 0]
                
                reserve_analysis = {
                    'analysis_scope': claim_type or 'All claim types',
                    'closed_claims_analysis': {
                        'total_closed': len(closed_claims),
                        'total_paid': float(closed_claims['Total Paid'].sum()),
                        'average_paid': float(closed_claims['Total Paid'].mean()),
                        'median_paid': float(closed_claims['Total Paid'].median()),
                        'payment_distribution': {
                            'under_25k': len(closed_claims[closed_claims['Total Paid'] < 25000]),
                            '25k_to_50k': len(closed_claims[(closed_claims['Total Paid'] >= 25000) & (closed_claims['Total Paid'] < 50000)]),
                            '50k_to_100k': len(closed_claims[(closed_claims['Total Paid'] >= 50000) & (closed_claims['Total Paid'] < 100000)]),
                            'over_100k': len(closed_claims[closed_claims['Total Paid'] >= 100000])
                        }
                    },
                    'open_claims_analysis': {
                        'total_open': len(open_claims),
                        'total_outstanding': float(open_claims['Total Outstanding'].sum()),
                        'total_already_paid': float(open_claims['Total Paid'].sum()),
                        'projected_total': float(open_claims['Total Exposure'].sum()),
                        'average_outstanding': float(open_claims['Total Outstanding'].mean())
                    }
                }
                
                # Calculate reserve accuracy metrics if we have both paid and outstanding
                claims_with_payments = filtered_df[(filtered_df['Total Paid'] > 0) & (filtered_df['Total Outstanding'] > 0)]
                if not claims_with_payments.empty:
                    reserve_analysis['reserve_indicators'] = {
                        'claims_with_ongoing_payments': len(claims_with_payments),
                        'paid_to_outstanding_ratio': float(claims_with_payments['Total Paid'].sum() / claims_with_payments['Total Outstanding'].sum()),
                        'likely_under_reserved': len(claims_with_payments[claims_with_payments['Total Paid'] > claims_with_payments['Total Outstanding'] * 2]),
                        'likely_over_reserved': len(claims_with_payments[claims_with_payments['Total Paid'] < claims_with_payments['Total Outstanding'] * 0.5])
                    }
                
                # Recommendations based on analysis
                recommendations = []
                if open_claims['Total Outstanding'].sum() > closed_claims['Total Paid'].mean() * len(open_claims):
                    recommendations.append("Reserves appear adequate based on historical settlements")
                else:
                    recommendations.append("Consider reviewing reserves - may be under-reserved based on historical data")
                
                if len(open_claims) > len(closed_claims):
                    recommendations.append("High number of open claims - focus on claim closure strategies")
                
                reserve_analysis['recommendations'] = recommendations
                
                return json.dumps(reserve_analysis, indent=2, default=str)
            except Exception as e:
                return f"Error analyzing reserves: {str(e)}"
        
        @tool
        def generate_cost_driver_report(top_n: int = 10) -> str:
            """Generate a report identifying the primary cost drivers in claims.
            Shows which factors contribute most to claim costs.
            
            Args:
                top_n: Number of top cost drivers to include (default: 10)
            """
            try:
                # Identify cost drivers
                cost_drivers = {
                    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                    'total_claims_analyzed': len(self.df),
                    'total_exposure': float(self.df['Total Exposure'].sum()),
                    'cost_drivers_by_category': {}
                }
                
                # By Claim Feature
                by_feature = self.df.groupby('Claim Feature').agg({
                    'Total Exposure': ['sum', 'mean', 'count']
                }).round(2)
                by_feature.columns = ['total_cost', 'avg_cost', 'claim_count']
                by_feature = by_feature.sort_values('total_cost', ascending=False).head(top_n)
                cost_drivers['cost_drivers_by_category']['by_claim_type'] = by_feature.to_dict('index')
                
                # By State
                by_state = self.df.groupby('Loss state').agg({
                    'Total Exposure': ['sum', 'mean', 'count']
                }).round(2)
                by_state.columns = ['total_cost', 'avg_cost', 'claim_count']
                by_state = by_state.sort_values('total_cost', ascending=False).head(top_n)
                cost_drivers['cost_drivers_by_category']['by_state'] = by_state.to_dict('index')
                
                # By Company
                by_company = self.df.groupby('Insured Company Name').agg({
                    'Total Exposure': ['sum', 'mean', 'count']
                }).round(2)
                by_company.columns = ['total_cost', 'avg_cost', 'claim_count']
                by_company = by_company.sort_values('total_cost', ascending=False).head(top_n)
                cost_drivers['cost_drivers_by_category']['by_company'] = by_company.to_dict('index')
                
                # By Injury Type
                by_injury = self.df.groupby('Type of injury').agg({
                    'Total Exposure': ['sum', 'mean', 'count']
                }).round(2)
                by_injury.columns = ['total_cost', 'avg_cost', 'claim_count']
                cost_drivers['cost_drivers_by_category']['by_injury_type'] = by_injury.to_dict('index')
                
                # Identify high-cost patterns
                high_cost_threshold = self.df['Total Exposure'].quantile(0.9)
                high_cost_claims = self.df[self.df['Total Exposure'] > high_cost_threshold]
                
                cost_drivers['high_cost_patterns'] = {
                    'threshold': float(high_cost_threshold),
                    'count': len(high_cost_claims),
                    'total_exposure': float(high_cost_claims['Total Exposure'].sum()),
                    'percentage_of_total': f"{high_cost_claims['Total Exposure'].sum() / self.df['Total Exposure'].sum() * 100:.1f}%",
                    'common_features': high_cost_claims['Claim Feature'].value_counts().head(3).to_dict()
                }
                
                # Cost trends
                cost_drivers['dcc_analysis'] = {
                    'avg_dcc_ratio': float((self.df['Paid DCC'] + self.df['Outstanding DCC']).sum() / 
                                          (self.df['Total Exposure'].sum()) * 100),
                    'claims_with_high_dcc': len(self.df[(self.df['Paid DCC'] + self.df['Outstanding DCC']) > 
                                                        (self.df['Paid Indemnity'] + self.df['Outstanding Indemnity']) * 0.3])
                }
                
                return json.dumps(cost_drivers, indent=2, default=str)
            except Exception as e:
                return f"Error generating cost driver report: {str(e)}"
        
        @tool
        def calculate_settlement_efficiency(days_threshold: int = 180) -> str:
            """Calculate settlement efficiency metrics to identify quick vs prolonged settlements.
            Analyzes the relationship between settlement speed and costs.
            
            Args:
                days_threshold: Number of days to classify as 'quick' settlement (default: 180)
            """
            try:
                # For this analysis, we'll look at closed claims
                closed_claims = self.df[self.df['Total Outstanding'] == 0].copy()
                
                if closed_claims.empty:
                    return "No closed claims available for efficiency analysis"
                
                efficiency_metrics = {
                    'analysis_parameters': {
                        'quick_settlement_threshold': f"{days_threshold} days"
                    },
                    'overall_metrics': {
                        'total_closed_claims': len(closed_claims),
                        'total_paid': float(closed_claims['Total Paid'].sum()),
                        'average_payment': float(closed_claims['Total Paid'].mean()),
                        'median_payment': float(closed_claims['Total Paid'].median())
                    }
                }
                
                # Analyze by claim amount brackets
                brackets = {
                    'small_claims': closed_claims[closed_claims['Total Paid'] < 25000],
                    'medium_claims': closed_claims[(closed_claims['Total Paid'] >= 25000) & 
                                                  (closed_claims['Total Paid'] < 100000)],
                    'large_claims': closed_claims[closed_claims['Total Paid'] >= 100000]
                }
                
                efficiency_metrics['by_claim_size'] = {}
                for bracket_name, bracket_df in brackets.items():
                    if not bracket_df.empty:
                        efficiency_metrics['by_claim_size'][bracket_name] = {
                            'count': len(bracket_df),
                            'average_paid': float(bracket_df['Total Paid'].mean()),
                            'average_dcc': float(bracket_df['Paid DCC'].mean()),
                            'dcc_ratio': f"{bracket_df['Paid DCC'].sum() / bracket_df['Total Paid'].sum() * 100:.1f}%"
                        }
                
                # Analyze DCC as indicator of complexity/efficiency
                efficiency_metrics['dcc_analysis'] = {
                    'claims_with_low_dcc': len(closed_claims[closed_claims['Paid DCC'] < closed_claims['Paid Indemnity'] * 0.2]),
                    'claims_with_high_dcc': len(closed_claims[closed_claims['Paid DCC'] > closed_claims['Paid Indemnity'] * 0.4]),
                    'average_dcc_ratio': f"{closed_claims['Paid DCC'].sum() / closed_claims['Total Paid'].sum() * 100:.1f}%"
                }
                
                # Identify efficient vs inefficient claim types
                by_type_efficiency = closed_claims.groupby('Claim Feature').agg({
                    'Total Paid': 'mean',
                    'Paid DCC': 'mean',
                    'Claim Number': 'count'
                })
                by_type_efficiency['dcc_ratio'] = by_type_efficiency['Paid DCC'] / by_type_efficiency['Total Paid']
                
                most_efficient = by_type_efficiency.nsmallest(3, 'dcc_ratio')
                least_efficient = by_type_efficiency.nlargest(3, 'dcc_ratio')
                
                efficiency_metrics['efficiency_by_type'] = {
                    'most_efficient': most_efficient[['Total Paid', 'dcc_ratio']].to_dict('index'),
                    'least_efficient': least_efficient[['Total Paid', 'dcc_ratio']].to_dict('index')
                }
                
                # Recommendations
                recommendations = []
                if closed_claims['Paid DCC'].mean() > closed_claims['Paid Indemnity'].mean() * 0.3:
                    recommendations.append("High DCC costs detected - consider streamlining claim processing")
                
                if len(brackets.get('large_claims', [])) > len(closed_claims) * 0.2:
                    recommendations.append("High proportion of large claims - consider early intervention strategies")
                
                efficiency_metrics['recommendations'] = recommendations
                
                return json.dumps(efficiency_metrics, indent=2, default=str)
            except Exception as e:
                return f"Error calculating settlement efficiency: {str(e)}"
        
        @tool
        def generate_financial_summary(year: str = None) -> str:
            """Generate a comprehensive financial summary report.
            Provides executive-level overview of claims financials.
            
            Args:
                year: Optional year to filter (e.g., '2023', '2024')
            """
            try:
                filtered_df = self.df.copy()
                
                # Filter by year if specified
                if year:
                    filtered_df = filtered_df[filtered_df['Loss Date'].dt.year == int(year)]
                    period = year
                else:
                    period = "All Time"
                
                if filtered_df.empty:
                    return f"No claims found for year: {year}"
                
                # Generate comprehensive summary
                summary = {
                    'report_date': datetime.now().strftime('%Y-%m-%d'),
                    'period': period,
                    'executive_summary': {
                        'total_claims': len(filtered_df),
                        'total_exposure': float(filtered_df['Total Exposure'].sum()),
                        'total_paid_to_date': float(filtered_df['Total Paid'].sum()),
                        'total_outstanding': float(filtered_df['Total Outstanding'].sum()),
                        'average_claim_value': float(filtered_df['Total Exposure'].mean()),
                        'median_claim_value': float(filtered_df['Total Exposure'].median())
                    },
                    'breakdown_by_status': {
                        'closed_claims': {
                            'count': len(filtered_df[filtered_df['Total Outstanding'] == 0]),
                            'total_paid': float(filtered_df[filtered_df['Total Outstanding'] == 0]['Total Paid'].sum())
                        },
                        'open_claims': {
                            'count': len(filtered_df[filtered_df['Total Outstanding'] > 0]),
                            'outstanding_reserves': float(filtered_df[filtered_df['Total Outstanding'] > 0]['Total Outstanding'].sum()),
                            'already_paid': float(filtered_df[filtered_df['Total Outstanding'] > 0]['Total Paid'].sum())
                        }
                    },
                    'top_exposures': {
                        'by_state': filtered_df.groupby('Loss state')['Total Exposure'].sum().nlargest(5).to_dict(),
                        'by_type': filtered_df.groupby('Claim Feature')['Total Exposure'].sum().nlargest(5).to_dict()
                    },
                    'cost_components': {
                        'indemnity': {
                            'paid': float(filtered_df['Paid Indemnity'].sum()),
                            'outstanding': float(filtered_df['Outstanding Indemnity'].sum()),
                            'total': float(filtered_df['Paid Indemnity'].sum() + filtered_df['Outstanding Indemnity'].sum())
                        },
                        'dcc': {
                            'paid': float(filtered_df['Paid DCC'].sum()),
                            'outstanding': float(filtered_df['Outstanding DCC'].sum()),
                            'total': float(filtered_df['Paid DCC'].sum() + filtered_df['Outstanding DCC'].sum())
                        },
                        'dcc_ratio': f"{(filtered_df['Paid DCC'].sum() + filtered_df['Outstanding DCC'].sum()) / filtered_df['Total Exposure'].sum() * 100:.1f}%"
                    },
                    'risk_indicators': {
                        'claims_over_100k': len(filtered_df[filtered_df['Total Exposure'] > 100000]),
                        'claims_over_250k': len(filtered_df[filtered_df['Total Exposure'] > 250000]),
                        'total_exposure_top_10_claims': float(filtered_df.nlargest(10, 'Total Exposure')['Total Exposure'].sum()),
                        'concentration_risk': f"{filtered_df.nlargest(10, 'Total Exposure')['Total Exposure'].sum() / filtered_df['Total Exposure'].sum() * 100:.1f}%"
                    }
                }
                
                # Add trend analysis if we have enough data
                if len(filtered_df) > 20:
                    monthly_trend = filtered_df.groupby(filtered_df['Loss Date'].dt.to_period('M'))['Total Exposure'].sum()
                    if len(monthly_trend) > 1:
                        summary['trend_indicator'] = {
                            'direction': 'increasing' if monthly_trend.iloc[-1] > monthly_trend.mean() else 'decreasing',
                            'recent_month': float(monthly_trend.iloc[-1]) if len(monthly_trend) > 0 else 0,
                            'average_monthly': float(monthly_trend.mean())
                        }
                
                return json.dumps(summary, indent=2, default=str)
            except Exception as e:
                return f"Error generating financial summary: {str(e)}"
        
        # Store tools for access
        self.tools = [
            calculate_total_exposure,
            analyze_reserve_adequacy,
            generate_cost_driver_report,
            calculate_settlement_efficiency,
            generate_financial_summary
        ]
        
    def get_tools(self):
        """Return the list of financial analysis tools"""
        return self.tools
    
    def analyze_with_llm(self, query: str) -> str:
        """Use LLM to provide natural language financial analysis"""
        prompt = f"""
        As an insurance financial analyst, analyze the following query:
        {query}
        
        Provide financial insights focusing on:
        - Cost analysis and trends
        - Reserve adequacy
        - Settlement efficiency
        - Risk exposure
        - Actionable recommendations
        """
        
        response = self.llm.invoke(prompt)
        return response.content