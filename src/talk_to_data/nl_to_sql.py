import os
import re
from src.utils.logger import setup_logger
from src.utils.config import GEMINI_API_KEY
from src.talk_to_data.prompt_templates import SQL_SYSTEM_PROMPT, ANALYSIS_SYSTEM_PROMPT
from src.talk_to_data.query_runner import SafeQueryRunner

logger = setup_logger("nl_to_sql")

class TalkToDataSystem:
    def __init__(self, gemini_api_key=GEMINI_API_KEY):
        self.api_key = gemini_api_key
        self.runner = SafeQueryRunner()
        self.client = None
        self.init_llm_client()

    def init_llm_client(self):
        """Initialise Gemini client if API key is provided."""
        if self.api_key:
            try:
                # Support both modern google-genai and standard google-generativeai SDKs
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Gemini google-genai client loaded successfully.")
            except ImportError:
                try:
                    import google.generativeai as genai_legacy
                    genai_legacy.configure(api_key=self.api_key)
                    self.client = genai_legacy
                    logger.info("Fallback: Legacy google-generativeai client loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load Gemini SDK: {str(e)}. Fallback to rule-based parser active.")
            except Exception as e:
                logger.warning(f"Could not connect to Gemini: {str(e)}. Fallback to rule-based parser active.")

    def translate_and_analyze(self, user_question: str) -> dict:
        """Translates NL to SQL, runs the query, and generates professional summaries."""
        logger.info(f"Received Natural Language query: '{user_question}'")
        
        # 1. Translate NL to SQL (LLM or Rule-based fallback)
        sql = self.translate_nl_to_sql(user_question)
        
        # 2. Run SQL query securely
        results = self.runner.execute_query(sql)
        
        # 3. Generate narrative business analysis (LLM or Rule-based fallback)
        analysis = self.generate_analysis(user_question, sql, results)
        
        results["analysis"] = analysis
        return results

    def translate_nl_to_sql(self, user_question: str) -> str:
        """Translates user questions to SQLite query string."""
        # Try Rule-based hybrid parser first for common exact questions
        rule_sql = self._parse_rules_to_sql(user_question)
        if rule_sql:
            logger.info("Translated using robust rule-based SQL generator.")
            return rule_sql
            
        # Fallback to LLM if key is available
        if self.client:
            try:
                # modern client execution
                if hasattr(self.client, "models"):
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"{SQL_SYSTEM_PROMPT}\n\nUser Question: {user_question}"
                    )
                    return response.text.strip()
                # legacy client execution
                else:
                    model = self.client.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(
                        f"{SQL_SYSTEM_PROMPT}\n\nUser Question: {user_question}"
                    )
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini SQL Translation Error: {str(e)}. Using emergency fallback queries.")
                
        # Ultimate emergency default query if LLM fails and no rules matched
        return "SELECT TARGET, COUNT(*) as count, ROUND(AVG(AMT_INCOME_TOTAL), 2) as avg_income FROM applications GROUP BY TARGET;"

    def generate_analysis(self, user_question: str, sql: str, results: dict) -> str:
        """Generates clear, expert narrative credit risk analysis of the query outcomes."""
        if results["status"] == "error":
            return f"Credit analysis halted due to execution error: {results['message']}"

        # If LLM is available, leverage it for bespoke business insights
        if self.client:
            try:
                prompt = f"""
                {ANALYSIS_SYSTEM_PROMPT}
                
                User Question: {user_question}
                SQL Executed: {sql}
                Results: {results['data']}
                """
                if hasattr(self.client, "models"):
                    response = self.client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt
                    )
                    return response.text.strip()
                else:
                    model = self.client.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(prompt)
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini Analysis Summary Error: {str(e)}")

        # Fallback to rich deterministic business analysis text
        return self._generate_rule_based_analysis(user_question, results)

    def _parse_rules_to_sql(self, text: str) -> str:
        """Heuristic parser representing 6+ vital analytical queries."""
        text = text.lower().strip()
        
        # Query 1: Average income by default/target status
        if re.search(r"average.*income.*(default|target)", text) or re.search(r"income.*(default|target|repaid)", text):
            return """
SELECT 
    CASE WHEN TARGET = 1 THEN 'Default / Delinquent' ELSE 'Repaid / Active' END as Default_Status,
    COUNT(*) as total_applicants,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) as average_income_usd,
    ROUND(AVG(AMT_CREDIT), 2) as average_loan_amount_usd
FROM applications 
GROUP BY TARGET;
"""
        # Query 2: Defaults/risk rating by occupation type
        if re.search(r"occupation.*(default|risk)", text) or re.search(r"default.*occupation", text):
            return """
SELECT 
    OCCUPATION_TYPE as Occupation,
    COUNT(*) as total_applicants,
    SUM(TARGET) as default_cases,
    ROUND(AVG(TARGET)*100, 2) as default_rate_percent
FROM applications 
WHERE OCCUPATION_TYPE IS NOT NULL 
GROUP BY OCCUPATION_TYPE 
ORDER BY default_rate_percent DESC;
"""
        # Query 3: Default counts or rate by gender
        if re.search(r"gender.*(default|repaid)", text) or re.search(r"male.*female.*default", text):
            return """
SELECT 
    CODE_GENDER as Gender,
    COUNT(*) as total_applicants,
    SUM(TARGET) as default_cases,
    ROUND(AVG(TARGET)*100, 2) as default_rate_percent
FROM applications 
WHERE CODE_GENDER IN ('M', 'F')
GROUP BY CODE_GENDER;
"""
        # Query 4: Credit vs Income type correlations
        if re.search(r"credit.*income.*type", text) or re.search(r"credit.*vs.*income", text):
            return """
SELECT 
    NAME_INCOME_TYPE as Income_Source,
    COUNT(*) as applicant_count,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) as avg_income_usd,
    ROUND(AVG(AMT_CREDIT), 2) as avg_loan_usd,
    ROUND(AVG(AMT_CREDIT)/AVG(AMT_INCOME_TOTAL), 2) as leverage_ratio
FROM applications 
GROUP BY NAME_INCOME_TYPE 
ORDER BY leverage_ratio DESC;
"""
        # Query 5: Previous applications statuses (approved vs refused)
        if re.search(r"previous.*(status|approved|refused)", text) or re.search(r"approved.*refused", text):
            return """
SELECT 
    NAME_CONTRACT_STATUS as Previous_Status,
    COUNT(*) as count,
    ROUND(AVG(AMT_APPLICATION), 2) as avg_requested_amt_usd,
    ROUND(AVG(AMT_CREDIT), 2) as avg_funded_amt_usd
FROM previous_applications 
GROUP BY NAME_CONTRACT_STATUS;
"""
        # Query 6: Default breakdown overview
        if re.search(r"(how many|total|percentage).*default", text) or re.search(r"default.*rate", text):
            return """
SELECT 
    CASE WHEN TARGET = 1 THEN 'Default / Delinquent' ELSE 'Repaid / Active' END as Default_Status,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applications), 2) as percentage
FROM applications 
GROUP BY TARGET;
"""
        return None

    def _generate_rule_based_analysis(self, question: str, results: dict) -> str:
        """Determines context and prints top-tier narrative expert summaries."""
        data = results.get("data", [])
        if not data:
            return "No data returned to formulate a business analysis summary."
            
        columns = results.get("columns", [])
        q_lower = question.lower()
        
        # Analyze outcome based on headers and matched content
        if "default_status" in [c.lower() for c in columns] and "average_income_usd" in [c.lower() for c in columns]:
            non_def_inc = next((d.get("average_income_usd") for d in data if "repaid" in str(d.get("Default_Status")).lower()), 0)
            def_inc = next((d.get("average_income_usd") for d in data if "default" in str(d.get("Default_Status")).lower()), 0)
            
            diff = non_def_inc - def_inc
            if diff > 0:
                return f"**Analytical Insight**: Active / Repaid clients exhibit a higher average gross income (${non_def_inc:,.2f}) compared to delinquent defaults (${def_inc:,.2f}), representing a margin difference of ${diff:,.2f}. This strongly supports utilizing gross income margins as a fundamental filtering criterion in initial scorecard calibrations."
            else:
                return f"**Analytical Insight**: Average income levels show close proximity (Repaid: ${non_def_inc:,.2f} vs Default: ${def_inc:,.2f}). This indicates that nominal gross income by itself is an insufficient risk separator, suggesting that debt-to-income ratios and external bureau scores are far more significant features."

        elif "default_rate_percent" in [c.lower() for c in columns] and "occupation" in [c.lower() for c in columns]:
            top_risk = data[0].get("Occupation", "Unknown")
            top_rate = data[0].get("default_rate_percent", 0)
            return f"**Analytical Insight**: Analysis of the occupational portfolio indicates that **{top_risk}** applications carry the highest delinquent risk ratios, peaking at **{top_rate}%**. Risk models should assign high-density risk offsets or require lower debt-service limits for this occupational demographic segment."

        elif "gender" in [c.lower() for c in columns] and "default_rate_percent" in [c.lower() for c in columns]:
            m_rate = next((d.get("default_rate_percent") for d in data if str(d.get("Gender")).upper() == "M"), 0)
            f_rate = next((d.get("default_rate_percent") for d in data if str(d.get("Gender")).upper() == "F"), 0)
            
            recomm = "Male applicants exhibit historically higher default margins" if m_rate > f_rate else "Female applicants exhibit higher delinquent ratios"
            return f"**Analytical Insight**: Gender demographic analysis shows Male defaults at **{m_rate}%** and Female defaults at **{f_rate}%**. {recomm}. Credit policies should review if auxiliary factors like contract type or car ownership explain this divergence before implementing explicit policy weights."

        elif "leverage_ratio" in [c.lower() for c in columns]:
            highest_leverage_source = data[0].get("Income_Source", "Unknown")
            highest_lev = data[0].get("leverage_ratio", 0)
            return f"**Analytical Insight**: The highest financial leverage ratio (credit sum / annual income) is observed in the **{highest_leverage_source}** category, scaling at **{highest_lev}x**. This exceeds normal baseline tolerances and should be mitigated by collateral or secondary asset verification."

        elif "previous_status" in [c.lower() for c in columns]:
            refused_cnt = next((d.get("count") for d in data if "refused" in str(d.get("Previous_Status")).lower()), 0)
            approved_cnt = next((d.get("count") for d in data if "approved" in str(d.get("Previous_Status")).lower()), 0)
            
            ratio = refused_cnt / (approved_cnt + 1e-5)
            return f"**Analytical Insight**: The internal credit system has processed **{approved_cnt} approved applications** and **{refused_cnt} refusals**. This represents a refusal-to-approval ratio of **{ratio*100:.1f}%**. Keeping tabs on high historical refusal ratios is highly correlated with default trends."

        return f"**Analytical Insight**: Query executed successfully, yielding a structured profile with headers: {', '.join(columns)}. The returned data highlights key underlying correlations within the credit database, suggesting strong separation properties for external bureau scores and leverage indices."
