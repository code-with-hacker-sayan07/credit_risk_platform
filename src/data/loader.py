import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.logger import setup_logger
from src.utils.config import DB_PATH, SQL_DIR

logger = setup_logger("data_loader")

class DataLoader:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def initialize_db(self, force_reload: bool = False):
        """Initialise database schema and populate with CSV data if available, falling back to mock data."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if tables exist
        try:
            cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='applications';")
            exists = cursor.fetchone()[0] > 0
        except sqlite3.Error:
            exists = False

        if not exists:
            logger.info("Database schema not found. Initialising schema from schema.sql...")
            schema_file = SQL_DIR / "schema.sql"
            if schema_file.exists():
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
                cursor.executescript(schema_sql)
                conn.commit()
                logger.info("Database schema successfully created.")
            else:
                raise FileNotFoundError(f"Schema file not found at {schema_file}")

        # Check if we should populate the database from CSVs
        csv_dir = self.db_path.parent / "Home_credit_dataset_files"
        app_csv = csv_dir / "application_train.csv"
        bureau_csv = csv_dir / "bureau.csv"
        prev_csv = csv_dir / "previous_application.csv"
        
        has_csvs = app_csv.exists() and bureau_csv.exists() and prev_csv.exists()
        
        # Check current count in applications table
        cursor.execute("SELECT COUNT(*) FROM applications;")
        count = cursor.fetchone()[0]
        
        # If actual CSVs exist, and we either have no data, mock data (<= 1500 records), or force_reload is active, load them!
        should_load_csv = has_csvs and (count == 0 or count <= 1500 or force_reload)
        
        if should_load_csv:
            logger.info("Actual Home Credit dataset CSV files found. Clearing tables and loading from CSVs...")
            # Disable foreign key constraints temporarily during bulk imports
            cursor.execute("PRAGMA foreign_keys = OFF;")
            cursor.execute("DELETE FROM applications;")
            cursor.execute("DELETE FROM bureau;")
            cursor.execute("DELETE FROM previous_applications;")
            conn.commit()
            
            try:
                self.load_from_csvs(conn, app_csv, bureau_csv, prev_csv)
                logger.info("Successfully populated database from CSVs.")
            except Exception as e:
                logger.error(f"Error loading from CSVs: {str(e)}. Falling back to mock data...")
                # Verify applications count, if empty fall back to mock data
                cursor.execute("SELECT COUNT(*) FROM applications;")
                if cursor.fetchone()[0] == 0:
                    self.generate_and_insert_mock_data(conn)
            finally:
                cursor.execute("PRAGMA foreign_keys = ON;")
                conn.commit()
        elif count == 0:
            logger.info("CSV files not found and database is empty. Populating with highly realistic mock data...")
            self.generate_and_insert_mock_data(conn)
            logger.info("Successfully populated database with mock data.")
        else:
            logger.info(f"Database already contains {count} records. Skipping data population.")
            
        conn.close()

    def load_from_csvs(self, conn: sqlite3.Connection, app_csv: Path, bureau_csv: Path, prev_csv: Path, limit: int = 15000):
        """Loads and filters Home Credit dataset CSV files into the SQLite database with referential integrity."""
        logger.info(f"Loading up to {limit} applications from {app_csv.name}...")
        
        # Load application_train.csv
        app_cols = [
            "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", 
            "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", 
            "AMT_GOODS_PRICE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", 
            "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE", 
            "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
        ]
        
        df_app = pd.read_csv(app_csv, usecols=app_cols, nrows=limit)
        
        # Rename DAYS_BIRTH to DAYS_BIRTHDAY to match SQLite schema expectations
        df_app = df_app.rename(columns={"DAYS_BIRTH": "DAYS_BIRTHDAY"})
        
        # Align datatypes with SQLite expectations
        df_app["TARGET"] = df_app["TARGET"].fillna(0).astype(int)
        df_app["CNT_CHILDREN"] = df_app["CNT_CHILDREN"].fillna(0).astype(int)
        df_app["CNT_FAM_MEMBERS"] = df_app["CNT_FAM_MEMBERS"].fillna(1).astype(int)
        df_app["REGION_RATING_CLIENT"] = df_app["REGION_RATING_CLIENT"].fillna(2).astype(int)
        df_app["DAYS_BIRTHDAY"] = df_app["DAYS_BIRTHDAY"].fillna(-15000).astype(int)
        df_app["DAYS_EMPLOYED"] = df_app["DAYS_EMPLOYED"].fillna(-2000).astype(int)
        
        # Insert into SQLite
        df_app.to_sql("applications", conn, if_exists="append", index=False)
        logger.info(f"Successfully loaded {len(df_app)} applications.")
        
        # Set of active keys for integrity mapping
        valid_ids = set(df_app["SK_ID_CURR"].tolist())
        
        # Load and filter bureau.csv
        logger.info(f"Loading and filtering bureau records from {bureau_csv.name}...")
        bureau_cols = [
            "SK_ID_BUREAU", "SK_ID_CURR", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "DAYS_CREDIT", 
            "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "AMT_CREDIT_MAX_OVERDUE", 
            "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", 
            "AMT_CREDIT_SUM_OVERDUE", "CREDIT_TYPE", "DAYS_CREDIT_UPDATE"
        ]
        
        bureau_chunks = []
        for chunk in pd.read_csv(bureau_csv, usecols=bureau_cols, chunksize=100000):
            filtered_chunk = chunk[chunk["SK_ID_CURR"].isin(valid_ids)]
            if not filtered_chunk.empty:
                bureau_chunks.append(filtered_chunk)
            if sum(len(c) for c in bureau_chunks) > 50000:
                break
                
        if bureau_chunks:
            df_bureau = pd.concat(bureau_chunks, ignore_index=True)
            df_bureau.to_sql("bureau", conn, if_exists="append", index=False)
            logger.info(f"Successfully loaded {len(df_bureau)} bureau credit records.")
        else:
            logger.warning("No matching bureau records found.")
            
        # Load and filter previous_application.csv
        logger.info(f"Loading and filtering previous applications from {prev_csv.name}...")
        prev_cols = [
            "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_ANNUITY", "AMT_APPLICATION", 
            "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE", "WEEKDAY_APPR_PROCESS_START", 
            "HOUR_APPR_PROCESS_START", "FLAG_LAST_APPL_PER_CONTRACT", "NFLAG_LAST_APPL_IN_DAY", 
            "RATE_DOWN_PAYMENT", "RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED", 
            "NAME_PRODUCT_TYPE", "NAME_CONTRACT_STATUS", "DAYS_DECISION", 
            "NAME_PAYMENT_TYPE", "CODE_REJECT_REASON"
        ]
        
        prev_chunks = []
        for chunk in pd.read_csv(prev_csv, usecols=prev_cols, chunksize=100000):
            filtered_chunk = chunk[chunk["SK_ID_CURR"].isin(valid_ids)]
            if not filtered_chunk.empty:
                prev_chunks.append(filtered_chunk)
            if sum(len(c) for c in prev_chunks) > 50000:
                break
                
        if prev_chunks:
            df_prev = pd.concat(prev_chunks, ignore_index=True)
            df_prev = df_prev.rename(columns={"NAME_PRODUCT_TYPE": "NAME_CASH_PORTFOLIO_LIMIT_EXP"})
            df_prev.to_sql("previous_applications", conn, if_exists="append", index=False)
            logger.info(f"Successfully loaded {len(df_prev)} previous applications.")
        else:
            logger.warning("No matching previous application records found.")

    def generate_and_insert_mock_data(self, conn: sqlite3.Connection, num_records: int = 1500):
        """Generates realistic synthetically correlated credit application datasets."""
        np.random.seed(42)
        
        # 1. Main Applications Table
        ids = np.arange(100001, 100001 + num_records)
        
        # Set target default (approx 8.5% default rate)
        targets = np.random.choice([0, 1], size=num_records, p=[0.915, 0.085])
        
        # Gender and demographics
        genders = np.random.choice(['F', 'M'], size=num_records, p=[0.65, 0.35])
        own_car = np.random.choice(['Y', 'N'], size=num_records, p=[0.34, 0.66])
        own_realty = np.random.choice(['Y', 'N'], size=num_records, p=[0.69, 0.31])
        
        # Children and family size
        cnt_children = np.random.choice([0, 1, 2, 3], size=num_records, p=[0.70, 0.20, 0.08, 0.02])
        cnt_fam_members = cnt_children + np.random.choice([1, 2], size=num_records, p=[0.25, 0.75])
        
        # Financials (Income log-normal, credit log-normal correlated with income)
        incomes = np.random.lognormal(mean=11.9, sigma=0.45, size=num_records)
        # Round income to clean numbers
        incomes = np.round(incomes / 5000) * 5000
        
        credit_ratios = np.random.uniform(2.5, 6.0, size=num_records)
        # Defaults have slightly higher credit ratio
        credit_ratios[targets == 1] += np.random.uniform(0.5, 1.5, size=sum(targets == 1))
        
        credits = incomes * credit_ratios
        credits = np.round(credits / 10000) * 10000
        
        # Annuity is around 5% of credit
        annuities = credits * np.random.uniform(0.04, 0.07, size=num_records)
        annuities = np.round(annuities / 100) * 100
        
        goods_prices = credits * np.random.uniform(0.85, 1.0, size=num_records)
        goods_prices = np.round(goods_prices / 10000) * 10000
        
        income_types = np.random.choice(
            ['Working', 'Commercial associate', 'Pensioner', 'State servant'],
            size=num_records,
            p=[0.53, 0.23, 0.18, 0.06]
        )
        
        education_types = np.random.choice(
            ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary'],
            size=num_records,
            p=[0.72, 0.23, 0.04, 0.01]
        )
        
        family_statuses = np.random.choice(
            ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
            size=num_records,
            p=[0.64, 0.15, 0.10, 0.07, 0.04]
        )
        
        housing_types = np.random.choice(
            ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment'],
            size=num_records,
            p=[0.88, 0.05, 0.04, 0.02, 0.007, 0.003]
        )
        
        # Age and employment (represented in days, negative value)
        ages = -1 * np.random.randint(21 * 365, 65 * 365, size=num_records)
        
        # Employment: Pensioners have 365243 (Kaggle flag for unemployed/retired)
        employed = []
        for i in range(num_records):
            if income_types[i] == 'Pensioner':
                employed.append(365243)
            else:
                # Employed days cannot exceed age + 18 years
                max_emp = -ages[i] - 18 * 365
                emp_days = -np.random.randint(30, max(31, max_emp))
                employed.append(emp_days)
        employed = np.array(employed)
        
        occupations = [
            'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers', 
            'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff',
            'Cooking staff', 'Cleaning staff'
        ]
        occupation_p = [0.28, 0.15, 0.14, 0.11, 0.08, 0.05, 0.05, 0.05, 0.04, 0.03, 0.02]
        
        # Assign occupation based on education and pensioner status
        assigned_occupations = []
        for i in range(num_records):
            if income_types[i] == 'Pensioner' or employed[i] == 365243:
                assigned_occupations.append(None)
            else:
                assigned_occupations.append(np.random.choice(occupations, p=occupation_p))
                
        region_rating = np.random.choice([1, 2, 3], size=num_records, p=[0.11, 0.74, 0.15])
        
        # External Sources (highly predictive!)
        # Non-defaults (0) have higher external scores, defaults (1) have lower external scores
        ext_1 = np.random.beta(a=5, b=5, size=num_records) # centered at 0.5
        ext_2 = np.zeros(num_records)
        ext_3 = np.zeros(num_records)
        
        for i in range(num_records):
            if targets[i] == 0:
                ext_2[i] = np.random.beta(a=6, b=3) # shifted higher (avg ~0.66)
                ext_3[i] = np.random.beta(a=5, b=3) # shifted higher (avg ~0.62)
            else:
                ext_2[i] = np.random.beta(a=3, b=6) # shifted lower (avg ~0.33)
                ext_3[i] = np.random.beta(a=2, b=5) # shifted lower (avg ~0.28)
                
        # Insert applications
        app_data = []
        for i in range(num_records):
            app_data.append((
                int(ids[i]), int(targets[i]), str(genders[i]), str(own_car[i]), str(own_realty[i]),
                int(cnt_children[i]), float(incomes[i]), float(credits[i]), float(annuities[i]), float(goods_prices[i]),
                str(income_types[i]), str(education_types[i]), str(family_statuses[i]), str(housing_types[i]),
                int(ages[i]), int(employed[i]), assigned_occupations[i], int(cnt_fam_members[i]),
                int(region_rating[i]), float(ext_1[i]), float(ext_2[i]), float(ext_3[i])
            ))
            
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO applications (
                SK_ID_CURR, TARGET, CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY,
                CNT_CHILDREN, AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
                NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE,
                DAYS_BIRTHDAY, DAYS_EMPLOYED, OCCUPATION_TYPE, CNT_FAM_MEMBERS,
                REGION_RATING_CLIENT, EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, app_data)
        
        # 2. Populate Credit Bureau Table
        # Generate bureau records for approx 80% of applicants
        bureau_data = []
        bureau_id = 5000001
        
        for app_id in ids[np.random.rand(num_records) < 0.8]:
            num_bureau_credits = np.random.randint(1, 8)
            for _ in range(num_bureau_credits):
                active = np.random.choice(['Closed', 'Active', 'Sold'], p=[0.70, 0.29, 0.01])
                days_credit = -np.random.randint(30, 2000)
                overdue = np.random.choice([0, np.random.randint(10, 500)], p=[0.97, 0.03])
                sum_credit = np.random.lognormal(11.2, 0.8)
                sum_credit = np.round(sum_credit / 5000) * 5000
                debt = 0.0 if active == 'Closed' else sum_credit * np.random.uniform(0.1, 0.8)
                limit = 0.0 if active == 'Closed' else np.random.choice([0.0, 50000.0], p=[0.8, 0.2])
                overdue_amt = 0.0 if overdue == 0 else overdue * np.random.uniform(10, 100)
                
                credit_type = np.random.choice(['Consumer credit', 'Credit card', 'Car loan'], p=[0.75, 0.20, 0.05])
                
                bureau_data.append((
                    bureau_id, int(app_id), active, 'currency 1', int(days_credit), int(overdue),
                    int(days_credit + 365), int(days_credit + 180) if active == 'Closed' else None,
                    0.0, 0, float(sum_credit), float(debt), float(limit), float(overdue_amt),
                    credit_type, int(days_credit + 10)
                ))
                bureau_id += 1
                
        cursor.executemany("""
            INSERT INTO bureau (
                SK_ID_BUREAU, SK_ID_CURR, CREDIT_ACTIVE, CREDIT_CURRENCY, DAYS_CREDIT,
                CREDIT_DAY_OVERDUE, DAYS_CREDIT_ENDDATE, DAYS_ENDDATE_FACT, AMT_CREDIT_MAX_OVERDUE,
                CNT_CREDIT_PROLONG, AMT_CREDIT_SUM, AMT_CREDIT_SUM_DEBT, AMT_CREDIT_SUM_LIMIT,
                AMT_CREDIT_SUM_OVERDUE, CREDIT_TYPE, DAYS_CREDIT_UPDATE
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, bureau_data)
        
        # 3. Populate Previous Applications Table
        # Generate previous application records for approx 85% of applicants
        prev_data = []
        prev_id = 2000001
        
        for app_id in ids[np.random.rand(num_records) < 0.85]:
            num_prev = np.random.randint(1, 5)
            for _ in range(num_prev):
                status = np.random.choice(['Approved', 'Refused', 'Canceled'], p=[0.72, 0.18, 0.10])
                days_decision = -np.random.randint(30, 1500)
                
                # Fetch income of applicant to scale previous credits
                app_idx = np.where(ids == app_id)[0][0]
                inc = incomes[app_idx]
                
                amt_app = inc * np.random.uniform(0.5, 3.0)
                amt_app = np.round(amt_app / 5000) * 5000
                
                amt_credit = amt_app if status == 'Approved' else 0.0
                annuity = amt_credit * np.random.uniform(0.05, 0.1) if status == 'Approved' else 0.0
                down_pmt = 0.0 if status == 'Refused' else amt_app * np.random.choice([0.0, 0.1, 0.2], p=[0.7, 0.2, 0.1])
                
                prev_data.append((
                    prev_id, int(app_id), 'Cash loans', float(annuity), float(amt_app), float(amt_credit),
                    float(down_pmt), float(amt_app), 'MONDAY', 12, 'Y', 1,
                    0.0, 0.0, 0.0, 'X-sell', status, int(days_decision), 'Cash through the bank',
                    'LIMIT' if status == 'Refused' else 'XAP'
                ))
                prev_id += 1
                
        cursor.executemany("""
            INSERT INTO previous_applications (
                SK_ID_PREV, SK_ID_CURR, NAME_CONTRACT_TYPE, AMT_ANNUITY, AMT_APPLICATION,
                AMT_CREDIT, AMT_DOWN_PAYMENT, AMT_GOODS_PRICE, WEEKDAY_APPR_PROCESS_START,
                HOUR_APPR_PROCESS_START, FLAG_LAST_APPL_PER_CONTRACT, NFLAG_LAST_APPL_IN_DAY,
                RATE_DOWN_PAYMENT, RATE_INTEREST_PRIMARY, RATE_INTEREST_PRIVILEGED,
                NAME_CASH_PORTFOLIO_LIMIT_EXP, NAME_CONTRACT_STATUS, DAYS_DECISION,
                NAME_PAYMENT_TYPE, CODE_REJECT_REASON
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, prev_data)
        
        conn.commit()

    def load_applications(self) -> pd.DataFrame:
        """Loads and returns all records from the applications table as a pandas DataFrame."""
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT * FROM applications;", conn)
        conn.close()
        return df
        
    def load_bureau(self) -> pd.DataFrame:
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT * FROM bureau;", conn)
        conn.close()
        return df

    def load_previous(self) -> pd.DataFrame:
        conn = self.get_connection()
        df = pd.read_sql_query("SELECT * FROM previous_applications;", conn)
        conn.close()
        return df
