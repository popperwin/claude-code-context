import pandas as pd
from typing import List, Dict

class DataProcessor:
    def __init__(self, config: Dict[str, str]):
        self.config = config
    
    def process_data(self, data: List[Dict]) -> pd.DataFrame:
        """Process raw data into structured format"""
        return pd.DataFrame(data)
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate processed data"""
        return not df.empty
