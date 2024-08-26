import os
import pandas as pd


class SymbolLookup:
    def __init__(self):
        # Define the path to the CSV file in the backend/miscellaneous directory
        self.csv_path = 'names.csv'

        # Load the CSV and create a mapping between Security Code and Security Id
        self.security_code_to_tradingsymbol = self.load_symbols()

    def load_symbols(self):
        """
        Load the names.csv file and create a dictionary mapping security codes to tradingsymbols.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        try:
            # Load the CSV into a pandas dataframe
            df = pd.read_csv(self.csv_path)

            # Create a mapping between Security Code and Security Id (tradingsymbol)
            security_code_to_tradingsymbol = pd.Series(df['Security Id'].values, index=df['Security Code']).to_dict()

            return security_code_to_tradingsymbol

        except Exception as e:
            raise Exception(f"Error loading or processing the CSV file: {str(e)}")

    def get_tradingsymbol(self, security_code):
        """
        Get the tradingsymbol corresponding to a given security code.
        """
        tradingsymbol = self.security_code_to_tradingsymbol.get(security_code)

        if tradingsymbol:
            return tradingsymbol
        else:
            raise ValueError(f"Tradingsymbol not found for Security Code: {security_code}")


# Example usage:
#lookup = SymbolLookup()
#print(lookup.get_tradingsymbol(500325))
