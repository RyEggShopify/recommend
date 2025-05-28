from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

class ShopifyBigQueryClient:
    def __init__(self, project_id="shopify-cloud-apps", credentials_path=None):
        """Initialize BigQuery client for Shopify data"""
        self.project_id = project_id
        
        if credentials_path:
            # Use service account
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = bigquery.Client(credentials=credentials, project=project_id)
        else:
            # Use default credentials (from gcloud auth or environment)
            self.client = bigquery.Client(project=project_id)
    
    def query_products(self, limit=5_000) -> pd.DataFrame:
        """Query products from the merchandising dataset"""
        query = f"""
        SELECT *
        FROM `shopify-dw.merchandising.products`
        LIMIT {limit}
        """
        
        try:
            # Execute query and return as DataFrame
            df = self.client.query(query).to_dataframe()
            return df
        except Exception as e:
            print(f"Error querying products: {e}")
            return None