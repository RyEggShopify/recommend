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

    def query_products(self, limit=5_000) -> pd.DataFrame | None:
        """Query restricted products with GMV data from the global product set"""
        query = f"""
        WITH top_restricted_products AS (         
        SELECT product_id,
                gmv_usd_60d,
                category,
                title,
                description,
        FROM `shopify-dw.mart_search.global_product_set`
        WHERE is_product_restricted = TRUE
            AND gmv_usd_60d IS NOT NULL
            AND gmv_usd_60d > 4000
        LIMIT {limit}
        )
        SELECT 
        *
        FROM top_restricted_products
        """
        
        try:
            # Execute query and return as DataFrame
            df = self.client.query(query).to_dataframe()
            return df
        except Exception as e:
            print(f"Error querying products: {e}")
            return None