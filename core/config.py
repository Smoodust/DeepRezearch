import os

from dotenv import load_dotenv


load_dotenv()

MODEL_URL = os.getenv("MODEL_URL", "")