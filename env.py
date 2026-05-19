from dotenv import load_dotenv
import os

# print("All environment variables:", os.environ)
# load_dotenv()  # Loads variables from .env into os.environ

# print(os.getenv("GROQ_BASE_URL"))

load_dotenv()
print("GROQ_BASE_URL:", os.getenv("GROQ_BASE_URL"))