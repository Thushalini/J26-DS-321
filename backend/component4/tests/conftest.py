"""Tests always use a throwaway local database, never the shared one in backend/.env."""
import os

os.environ["DATABASE_URL"] = ""
