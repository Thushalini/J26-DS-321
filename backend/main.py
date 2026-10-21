"""
Shared backend entry point for the DS Team Behavioral Analytics system.

Each component adds its own router here. Start the server from the backend folder:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs to try every endpoint in the browser.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from buddy.router import router as app_router
from component2.api.router import router as c2_router
from component4.api.router import router as c4_router

try:
    from component1.hook import start_timer
except Exception as e:
    start_timer = None
    print(f"[component1] timer not loaded: {e}")

app = FastAPI(title="DS Team Behavioral Analytics API", version="0.1")


@app.on_event("startup")
def _component1_timer():
    if start_timer:
        start_timer()


# Let the Next.js frontend (local and deployed) call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://j26-ds-321.vercel.app"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(c2_router)
app.include_router(app_router)      # sign-in, teams, chat, blueprint, feedback (Buddy.ds app)
app.include_router(c4_router)       # success prediction, explanation, forecast, recommendations

try:
    from component1.api.router import router as c1_router
    app.include_router(c1_router)
except Exception as e:
    print(f"[component1] blueprint routes not loaded: {e}")

# Other components add theirs here, e.g.:
from component3.api.router import router as c3_router
app.include_router(c3_router)


@app.get("/")
def root():
    return {"message": "DS Team Behavioral Analytics API - see /docs"}
