#!/usr/bin/env python3
"""
DigitalOcean deployment: FastAPI service for Aesthetic RAG Search.

3 APIs:
1) Get common concerns for a selected sub-zone
2) Validate and return user selections (concerns + procedure type)
3) Final search to get recommended procedures

Run locally:
  pip install -r requirements.txt
  uvicorn api_server:app --host 0.0.0.0 --port 8000

Env:
  DB_XLSX=database.xlsx
  EMB_CACHE=treatment_embeddings.pkl
  LOCAL_LLM_PROVIDER=ollama|transformers
  OLLAMA_HOST=http://localhost:11434
  OLLAMA_MODEL=llama3.2:1b
"""

from __future__ import annotations

import os
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_treatment_app import RAGTreatmentSearchApp


# ---------------------------- app init ----------------------------

APP_TITLE = "Aesthetic RAG Search API"

rag = RAGTreatmentSearchApp(
    excel_path=os.getenv("DB_XLSX", "database.xlsx"),
    embeddings_cache_path=os.getenv("EMB_CACHE", "treatment_embeddings.pkl"),
)

app = FastAPI(title=APP_TITLE, version="2.0.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------- schemas ----------------------------

TreatmentPreference = Literal["Surgical", "Non-Surgical", "Both"]


class CommonConcernsRequest(BaseModel):
    sub_zone: str = Field(..., min_length=1, description="Selected sub-zone (e.g., 'Eyes', 'Nose')")


class CommonConcernsResponse(BaseModel):
    sub_zone: str
    common_concerns: List[str] = Field(description="Common concerns for this sub-zone from database")


class UserSelectionRequest(BaseModel):
    sub_zone: str = Field(..., min_length=1, description="Selected sub-zone")
    concerns: List[str] = Field(..., min_items=1, description="User selected concerns (one or more)")
    treatment_preference: TreatmentPreference = Field(..., description="User selected treatment type")


class UserSelectionResponse(BaseModel):
    sub_zone: str
    concerns: List[str]
    treatment_preference: str


class SearchRequest(BaseModel):
    sub_zone: str = Field(..., min_length=1, description="Sub-zone from API 2")
    concerns: List[str] = Field(..., min_items=1, description="Concerns array from API 2")
    treatment_preference: TreatmentPreference = Field(..., description="Treatment type from API 2")
    retrieval_k: int = Field(12, ge=3, le=50, description="Number of candidates to retrieve")
    final_k: int = Field(5, ge=1, le=10, description="Number of final recommendations")


class SearchResponse(BaseModel):
    mismatch: bool
    notice: str = ""
    recommended_procedures: List[str] = Field(default_factory=list, description="List of procedure names")
    suggested_region_subzones: List[dict] = Field(default_factory=list)


# ---------------------------- endpoints ----------------------------

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "aesthetic-rag-api", "version": "2.0.0"}


@app.post("/common_concerns", response_model=CommonConcernsResponse)
def get_common_concerns(req: CommonConcernsRequest):
    """
    API 1: Get common concerns for a selected sub-zone from database.
    
    Args:
        sub_zone: Selected sub-zone (e.g., 'Eyes', 'Nose', 'Cheeks')
    
    Returns:
        sub_zone: The requested sub-zone
        common_concerns: List of common concerns from database for this sub-zone
    """
    sub_zone = req.sub_zone.strip()
    if not sub_zone:
        raise HTTPException(status_code=400, detail="sub_zone is required")
    
    # Get common concerns for this sub-zone from database
    concerns = rag.get_concerns_for_subzone(sub_zone)
    
    if not concerns:
        raise HTTPException(status_code=404, detail=f"No concerns found for sub-zone: {sub_zone}")
    
    return {
        "sub_zone": sub_zone,
        "common_concerns": concerns
    }


@app.post("/user_selection", response_model=UserSelectionResponse)
def validate_user_selection(req: UserSelectionRequest):
    """
    API 2: Validate and return user selections (concerns + procedure type).
    
    User selects:
    - One or more concerns from API 1 response
    - One procedure type (Surgical, Non-Surgical, or Both)
    
    This API validates the input and returns it for API 3.
    
    Args:
        sub_zone: Selected sub-zone
        concerns: Array of selected concerns (min 1)
        treatment_preference: Selected treatment type
    
    Returns:
        sub_zone: The selected sub-zone
        concerns: The selected concerns array
        treatment_preference: The selected treatment type
    """
    sub_zone = req.sub_zone.strip()
    concerns = [c.strip() for c in req.concerns if c.strip()]
    
    if not sub_zone:
        raise HTTPException(status_code=400, detail="sub_zone is required")
    
    if not concerns:
        raise HTTPException(status_code=400, detail="At least one concern must be selected")
    
    # Validate that sub-zone exists
    region = rag.get_region_from_subzone(sub_zone)
    if not region:
        raise HTTPException(status_code=404, detail=f"Invalid sub-zone: {sub_zone}")
    
    return {
        "sub_zone": sub_zone,
        "concerns": concerns,
        "treatment_preference": req.treatment_preference
    }


@app.post("/search", response_model=SearchResponse)
def search_procedures(req: SearchRequest):
    """
    API 3: Final search to get recommended procedures.
    
    Takes the output from API 2 (user selections) and performs the search.
    
    Args:
        sub_zone: From API 2
        concerns: From API 2
        treatment_preference: From API 2
        retrieval_k: Optional, number of candidates
        final_k: Optional, number of final results
    
    Returns:
        mismatch: Whether mismatch was detected
        notice: Message if mismatch
        recommended_procedures: List of procedure names (empty if mismatch)
        suggested_region_subzones: Suggestions if mismatch
    """
    sub_zone = req.sub_zone.strip()
    concerns = [c.strip() for c in req.concerns if c.strip()]
    
    if not sub_zone:
        raise HTTPException(status_code=400, detail="sub_zone is required")
    
    if not concerns:
        raise HTTPException(status_code=400, detail="At least one concern is required")
    
    # Get region from sub-zone
    region = rag.get_region_from_subzone(sub_zone)
    
    if not region:
        raise HTTPException(status_code=404, detail=f"No region found for sub-zone: {sub_zone}")
    
    # Perform search
    result = rag.search_by_concerns(
        region=region,
        sub_zone=sub_zone,
        type_choice=req.treatment_preference,
        concerns=concerns,
        retrieval_k=req.retrieval_k,
        final_k=req.final_k,
    )
    
    return result
