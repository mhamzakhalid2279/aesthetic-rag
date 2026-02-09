#!/usr/bin/env python3
"""
DigitalOcean deployment: FastAPI service for Aesthetic RAG Search.

Simplified to 2 APIs:
1) Get sub-zones for a region
2) Search with concerns array + procedure type

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


class SubZonesRequest(BaseModel):
    region: str = Field(..., min_length=1, description="Selected region (e.g., 'Face', 'Body')")


class SubZonesResponse(BaseModel):
    region: str
    sub_zones: List[str]
    common_concerns: List[str] = Field(default_factory=list, description="All common concerns for this region")


class SearchRequest(BaseModel):
    sub_zone: str = Field(..., min_length=1, description="Selected sub-zone (e.g., 'Eyes', 'Nose')")
    concerns: List[str] = Field(..., min_items=1, description="List of selected concerns")
    treatment_preference: TreatmentPreference = Field("Both", description="Treatment type preference")
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


@app.post("/subzones", response_model=SubZonesResponse)
def get_subzones(req: SubZonesRequest):
    """
    Get sub-zones and common concerns for a selected region.
    
    Returns:
    - sub_zones: List of available sub-zones for the region
    - common_concerns: All common concerns from database for this region
    """
    region = req.region.strip()
    if not region:
        raise HTTPException(status_code=400, detail="region is required")
    
    # Get sub-zones for this region
    sub_zones = rag.get_sub_zones(region)
    
    if not sub_zones:
        raise HTTPException(status_code=404, detail=f"No sub-zones found for region: {region}")
    
    # Get all common concerns for this region (from all sub-zones)
    all_concerns = rag.get_all_concerns_for_region(region)
    
    return {
        "region": region,
        "sub_zones": sub_zones,
        "common_concerns": all_concerns
    }


@app.post("/search", response_model=SearchResponse)
def search_procedures(req: SearchRequest):
    """
    Search for treatment procedures based on:
    - sub_zone: Selected sub-zone
    - concerns: Array of selected concerns
    - treatment_preference: Surgical, Non-Surgical, or Both
    
    Returns:
    - If mismatch: suggested region/sub-zones
    - If match: list of recommended procedure names only
    """
    sub_zone = req.sub_zone.strip()
    concerns = [c.strip() for c in req.concerns if c.strip()]
    
    if not sub_zone:
        raise HTTPException(status_code=400, detail="sub_zone is required")
    
    if not concerns:
        raise HTTPException(status_code=400, detail="At least one concern is required")
    
    # Combine concerns into issue text
    issue_text = ", ".join(concerns)
    
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
