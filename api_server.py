#!/usr/bin/env python3
"""
DigitalOcean deployment: FastAPI service for Aesthetic RAG Search.

Provides JSON APIs for:
1) Regions
2) Sub-zones for a region
3) Common concerns for (region, sub-zone)  [internet -> fallback DB]
4) Recommendations (region, sub-zone, issue_text, preference)

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
from typing import List, Literal, Optional

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

app = FastAPI(title=APP_TITLE, version="1.0.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


# ---------------------------- schemas ----------------------------

Preference = Literal["Surgical Treatment", "Non-surgical Treatment", "Both"]


class RegionsResponse(BaseModel):
    regions: List[str]


class SubZonesResponse(BaseModel):
    region: str
    sub_zones: List[str]


class CommonConcernsResponse(BaseModel):
    region: str
    sub_zone: str
    concerns: List[str]


class RecommendRequest(BaseModel):
    region: str = Field(..., min_length=1)
    sub_zone: str = Field(..., min_length=1)
    issue_text: str = Field(..., min_length=1, description="User problem statement (free text)")
    preference: Preference = "Both"
    retrieval_k: int = Field(12, ge=3, le=50)
    final_k: int = Field(5, ge=1, le=10)


class RecommendResponse(BaseModel):
    status: Literal["blocked", "mismatch", "ok"]
    message: str
    recommended_procedures: list = []
    suggested_region_subzones: list = []
    debug: dict = {}


# ---------------------------- endpoints ----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/regions", response_model=RegionsResponse)
def get_regions():
    return {"regions": rag.get_regions()}


@app.get("/sub-zones", response_model=SubZonesResponse)
def get_sub_zones(region: str):
    region = (region or "").strip()
    if not region:
        raise HTTPException(status_code=400, detail="region is required")
    return {"region": region, "sub_zones": rag.get_sub_zones(region)}


@app.get("/common-concerns", response_model=CommonConcernsResponse)
def get_common_concerns(region: str, sub_zone: str, n: int = 4):
    region = (region or "").strip()
    sub_zone = (sub_zone or "").strip()
    if not region or not sub_zone:
        raise HTTPException(status_code=400, detail="region and sub_zone are required")
    n = max(1, min(int(n), 10))
    concerns = rag.get_common_concerns(region, sub_zone, n=n)
    return {"region": region, "sub_zone": sub_zone, "concerns": concerns}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    # Map preference into your pipeline's expected string
    out = rag.recommend(
        region=req.region,
        sub_zone=req.sub_zone,
        type_choice=req.preference,
        issue_text=req.issue_text,
        retrieval_k=req.retrieval_k,
        final_k=req.final_k,
    )
    return out
