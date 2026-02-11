from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .exports import parity_to_xlsx
from .indexer import load_runs_index
from .insights import build_insights
from .metrics import build_monthly_parity, compute_run_dashboard
from .ops import manual_accept
from .readers import preview_file, read_zip_json_preview, safe_resolve, safe_tail, safe_read_csv, safe_read_json
from .schema import resolve_schema
from .settings import SETTINGS

app = FastAPI(title="Parser Analytics UI Local")
BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def latest_run_id() -> str | None:
    runs = load_runs_index().get("runs", [])
    return runs[0]["run_id"] if runs else None


@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse("/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, run_id: str | None = None):
    run_id = run_id or latest_run_id()
    return templates.TemplateResponse("dashboard.html", {"request": request, "run_id": run_id})


@app.get("/parity", response_class=HTMLResponse)
def parity(request: Request, run_id: str | None = None):
    return templates.TemplateResponse("parity.html", {"request": request, "run_id": run_id or latest_run_id()})


@app.get("/projects", response_class=HTMLResponse)
def projects(request: Request, run_id: str | None = None):
    return templates.TemplateResponse("projects.html", {"request": request, "run_id": run_id or latest_run_id()})


@app.get("/projects/{project_key}", response_class=HTMLResponse)
def project_detail(request: Request, project_key: str, run_id: str | None = None):
    return templates.TemplateResponse("project_detail.html", {"request": request, "project_key": project_key, "run_id": run_id or latest_run_id()})


@app.get("/ops", response_class=HTMLResponse)
def ops_page(request: Request):
    return templates.TemplateResponse("ops.html", {"request": request, "index": load_runs_index()})


@app.get("/ops/runs/{run_id}", response_class=HTMLResponse)
def run_details(request: Request, run_id: str):
    return templates.TemplateResponse("run_detail.html", {"request": request, "run_id": run_id})


@app.get("/ops/quarantine/{run_id}", response_class=HTMLResponse)
def quarantine_page(request: Request, run_id: str):
    qdir = SETTINGS.launcher_data_root / "quarantine" / run_id
    files = sorted([p.name for p in qdir.glob("*.csv")]) if qdir.exists() else []
    return templates.TemplateResponse("quarantine.html", {"request": request, "run_id": run_id, "files": files})


@app.get("/ops/logs/{run_id}", response_class=HTMLResponse)
def logs_page(request: Request, run_id: str):
    log = SETTINGS.launcher_root / "logs" / run_id / "launcher.log"
    return templates.TemplateResponse("logs.html", {"request": request, "run_id": run_id, "tail": safe_tail(log)})


@app.get("/api/index")
def api_index():
    return load_runs_index()


@app.get("/api/dashboard")
def api_dashboard(run_id: str | None = None):
    run_id = run_id or latest_run_id()
    if not run_id:
        return {"kpi": {}, "charts": {}, "insights": []}
    run_dir = SETTINGS.launcher_data_root / "published_snapshots" / run_id
    payload = compute_run_dashboard(str(run_dir), "|".join(SETTINGS.ignore_statuses), str(SETTINGS.synonyms_override))
    payload["kpi"]["quarantine_count"] = len(list((SETTINGS.launcher_data_root / "quarantine" / run_id).glob("*.csv")))
    payload["insights"] = build_insights(payload["kpi"], payload.get("charts", {}).get("sales", []))
    return payload


@app.get("/api/parity")
def api_parity(run_id: str | None = None):
    run_id = run_id or latest_run_id()
    if not run_id:
        return {"rows": []}
    run_dir = SETTINGS.launcher_data_root / "published_snapshots" / run_id
    files = sorted(run_dir.glob("*.csv"))
    frames = [safe_read_csv(f) for f in files[:6]]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    schema = resolve_schema(list(df.columns))
    parity = build_monthly_parity(df, schema)
    rows = parity.to_dict("records") if not parity.empty else []
    return {"rows": rows}


@app.get("/api/project/{project_key}")
def api_project(project_key: str, run_id: str | None = None):
    data = api_parity(run_id)
    rows = [r for r in data.get("rows", []) if str(r.get("complex_name") or r.get("project_name") or "").lower() == project_key.lower()]
    return {"project_key": project_key, "rows": rows}


@app.get("/api/ops/run/{run_id}")
def api_ops_run(run_id: str):
    rep = SETTINGS.launcher_data_root / "reports" / run_id
    result_files = list(rep.glob("**/result.json"))
    reason_files = list(rep.glob("**/reason.json"))
    bundles = list((rep / "support_bundles").glob("*.zip")) if (rep / "support_bundles").exists() else []
    return {
        "run_id": run_id,
        "result_files": [str(p.relative_to(SETTINGS.launcher_data_root)) for p in result_files],
        "reason_files": [str(p.relative_to(SETTINGS.launcher_data_root)) for p in reason_files],
        "bundles": [str(p.relative_to(SETTINGS.launcher_data_root)) for p in bundles],
    }


@app.get("/api/file")
def api_file(path: str = Query(...)):
    full = safe_resolve(SETTINGS.launcher_data_root, path)
    if not full.exists():
        raise HTTPException(404, "Not found")
    if full.suffix.lower() == ".zip":
        return {"type": "zip-json-preview", "content": read_zip_json_preview(full)}
    return preview_file(full)


@app.get("/api/download")
def api_download(path: str = Query(...)):
    full = safe_resolve(SETTINGS.launcher_data_root, path)
    if not full.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(full), filename=full.name)


@app.post("/api/manual_accept")
def api_manual_accept(payload: dict):
    rec = manual_accept(
        run_id=str(payload.get("run_id", "")),
        snapshot_name=str(payload.get("snapshot_name", "")),
        reason=str(payload.get("reason", "")).strip(),
        operator=str(payload.get("operator", "ui_local")),
    )
    return {"ok": True, "audit": rec}


@app.get("/api/export/parity.xlsx")
def api_export_parity(run_id: str | None = None):
    rows = api_parity(run_id).get("rows", [])
    data = parity_to_xlsx(rows)
    return Response(content=data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=parity_matrix.xlsx"})
