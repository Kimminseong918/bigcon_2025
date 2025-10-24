# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
ART = BASE / "artifacts" / "lgbm"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)

KINDS = ["delta"]  
TAGS = ["3m", "6m"]
RNG = 42


GROUP_RULES = {
    "customer": ["CustomerRiskIndex_3m", "CustomerRiskIndex_6m"],
    "industry": ["IndustryRiskIndex_3m", "IndustryRiskIndex_6m"],
    "market":   ["market_close_rate", "market_density"],
    "sales": [
        "sales_index", "sales_index_district", "sales_index_category", "sales_index_dc",
        "salees_index_category"  
    ],
    "macro": [
        "CPI","CPI_yoy","PPI","PPI_yoy","CPI_shock3","PPI_shock3","rent_index",
        "CPIyoyxIndu_3m","CPIyoyxIndu_6m","PPIyoyxIndu_3m","PPIyoyxIndu_6m",
        "RentxMarket","m_sin","m_cos"
    ],
}
DERIV_SUFFIX = ["_z", "_ma3", "_chg3", "_ma6", "_chg6"]


def _load_df():
 
    cand = [
        OUT / "merged_indices_monthly.parquet",   
        OUT / "merged_indices_monthly_preview.csv",  
        OUT / "merged_indices_monthly.csv",     
        OUT / "merged_indices.csv",             
    ]
    p = next((x for x in cand if x.exists()), None)
    if p is None:
        raise FileNotFoundError(
            "merged_indices_monthly.(parquet/csv)를 찾지 못했습니다. "
            "먼저 `python -m src.pipeline`을 성공적으로 실행했는지 확인하세요."
        )

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)


    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")\
                          .dt.to_period("M").dt.to_timestamp()
    if "store_id" in df.columns:
        df["store_id"] = df["store_id"].astype(str)

    if "year" not in df.columns:
        df["year"] = df["month"].dt.year
    return df


def _target(kind: str, tag: str) -> str:
    return f"y_risk{'_trend' if kind == 'trend' else ''}_{tag}"

def _select_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for klist in GROUP_RULES.values():
        for p in klist:
            for suf in ["", "_z", "_ma3", "_chg3"]:
                c = f"{p}{suf}" if suf else p
                if c in df.columns:
                    cols.append(c)

    drop_keys = {"store_id", "district", "category", "month", "year"}
    cols = [c for c in cols if c not in drop_keys]

    tr_mask = (df["month"] >= pd.Timestamp(2023, 1, 1)) & (df["month"] <= pd.Timestamp(2024, 6, 1))

    nun = df.loc[tr_mask, cols].apply(lambda s: s.dropna().nunique())
    consts = nun[nun <= 1].index.tolist()

    sales_cols = [c for c in cols if c.startswith(("sales_index","sales_index_district","sales_index_category","sales_index_dc","salees_index_category"))]
    if sales_cols:
        cov = df.loc[tr_mask, sales_cols].notna().mean().round(3)
        print("train-window sales coverage:", cov.to_dict())

    if consts:
        print(f"drop constant-in-train(valid): {consts}")
    cols = [c for c in cols if c not in consts]
    return cols

def _map_group(col: str) -> str:
    base = col
    for s in DERIV_SUFFIX:
        if base.endswith(s):
            base = base[:-len(s)]
    for g, keys in GROUP_RULES.items():
        if any(base.startswith(k) for k in keys):
            return g
    return "macro"

def _valid_mask(meta: pd.DataFrame, tag: str) -> np.ndarray:
    last_month = meta["month"].max()
    horizon = 3 if tag == "3m" else 6
    if pd.isna(last_month):
        return np.zeros(len(meta), dtype=bool)
    valid_end = (last_month - pd.DateOffset(months=horizon)).to_period("M").to_timestamp()
    valid_start = (valid_end - pd.DateOffset(months=5)).to_period("M").to_timestamp()
    vm = (meta["month"] >= valid_start) & (meta["month"] <= valid_end)
    if vm.sum() == 0:
        order = meta["month"].sort_values().index
        cut = max(1, int(len(order) * 0.9))
        vm = meta.index.isin(order[cut:])
        print("no valid rows for chosen window → fallback to last 10% time-based holdout")
    return vm.values

def _train_one(df: pd.DataFrame, kind: str, tag: str):
    target = _target(kind, tag)
    if target not in df.columns:
        raise KeyError(f"{target} not found in dataset")

    feat = _select_features(df)
    if len(feat) == 0:
        raise RuntimeError("No features selected. Check GROUP_RULES and data coverage.")

    meta = df[["store_id", "district", "month"]].copy()
    X = df.reindex(columns=feat).astype(np.float32).fillna(0.0)
    y = pd.to_numeric(df[target], errors="coerce")
    m = y.notna()
    X, y, meta = X[m], y[m], meta[m]

    vm = _valid_mask(meta, tag)
    Xtr, ytr = X[~vm], y[~vm]
    Xva, yva = X[vm], y[vm]
    print(f"Target={target} | rows={len(X)} | train={len(Xtr)} valid={len(Xva)} | feats={len(feat)}")

    pos = int((ytr == 1).sum()); neg = int((ytr == 0).sum())
    scale_pos_weight = (neg / max(1, pos)) if pos > 0 else 1.0

    params = dict(
        objective="binary", metric=["auc","binary_logloss"],
        learning_rate=0.045, num_leaves=48,
        feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=2,
        min_data_in_leaf=50, reg_lambda=3.0, reg_alpha=2.0,
        random_state=RNG, n_estimators=4000, scale_pos_weight=scale_pos_weight,
    )
    model = lgb.LGBMClassifier(**params)

    if len(Xva) == 0:
        print("training without validation (no valid rows).")
        model.fit(Xtr, ytr)
        refX, refy = Xtr, ytr
        auc = float("nan")
    else:
        model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)],
                  eval_names=["train","valid"],
                  callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(100)])
        refX, refy = Xva, yva
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(refy, model.predict_proba(refX)[:,1]))
        except Exception:
            auc = float("nan")

    joblib.dump(model, ART / f"risk_lgbm_{kind}_{tag}.joblib", compress=3)

    (ART / f"feature_list_{kind}_{tag}.json").write_text(
        json.dumps(list(feat), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    proba = model.predict_proba(refX)[:, 1]
    sweep = []
    for thr in np.linspace(0.1, 0.9, 81):
        p = (proba >= thr).astype(int)
        tp = int(((p == 1) & (refy == 1)).sum())
        fp = int(((p == 1) & (refy == 0)).sum())
        fn = int(((p == 0) & (refy == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        sweep.append((thr, prec, rec, f1))
    best_idx = int(np.argmax([r[3] for r in sweep]))
    best_thr, _, _, best_f1 = sweep[best_idx]
    (ART / f"threshold_summary_{kind}_{tag}.json").write_text(
        json.dumps({"best_thr": float(best_thr), "f1": float(best_f1), "auc": auc},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    try:
        import shap
        explainer = shap.TreeExplainer(model.booster_)
        sv = explainer.shap_values(refX)
        imp = (np.abs(sv[1]).mean(axis=0) if isinstance(sv, list) else np.abs(sv).mean(axis=0))
        imp = pd.Series(imp, index=list(feat))
    except Exception:
        gain = model.booster_.feature_importance(importance_type="gain")
        imp = pd.Series(gain, index=list(feat)).astype(float)
        imp = (imp + 1e-9) / (imp.sum() + 1e-9)

    gimp = imp.groupby(imp.index.map(_map_group)).sum()
    order = ["customer","market","industry","sales","macro"]
    gimp = gimp.reindex(order).fillna(0.0)
    weights = (gimp / (gimp.sum() + 1e-9)).to_dict()
    (ART / f"group_weights_{kind}_{tag}.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("group weights [{} {}]: ".format(kind, tag) +
          " ".join(f"{k}={weights.get(k,0.0):.3f}" for k in order))

    sub = df[(df["month"] >= pd.Timestamp(2023, 1, 1)) & (df["month"] <= pd.Timestamp(2024, 12, 1))]
    scale_cols = {
        "customer_3m":"CustomerRiskIndex_3m","customer_6m":"CustomerRiskIndex_6m",
        "industry_3m":"IndustryRiskIndex_3m","industry_6m":"IndustryRiskIndex_6m",
        "market_close_rate":"market_close_rate","market_density":"market_density",
        "sales_index":"sales_index","CPI_yoy":"CPI_yoy","PPI_yoy":"PPI_yoy","rent_index":"rent_index",
    }
    scalers = {}
    for k, c in scale_cols.items():
        if c in sub.columns:
            s = pd.to_numeric(sub[c], errors="coerce")
            mn, mx = (float(s.min()), float(s.max())) if s.notna().any() else (0.0, 1.0)
            if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx <= mn:
                mn, mx = 0.0, 1.0
            scalers[k] = {"min": mn, "max": mx}
    (ART / f"group_scalers_{kind}_{tag}.json").write_text(
        json.dumps(scalers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def main():
    df = _load_df()
    for k in KINDS:
        for t in TAGS:
            _train_one(df, k, t)

if __name__ == "__main__":
    main()
