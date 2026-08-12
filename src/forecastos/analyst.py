from __future__ import annotations

import json
import os


def evidence_packet(result, profile) -> dict:
    top_features = result.feature_importance.head(8).to_dict(orient="records") if not result.feature_importance.empty else []
    return {
        "winning_model": result.model_name,
        "metrics": {k: round(float(v), 4) for k, v in result.metrics.items()},
        "trust_score": round(float(result.trust_score), 1),
        "trust_components": {k: round(float(v), 1) for k, v in result.trust_components.items()},
        "top_features": top_features,
        "dataset": {
            "rows": profile.rows,
            "frequency": profile.inferred_frequency,
            "seasonal_period": profile.selected_seasonal_period,
            "regularity": round(profile.regularity_score, 3),
            "drift_score": round(profile.drift_score, 3),
            "warnings": profile.warnings,
        },
    }


def deterministic_brief(result, profile) -> str:
    skill = result.metrics.get("Skill vs baseline (%)", float("nan"))
    skill_text = f"{skill:.1f}%" if skill == skill else "not available"
    feature_text = ""
    if not result.feature_importance.empty:
        tops = result.feature_importance.head(3)["feature"].tolist()
        feature_text = " The strongest measured inputs were " + ", ".join(tops) + "."
    drift = "low" if profile.drift_score < 0.25 else "moderate" if profile.drift_score < 0.6 else "high"
    return (
        f"ForecastOS selected **{result.model_name}** after rolling backtests. "
        f"Its RMSE was **{result.metrics['RMSE']:.3f}**, with **{skill_text} forecast skill** versus the seasonal baseline. "
        f"The current Forecast Trust Score is **{result.trust_score:.0f}/100** and detected drift risk is {drift}."
        f"{feature_text} Treat the 90% interval as an empirical backtest-based uncertainty band rather than a guarantee."
    )


def ask_openai(question: str, result, profile, model: str = "gpt-5-mini") -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    from openai import OpenAI

    client = OpenAI(api_key=key)
    evidence = evidence_packet(result, profile)
    prompt = (
        "You are ForecastOS Analyst. Answer only from the supplied forecasting evidence. "
        "Do not invent causal claims, feature values, or analyses that were not run. Distinguish association/model attribution from causation. "
        "If evidence is insufficient, say what experiment would be needed. Be concise and decision-oriented.\n\n"
        f"EVIDENCE:\n{json.dumps(evidence, default=str)}\n\nUSER QUESTION:\n{question}"
    )
    response = client.responses.create(model=model, input=prompt)
    return response.output_text
