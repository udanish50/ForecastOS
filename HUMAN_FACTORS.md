# ForecastOS Human-Factors Design Specification

ForecastOS is designed as a decision-support system, not a model playground. The interface therefore prioritizes comprehension, control, error prevention, and calibrated trust over exposing every technical parameter.

## Design objectives

1. **Make system status visible**
   - Training uses a persistent status container and progress indicator.
   - The model currently being tested is named in both plain language and its technical name.
   - Failed models are surfaced rather than silently omitted.

2. **Prevent high-cost mistakes before they happen**
   - Known-at-forecast-time variables are explicitly separated from the prediction target.
   - The interface warns that selecting unavailable future information can create leakage.
   - A data-readiness check is shown before model training.
   - Configuration changes after training trigger a stale-results warning.

3. **Progressive disclosure**
   - Required choices are visible first.
   - Deep learning is placed in an optional lab.
   - Technical diagnostics are divided into focused tabs after the decision summary.
   - Backtest folds, fingerprint details, limitations, and model failures are available without dominating the main path.

4. **Recognition rather than recall**
   - Technical model names are paired with plain-language labels.
   - Common feature names such as `lag_24` and cyclic calendar variables are explained in context.
   - The AI analyst provides example questions so users do not need to remember forecasting vocabulary.

5. **Trust calibration instead of automation bias**
   - Forecast Trust is described as a heuristic decision aid, never as a probability of correctness.
   - A neural model receives no preference simply because it is more complex.
   - Model selection is based on rolling temporal backtest performance.
   - Negative skill versus the seasonal baseline creates an explicit caution message.
   - Prediction intervals are described as empirical and non-guaranteed.
   - Feature importance and scenario experiments are explicitly separated from causal claims.

6. **Accessible interaction**
   - Light theme with strong foreground/background contrast.
   - Visible keyboard focus treatment.
   - Enlarged interactive target heights.
   - No required drag-only interaction.
   - Reduced-motion preference is respected through CSS.
   - Information is not communicated by color alone; status always includes text labels.

7. **Decision-first hierarchy**
   - The first post-training section shows selected model, trust, baseline skill, and RMSE.
   - The forecast view explains what should be checked before acting.
   - Detailed model comparison, XAI, stress testing, AI analysis, and export follow afterward.

8. **User control and reversibility**
   - Nothing trains until the user explicitly starts the tournament.
   - A new-experiment action clears trained session results.
   - Stress tests and scenarios run only after explicit user actions.
   - Optional LLM analysis is separated from numerical forecasting.

## Deep-learning UX policy

ForecastOS deliberately treats deep learning as an optional model family rather than a default recommendation.

Available models:

- Deep MLP autoregression
- LSTM sequence model
- Temporal convolutional network (TCN)
- Transformer sequence encoder

The user sees a compute-burden label before training. Each deep model is compared with naïve and classical baselines using the same rolling-origin evaluation protocol. This is intended to reduce complexity bias and make the system's recommendation defensible.

## Accessibility baseline

The Streamlit UI is designed toward WCAG 2.2 concepts relevant to an analytical application, including visible focus, minimum interactive target sizing, predictable controls, consistent contextual help, and programmatic status communication. Formal WCAG conformance still requires browser-level audit and assistive-technology testing after deployment.

## Remaining human-factors research opportunities

- User studies comparing expert and novice forecast interpretation.
- Decision-quality experiments with and without the Trust Score.
- Explanation comprehension testing for feature/lag/horizon visualizations.
- Adaptive terminology based on user expertise.
- Cost-of-error framing for asymmetric operational decisions.
- Cognitive-load measurement across compact and expert interface modes.
- Accessibility audit with keyboard-only, screen reader, zoom, and high-contrast workflows.
