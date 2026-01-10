from textblob import TextBlob
from .steering import generate_steered_response


def run_steering_eval(
        prompt,
        vector,
        multiplier,
        model_name,
        layer_idx
):
    """
    Conducts a scientific A/B test:
    1. Generates text NORMALLY (Control).
    2. Generates text with STEERING (Treatment).
    3. Calculates metrics to quantify the impact.
    """

    # 1. Generate Control (Baseline)
    control_text = generate_steered_response(prompt, vector, 0.0, model_name, layer_idx)

    # 2. Generate Treatment (Intervention)
    steered_text = generate_steered_response(prompt, vector, multiplier, model_name, layer_idx)

    # 3. Analyze Sentiment (Polarity: -1.0 to +1.0)
    # This measures "Positivity/Negativity"
    score_control = TextBlob(control_text).sentiment.polarity
    score_steered = TextBlob(steered_text).sentiment.polarity
    delta = score_steered - score_control

    # 4. Analyze "Subjectivity" (0.0 to 1.0)
    # This measures if the text is factual (0) or opinionated (1)
    subj_control = TextBlob(control_text).sentiment.subjectivity
    subj_steered = TextBlob(steered_text).sentiment.subjectivity

    return {
        "control_text": control_text,
        "steered_text": steered_text,
        "metrics": {
            "sentiment_control": score_control,
            "sentiment_steered": score_steered,
            "sentiment_delta": delta,
            "subjectivity_control": subj_control,
            "subjectivity_steered": subj_steered
        }
    }