
import os
import pickle
from pathlib import Path
from flask import Flask, render_template_string, request

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# LOAD TRAINED MODEL AND VECTORIZER
# ==========================================

with open(BASE_DIR / "vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open(BASE_DIR / "sentiment.pkl", "rb") as file:
    model = pickle.load(file)


# ==========================================
# HTML, CSS AND JAVASCRIPT
# ==========================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>SentimentAI | Intelligent Sentiment Analysis</title>

<style>

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --background: #080d1b;
    --card: #121a2c;
    --border: #283249;
    --text: #f8fafc;
    --muted: #94a3b8;
    --positive: #22c55e;
    --negative: #f43f5e;
}

body {
    font-family: "Segoe UI", Arial, sans-serif;
    min-height: 100vh;
    color: var(--text);
    background:
        radial-gradient(circle at 10% 10%,
            rgba(99,102,241,0.17), transparent 35%),
        radial-gradient(circle at 90% 70%,
            rgba(139,92,246,0.13), transparent 35%),
        var(--background);
    display: flex;
    flex-direction: column;
}

/* NAVBAR */

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 7%;
    background: rgba(8,13,27,0.85);
    border-bottom: 1px solid var(--border);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 23px;
    font-weight: 800;
}

.logo {
    width: 43px;
    height: 43px;
    display: grid;
    place-items: center;
    border-radius: 13px;
    background: linear-gradient(135deg,
        var(--primary), var(--secondary));
    font-size: 23px;
}

.brand span {
    color: #a5b4fc;
}

.status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #86efac;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    padding: 9px 15px;
    border-radius: 30px;
    font-size: 12px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--positive);
    box-shadow: 0 0 12px var(--positive);
}

/* MAIN CONTENT */

main {
    flex: 1;
    width: 100%;
    max-width: 1250px;
    margin: auto;
    padding: 55px 30px;
}

.hero {
    text-align: center;
    margin-bottom: 45px;
}

.badge {
    display: inline-block;
    padding: 8px 17px;
    color: #c4b5fd;
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 22px;
}

.hero h1 {
    font-size: clamp(34px, 5vw, 57px);
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 18px;
}

.gradient-text {
    background: linear-gradient(90deg,
        #818cf8, #c084fc, #38bdf8);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: var(--muted);
    font-size: 16px;
    line-height: 1.8;
    max-width: 680px;
    margin: auto;
}

/* INFORMATION CARDS */

.info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-bottom: 25px;
}

.info-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px;
    border-radius: 16px;
    background: var(--card);
    border: 1px solid var(--border);
}

.info-icon {
    width: 48px;
    height: 48px;
    flex-shrink: 0;
    display: grid;
    place-items: center;
    border-radius: 13px;
    font-size: 23px;
    background: rgba(99,102,241,0.15);
}

.info-card h3 {
    font-size: 15px;
    margin-bottom: 5px;
}

.info-card p {
    font-size: 12px;
    color: var(--muted);
}

/* DASHBOARD */

.dashboard {
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 24px;
    align-items: stretch;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.18);
}

.card-heading {
    display: flex;
    gap: 13px;
    align-items: center;
    margin-bottom: 10px;
}

.card-heading h2 {
    font-size: 22px;
}

.card-description {
    font-size: 14px;
    color: var(--muted);
    line-height: 1.6;
    margin-bottom: 24px;
}

.input-label {
    display: block;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

textarea {
    width: 100%;
    min-height: 220px;
    resize: vertical;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: #0b1120;
    padding: 18px;
    font-family: inherit;
    font-size: 15px;
    line-height: 1.7;
    color: var(--text);
    outline: none;
    transition: 0.3s;
}

textarea:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 4px rgba(99,102,241,0.1);
}

textarea::placeholder {
    color: #64748b;
}

.input-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
    font-size: 12px;
    color: var(--muted);
}

.clear-button {
    background: none;
    border: none;
    color: #a5b4fc;
    cursor: pointer;
    font-size: 13px;
}

.clear-button:hover {
    color: white;
}

/* BUTTONS */

.analyze-btn {
    width: 100%;
    border: none;
    padding: 16px;
    border-radius: 13px;
    color: white;
    background: linear-gradient(135deg,
        var(--primary), var(--secondary));
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    margin-top: 20px;
    transition: 0.3s;
}

.analyze-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.35);
}

.analyze-btn:disabled {
    opacity: 0.7;
    cursor: wait;
}

.examples {
    margin-top: 24px;
}

.examples p {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 12px;
}

.example-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
}

.example-btn {
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
    padding: 9px 13px;
    font-size: 12px;
    border-radius: 9px;
    cursor: pointer;
    transition: 0.2s;
}

.example-btn:hover {
    border-color: var(--primary);
    background: rgba(99,102,241,0.15);
}

/* RESULTS */

.result-card {
    display: flex;
    flex-direction: column;
}

.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 330px;
    text-align: center;
}

.empty-icon {
    display: grid;
    place-items: center;
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: rgba(99,102,241,0.08);
    border: 1px dashed rgba(99,102,241,0.4);
    font-size: 48px;
    margin-bottom: 24px;
}

.empty-state h3 {
    font-size: 20px;
    margin-bottom: 12px;
}

.empty-state p {
    max-width: 300px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
}

.result-banner {
    text-align: center;
    padding: 27px 16px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.result-banner.positive {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.3);
    color: #4ade80;
}

.result-banner.negative {
    background: rgba(244,63,94,0.08);
    border: 1px solid rgba(244,63,94,0.3);
    color: #fb7185;
}

.sentiment-emoji {
    font-size: 57px;
    margin-bottom: 10px;
}

.result-banner h2 {
    font-size: 32px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-banner p {
    font-size: 13px;
    margin-top: 10px;
}

.confidence {
    background: #0b1120;
    border: 1px solid var(--border);
    padding: 18px;
    border-radius: 13px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.confidence p {
    font-size: 13px;
    color: var(--muted);
}

.confidence strong {
    font-size: 29px;
    color: #a5b4fc;
}

.probability-title {
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 20px;
}

.probability-row {
    margin-bottom: 19px;
}

.probability-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 10px;
}

.progress-track {
    height: 11px;
    width: 100%;
    border-radius: 10px;
    background: #273247;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 10px;
    animation: reveal 0.8s ease;
}

.positive-fill {
    background: linear-gradient(90deg,
        #16a34a, #4ade80);
}

.negative-fill {
    background: linear-gradient(90deg,
        #e11d48, #fb7185);
}

@keyframes reveal {
    from { width: 0; }
}

.result-note {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
    margin-top: 12px;
}

.error {
    background: rgba(244,63,94,0.1);
    color: #fda4af;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(244,63,94,0.3);
    font-size: 14px;
    margin-bottom: 20px;
}

/* HOW IT WORKS */

.how-it-works {
    margin-top: 45px;
}

.how-it-works h2 {
    font-size: 25px;
    text-align: center;
    margin-bottom: 27px;
}

.steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}

.step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}

.step-number {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    margin: 0 auto 16px;
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    border-radius: 12px;
    font-weight: 800;
}

.step h3 {
    font-size: 16px;
    margin-bottom: 10px;
}

.step p {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
}

/* FOOTER */

footer {
    border-top: 1px solid var(--border);
    text-align: center;
    padding: 25px 15px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.8;
}

footer strong {
    color: #a5b4fc;
}

/* RESPONSIVE DESIGN */

@media (max-width: 850px) {

    .dashboard {
        grid-template-columns: 1fr;
    }

    .info-grid {
        grid-template-columns: 1fr;
    }

    .steps {
        grid-template-columns: 1fr;
    }

    main {
        padding: 35px 20px;
    }

}

@media (max-width: 480px) {

    .navbar {
        padding: 15px;
    }

    .brand {
        font-size: 18px;
        gap: 8px;
    }

    .logo {
        width: 35px;
        height: 35px;
        font-size: 19px;
    }

    .status {
        font-size: 10px;
        padding: 7px 10px;
    }

    .card {
        padding: 21px;
    }

    .card-heading h2 {
        font-size: 19px;
    }

    .hero p {
        font-size: 14px;
    }

}

</style>

</head>

<body>

<nav class="navbar">

    <div class="brand">
        <div class="logo">✦</div>
        <div>Sentiment<span>AI</span></div>
    </div>

    <div class="status">
        <span class="status-dot"></span>
        Model Online
    </div>

</nav>

<main>

    <section class="hero">

        <div class="badge">
            MACHINE LEARNING · NLP
        </div>

        <h1>
            Understand the Emotion<br>
            <span class="gradient-text">
                Behind Every Word.
            </span>
        </h1>

        <p>
            Transform text into meaningful insights with
            intelligent sentiment analysis. Analyze reviews,
            feedback, opinions and messages in seconds.
        </p>

    </section>

    <section class="info-grid">

        <div class="info-card">
            <div class="info-icon">🧠</div>
            <div>
                <h3>Machine Learning</h3>
                <p>Multinomial Naive Bayes</p>
            </div>
        </div>

        <div class="info-card">
            <div class="info-icon">⚡</div>
            <div>
                <h3>Instant Analysis</h3>
                <p>Real-time text classification</p>
            </div>
        </div>

        <div class="info-card">
            <div class="info-icon">📊</div>
            <div>
                <h3>Intelligent NLP</h3>
                <p>5,000 TF-IDF features</p>
            </div>
        </div>

    </section>

    <section class="dashboard">

        <!-- INPUT SECTION -->

        <div class="card">

            <div class="card-heading">
                <div class="info-icon">✎</div>
                <h2>Analyze Your Text</h2>
            </div>

            <p class="card-description">
                Enter any English text to discover
                its underlying sentiment.
            </p>

            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}

            <form method="POST" id="analysisForm">

                <label class="input-label" for="text">
                    Your text
                </label>

                <textarea
                    id="text"
                    name="text"
                    maxlength="5000"
                    placeholder="Enter a review, feedback, opinion or any text here..."
                    required
                    oninput="updateCount()"
                >{{ text }}</textarea>

                <div class="input-footer">

                    <span id="charCount">
                        0 / 5000 characters
                    </span>

                    <button
                        type="button"
                        class="clear-button"
                        onclick="clearText()"
                    >
                        Clear text
                    </button>

                </div>

                <button
                    type="submit"
                    class="analyze-btn"
                    id="analyzeButton"
                >
                    ✦ Analyze Sentiment
                </button>

            </form>

            <div class="examples">

                <p>TRY AN EXAMPLE</p>

                <div class="example-buttons">

                    <button
                        type="button"
                        class="example-btn"
                        onclick="setExample(
                            'I absolutely love this product. The quality is amazing and the service was excellent.'
                        )"
                    >
                        😊 Positive Review
                    </button>

                    <button
                        type="button"
                        class="example-btn"
                        onclick="setExample(
                            'This was a terrible experience. The product was disappointing and a complete waste of money.'
                        )"
                    >
                        😞 Negative Review
                    </button>

                    <button
                        type="button"
                        class="example-btn"
                        onclick="setExample(
                            'The movie was wonderful and I enjoyed every moment of it.'
                        )"
                    >
                        🎬 Movie Review
                    </button>

                </div>

            </div>

        </div>

        <!-- RESULT SECTION -->

        <div class="card result-card">

            <div class="card-heading">
                <div class="info-icon">📈</div>
                <h2>Analysis Results</h2>
            </div>

            <p class="card-description">
                Sentiment prediction and model insights.
            </p>

            {% if result %}

                <div class="result-banner {{ result }}">

                    <div class="sentiment-emoji">
                        {% if result == "positive" %}
                            😊
                        {% else %}
                            😞
                        {% endif %}
                    </div>

                    <h2>{{ result }}</h2>

                    <p>
                        {% if result == "positive" %}
                            Your text expresses a positive sentiment.
                        {% else %}
                            Your text expresses a negative sentiment.
                        {% endif %}
                    </p>

                </div>

                <div class="confidence">
                    <p>Model Confidence</p>
                    <strong>{{ confidence }}%</strong>
                </div>

                <h3 class="probability-title">
                    Sentiment Distribution
                </h3>

                <div class="probability-row">

                    <div class="probability-label">
                        <span style="color:#4ade80">
                            Positive
                        </span>

                        <strong>{{ positive }}%</strong>
                    </div>

                    <div class="progress-track">
                        <div
                            class="progress-fill positive-fill"
                            style="width: {{ positive }}%">
                        </div>
                    </div>

                </div>

                <div class="probability-row">

                    <div class="probability-label">
                        <span style="color:#fb7185">
                            Negative
                        </span>

                        <strong>{{ negative }}%</strong>
                    </div>

                    <div class="progress-track">
                        <div
                            class="progress-fill negative-fill"
                            style="width: {{ negative }}%">
                        </div>
                    </div>

                </div>

                <p class="result-note">
                    Confidence represents the model's estimated
                    probability, not a guarantee of correctness.
                    This model classifies text as positive or negative.
                </p>

            {% else %}

                <div class="empty-state">

                    <div class="empty-icon">📊</div>

                    <h3>Ready to Analyze</h3>

                    <p>
                        Enter your text and click Analyze Sentiment
                        to view the prediction and confidence scores.
                    </p>

                </div>

            {% endif %}

        </div>

    </section>

    <section class="how-it-works">

        <h2>How It Works</h2>

        <div class="steps">

            <div class="step">
                <div class="step-number">01</div>
                <h3>Enter Text</h3>
                <p>
                    Provide a review, feedback or message
                    that you want to analyze.
                </p>
            </div>

            <div class="step">
                <div class="step-number">02</div>
                <h3>NLP Processing</h3>
                <p>
                    TF-IDF converts your text into numerical
                    features understood by the model.
                </p>
            </div>

            <div class="step">
                <div class="step-number">03</div>
                <h3>Sentiment Prediction</h3>
                <p>
                    The trained machine learning model predicts
                    sentiment and estimates class probabilities.
                </p>
            </div>

        </div>

    </section>

</main>

<footer>

    <strong>SentimentAI</strong>
    &nbsp; | &nbsp; Intelligent Sentiment Analysis

    <br>

    Built with Python, Flask, Machine Learning & NLP

</footer>

<script>

const textInput = document.getElementById("text");
const charCount = document.getElementById("charCount");

function updateCount() {
    charCount.textContent =
        textInput.value.length + " / 5000 characters";
}

function clearText() {
    textInput.value = "";
    updateCount();
    textInput.focus();
}

function setExample(value) {
    textInput.value = value;
    updateCount();
    textInput.focus();
}

document.getElementById("analysisForm")
    .addEventListener("submit", function(event) {

        if (!textInput.value.trim()) {
            event.preventDefault();
            alert("Please enter some text.");
            return;
        }

        const button =
            document.getElementById("analyzeButton");

        button.innerText = "Analyzing...";
        button.disabled = true;

    });

window.addEventListener("pageshow", function() {

    const button =
        document.getElementById("analyzeButton");

    button.disabled = false;
    button.innerText = "✦ Analyze Sentiment";

});

updateCount();

</script>

</body>
</html>
"""


# ==========================================
# SENTIMENT PREDICTION
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    text = ""
    result = None
    confidence = None
    positive = None
    negative = None
    error = None

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if not text:

            error = "Please enter some text to analyze."

        elif len(text) > 5000:

            error = "Text must not exceed 5000 characters."

        else:

            try:

                # Convert text into numerical features
                transformed_text = vectorizer.transform([text])

                if transformed_text.nnz == 0:

                    error = (
                        "No recognizable words were found. "
                        "Please enter a meaningful English sentence."
                    )

                else:

                    # Predict sentiment
                    prediction = model.predict(transformed_text)[0]

                    # Calculate class probabilities
                    probabilities = model.predict_proba(
                        transformed_text
                    )[0]

                    probability_map = {
                        str(label).lower(): float(probability)
                        for label, probability in zip(
                            model.classes_,
                            probabilities
                        )
                    }

                    result = str(prediction).lower()

                    positive = round(
                        probability_map["positive"] * 100, 2
                    )

                    negative = round(
                        probability_map["negative"] * 100, 2
                    )

                    confidence = round(
                        probability_map[result] * 100, 2
                    )

            except Exception:

                app.logger.exception("Sentiment prediction failed")

                result = None

                error = (
                    "Unable to analyze the text. "
                    "Please try again."
                )

    return render_template_string(
        HTML,
        text=text,
        result=result,
        confidence=confidence,
        positive=positive,
        negative=negative,
        error=error
    )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():

    return {
        "status": "healthy",
        "model": "MultinomialNB",
        "vectorizer": "TfidfVectorizer"
    }


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
