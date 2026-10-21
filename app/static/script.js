document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const form = document.getElementById("sentiment-form");
    const tweetInput = document.getElementById("tweet-input");
    const charCounter = document.getElementById("char-counter");
    const clearBtn = document.getElementById("clear-btn");
    const submitBtn = document.getElementById("submit-btn");
    const sampleButtons = document.querySelectorAll(".sample-btn");

    const errorAlert = document.getElementById("error-alert");
    const errorMessage = document.getElementById("error-message");

    const resultSection = document.getElementById("result-section");
    const sentimentBadge = document.getElementById("sentiment-badge");
    const resultSentimentText = document.getElementById("result-sentiment-text");
    const resultConfidenceText = document.getElementById("result-confidence-text");
    const resultProbText = document.getElementById("result-prob-text");

    const negPct = document.getElementById("neg-pct");
    const posPct = document.getElementById("pos-pct");
    const negBar = document.getElementById("neg-bar");
    const posBar = document.getElementById("pos-bar");

    const cleanedTextDisplay = document.getElementById("cleaned-text-display");
    const tokenTagsContainer = document.getElementById("token-tags");

    // Live Character Counter
    function updateCharCounter() {
        const count = tweetInput.value.length;
        charCounter.textContent = `${count} / 280`;
        if (count >= 270) {
            charCounter.style.color = "#f87171";
        } else {
            charCounter.style.color = "";
        }
    }

    tweetInput.addEventListener("input", updateCharCounter);

    // Ctrl/Cmd + Enter to submit
    tweetInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            form.requestSubmit();
        }
    });

    // Sample Tweet Pills Click
    sampleButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const sampleText = btn.getAttribute("data-text");
            tweetInput.value = sampleText;
            updateCharCounter();
            tweetInput.focus();
            analyzeSentiment(sampleText);
        });
    });

    // Clear Action
    clearBtn.addEventListener("click", () => {
        tweetInput.value = "";
        updateCharCounter();
        hideError();
        resultSection.classList.add("hidden");
        tweetInput.focus();
    });

    // Form Submit
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = tweetInput.value.trim();
        if (!text) {
            showError("Please enter some text before analyzing.");
            return;
        }
        analyzeSentiment(text);
    });

    // Core Sentiment Analysis Fetch
    async function analyzeSentiment(text) {
        hideError();
        setLoading(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ text: text }),
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || "An error occurred during prediction.");
                resultSection.classList.add("hidden");
                return;
            }

            renderResults(data);
        } catch (err) {
            console.error("Prediction request failed:", err);
            showError("Could not connect to the backend server. Make sure app.py is running.");
            resultSection.classList.add("hidden");
        } finally {
            setLoading(false);
        }
    }

    // Render results to UI
    function renderResults(data) {
        const isPos = data.is_positive;
        const icon = isPos ? "😄" : "🙁";
        
        // Badge
        sentimentBadge.className = `sentiment-badge ${isPos ? "positive" : "negative"}`;
        sentimentBadge.textContent = `${icon} ${data.sentiment}`;

        // Text & Scores
        resultSentimentText.textContent = data.sentiment;
        resultSentimentText.style.color = isPos ? "var(--positive)" : "var(--negative)";
        resultConfidenceText.textContent = `${data.confidence}%`;
        resultProbText.textContent = data.score.toFixed(4);

        // Probability distribution bar
        posPct.textContent = `${data.positive_pct}%`;
        negPct.textContent = `${data.negative_pct}%`;
        posBar.style.width = `${data.positive_pct}%`;
        negBar.style.width = `${data.negative_pct}%`;

        // Preprocessing & Tokens Breakdown
        cleanedTextDisplay.textContent = data.cleaned_text || "(empty)";
        tokenTagsContainer.innerHTML = "";

        if (data.tokens && data.tokens.length > 0) {
            data.tokens.forEach((token) => {
                const tag = document.createElement("span");
                tag.className = "token-tag";
                tag.textContent = token;
                tokenTagsContainer.appendChild(tag);
            });
        } else {
            const noneSpan = document.createElement("span");
            noneSpan.style.color = "var(--text-muted)";
            noneSpan.style.fontSize = "0.8rem";
            noneSpan.textContent = "No recognized vocabulary tokens found.";
            tokenTagsContainer.appendChild(noneSpan);
        }

        // Show section and scroll into view smoothly
        resultSection.classList.remove("hidden");
        resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // UI State Helpers
    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.classList.add("loading");
            submitBtn.disabled = true;
        } else {
            submitBtn.classList.remove("loading");
            submitBtn.disabled = false;
        }
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.classList.remove("hidden");
    }

    function hideError() {
        errorMessage.textContent = "";
        errorAlert.classList.add("hidden");
    }
});
