const questionForm = document.getElementById("question-form");
const questionInput = document.getElementById("question");
const submitButton = document.getElementById("submit-button");
const statusMessage = document.getElementById("status");
const answerSection = document.getElementById("answer-section");
const answerContainer = document.getElementById("answer");
const sourcesList = document.getElementById("sources");
const timingsList = document.getElementById("timings");
const citationsList = document.getElementById("citations");
const clearButton = document.getElementById("clear-button");
const loadingSpinner = document.getElementById("loading-spinner");
const suggestionButtons = document.querySelectorAll(".suggestion-button");
let activeRequestController = null;
let requestId = 0;


function renderAnswer(answer) {
    answerContainer.replaceChildren();

    const lines = answer.split("\n");
    let currentList = null;

    function appendFormattedText(container, text) {
        const parts = text.split(/(\*\*.*?\*\*|\\\*\\\*.*?\\\*\\\*)/g);

        parts.forEach((part) => {
            const boldMatch = part.match(
                /^(?:\*\*(.*?)\*\*|\\\*\\\*(.*?)\\\*\\\*)$/
            );

            if (boldMatch) {
                const strong = document.createElement("strong");
                strong.textContent = boldMatch[1] ?? boldMatch[2];
                container.appendChild(strong);
            } else {
                container.appendChild(document.createTextNode(part));
            }
        });
    }

    lines.forEach((line) => {
        const trimmedLine = line.trim();

        if (!trimmedLine) {
            currentList = null;
            return;
        }

        const listMatch = trimmedLine.match(/^\d+[.)]\s+(.+)$/);

        if (listMatch) {
            if (!currentList) {
                currentList = document.createElement("ol");
                answerContainer.appendChild(currentList);
            }

            const item = document.createElement("li");
            appendFormattedText(item, listMatch[1]);
            currentList.appendChild(item);
            return;
        }

        currentList = null;

        const paragraph = document.createElement("p");
        appendFormattedText(paragraph, trimmedLine);
        answerContainer.appendChild(paragraph);
    });
}


questionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) {
        statusMessage.textContent = "Please enter a question.";
        return;
    }

    submitButton.disabled = true;
    submitButton.textContent = "Generating answer...";
    statusMessage.classList.remove("status-error");
    statusMessage.textContent = "Searching the document and generating an answer...";
    loadingSpinner.hidden = false;
    answerSection.hidden = true;
    answerContainer.replaceChildren();
    sourcesList.replaceChildren();

    const currentRequestId = ++requestId;
    activeRequestController = new AbortController();
    const signal = activeRequestController.signal;

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question }),
            signal
        });

        const data = await response.json();

        if (currentRequestId !== requestId) {
            return;
        }

        if (!response.ok) {
            throw new Error(
                typeof data.detail === "string"
                    ? data.detail
                    : "The request failed."
            );
        }

        renderAnswer(data.answer);
        // Extract unique page citations explicitly mentioned in the answer.
        citationsList.replaceChildren();

        const citedPages = [
            ...new Set(
                [...data.answer.matchAll(/\[Pages?\s+(\d+(?:\s*,\s*\d+)*)\]/gi)]
                    .flatMap((match) =>
                        match[1].split(",").map((page) => Number(page.trim()))
                    )
            )
        ];


        if (citedPages.length === 0) {
            const item = document.createElement("li");
            item.textContent = "No page citations found in the answer.";
            citationsList.appendChild(item);
        } else {
            citedPages.forEach((page) => {
                const item = document.createElement("li");
                item.textContent = `Page ${page}`;
                citationsList.appendChild(item);
            });
        }
        const allSources = Array.isArray(data.sources) ? data.sources : [];

        const sources = [
            ...new Map(
                allSources.map((source) => [
                    `${source.source ?? "unknown"}|${source.page ?? "unknown"}`,
                    source
                ])
            ).values()
        ];

        if (sources.length === 0) {
            const item = document.createElement("li");
            item.textContent = "No sources were returned.";
            sourcesList.appendChild(item);
        } else {
            sources.forEach((source) => {
                const item = document.createElement("li");
                const page = source.page ?? "unknown";
                const name = source.source ?? "unknown";
                item.textContent = `${name} — Page ${page}`;
                sourcesList.appendChild(item);
            });
        }

        timingsList.replaceChildren();

        const timings = [
            ["Retrieval", data.retrieval_time_seconds],
            ["Generation", data.generation_time_seconds],
            ["Total", data.total_time_seconds]
        ];

        timings.forEach(([label, seconds]) => {
            const item = document.createElement("li");
            const value = typeof seconds === "number"
                ? `${seconds.toFixed(3)} seconds`
                : "Unavailable";

            item.textContent = `${label}: ${value}`;
            timingsList.appendChild(item);
        });

        answerSection.hidden = false;
        statusMessage.textContent = "Answer generated successfully.";
    } catch (error) {
        if (error.name === "AbortError") {
            return;
        }

        if (currentRequestId === requestId) {
            statusMessage.textContent = error.message;
            statusMessage.classList.add("status-error");
        }
    }
    finally {
        if (currentRequestId === requestId) {
            loadingSpinner.hidden = true;
            submitButton.disabled = false;
            submitButton.textContent = "Ask question";
            activeRequestController = null;
        }
    }
});


clearButton.addEventListener("click", () => {
    requestId++;

    if (activeRequestController) {
        activeRequestController.abort();
        activeRequestController = null;
    }

    questionInput.value = "";
    answerContainer.replaceChildren();
    citationsList.replaceChildren();
    sourcesList.replaceChildren();
    timingsList.replaceChildren();

    loadingSpinner.hidden = true;
    answerSection.hidden = true;
    statusMessage.classList.remove("status-error");
    statusMessage.textContent = "";
    submitButton.disabled = false;
    submitButton.textContent = "Ask question";
    questionInput.focus();
});

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) {
        event.preventDefault();
        questionForm.requestSubmit();
    }
});

suggestionButtons.forEach((button) => {
    button.addEventListener("click", () => {
        questionInput.value = button.dataset.question;
        questionInput.focus();
    });
});