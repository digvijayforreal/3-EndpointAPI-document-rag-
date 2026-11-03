
const API = "";

const fileInput = document.getElementById("fileInput");
const documentSelect = document.getElementById("documentSelect");
const documentStatus = document.getElementById("documentStatus");
const statusMessage = document.getElementById("statusMessage");
const messages = document.getElementById("chatMessages");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");

const chatHistory = {};

function setStatus(message) {
    statusMessage.textContent = message;
}

async function apiRequest(url, options = {}) {
    const response = await fetch(API + url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(
            typeof data.detail === "string"
                ? data.detail
                : "Request failed"
        );
    }

    return data;
}

async function loadDocuments(selectedId = null) {
    const data = await apiRequest("/documents");

    const previousId = selectedId || documentSelect.value;
    documentSelect.replaceChildren();

    const placeholder = new Option("Select document", "");
    documentSelect.add(placeholder);

    data.documents.forEach(doc => {
        const option = new Option(
            `${doc.filename} (${doc.embedding_status})`,
            doc.document_id
        );
        option.dataset.status = doc.embedding_status;
        option.dataset.chunks = doc.total_chunks;
        documentSelect.add(option);
    });

    if (data.documents.some(d => d.document_id === previousId)) {
        documentSelect.value = previousId;
    }

    updateDocumentStatus();
    renderHistory();
}

function updateDocumentStatus() {
    const selected = documentSelect.selectedOptions[0];

    if (!documentSelect.value) {
        documentStatus.textContent = "No document selected";
        return;
    }

    documentStatus.textContent =
        `${selected.dataset.chunks} chunks · ` +
        `Embedding: ${selected.dataset.status}`;
}

document.getElementById("uploadBtn").onclick = async () => {
    const file = fileInput.files[0];

    if (!file) {
        setStatus("Choose a PDF or DOCX file first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setStatus("Uploading and chunking document...");

    try {
        const result = await apiRequest("/upload", {
            method: "POST",
            body: formData
        });

        await loadDocuments(result.document_id);
        setStatus(`Uploaded successfully: ${result.total_chunks} chunks`);
        fileInput.value = "";

    } catch (error) {
        setStatus(error.message);
    }
};

document.getElementById("embedBtn").onclick = async () => {
    const documentId = documentSelect.value;

    if (!documentId) {
        setStatus("Select a document first.");
        return;
    }

    setStatus("Generating embeddings using Jina AI...");

    try {
        const result = await apiRequest(
            `/embed/${encodeURIComponent(documentId)}`,
            { method: "POST" }
        );

        await loadDocuments(documentId);
        setStatus(
            `Embedding completed. ${result.total_embedded} chunks processed.`
        );

    } catch (error) {
        setStatus(error.message);
    }
};

function createMessage(role, content, sources = []) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const text = document.createElement("div");
    text.className = "message-text";
    text.textContent = content;
    wrapper.appendChild(text);

    if (role === "assistant" && sources.length) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = `Retrieved evidence (${sources.length})`;
        details.appendChild(summary);

        sources.forEach(source => {
            const evidence = document.createElement("div");
            evidence.className = "evidence";

            const title = document.createElement("strong");
            const location = source.page !== null
                ? `Page ${source.page}`
                : `Chunk ${source.chunk_id}`;

            title.textContent =
                `Source ${source.source_id} — ${source.filename} — ${location}`;

            const score = document.createElement("small");
            score.textContent =
                `Similarity score: ${source.score.toFixed(3)}`;

            const body = document.createElement("p");
            body.textContent = source.text;

            evidence.append(title, score, body);
            details.appendChild(evidence);
        });

        wrapper.appendChild(details);
    }

    return wrapper;
}

function renderHistory() {
    const documentId = documentSelect.value;
    const history = chatHistory[documentId] || [];

    messages.replaceChildren();

    if (!history.length) {
        const welcome = document.createElement("div");
        welcome.className = "welcome";
        welcome.textContent =
            "Select an embedded document and ask a question.";
        messages.appendChild(welcome);
        return;
    }

    history.forEach(item => {
        messages.appendChild(
            createMessage(item.role, item.text, item.sources || [])
        );
    });

    messages.scrollTop = messages.scrollHeight;
}

document.getElementById("chatForm").onsubmit = async event => {
    event.preventDefault();

    const documentId = documentSelect.value;
    const question = questionInput.value.trim();

    if (!documentId || !question) {
        setStatus("Select a document and enter a question.");
        return;
    }

    const selected = documentSelect.selectedOptions[0];

    if (selected.dataset.status !== "completed") {
        setStatus("Generate embeddings before asking questions.");
        return;
    }

    chatHistory[documentId] ||= [];
    chatHistory[documentId].push({
        role: "user",
        text: question
    });

    questionInput.value = "";
    askBtn.disabled = true;
    setStatus("Searching document and generating answer...");
    renderHistory();

    try {
        const result = await apiRequest("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                document_id: documentId,
                question: question
            })
        });

        chatHistory[documentId].push({
            role: "assistant",
            text: result.answer,
            sources: result.sources
        });

        setStatus("");

    } catch (error) {
        chatHistory[documentId].push({
            role: "assistant",
            text: `Error: ${error.message}`
        });
        setStatus(error.message);

    } finally {
        askBtn.disabled = false;
        renderHistory();
    }
};

documentSelect.onchange = () => {
    updateDocumentStatus();
    renderHistory();
};

document.getElementById("clearBtn").onclick = () => {
    chatHistory[documentSelect.value] = [];
    renderHistory();
};

loadDocuments().catch(error => setStatus(error.message));
