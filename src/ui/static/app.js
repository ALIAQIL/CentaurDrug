const state = {
  evaluation: null,
  optimization: null,
  selectedCandidateId: null,
  chatMessages: [],
  currentChatContext: null,
  currentChatContextKey: null,
};

const els = {
  apiStatus: document.getElementById("apiStatus"),
  smilesInput: document.getElementById("smilesInput"),
  depthInput: document.getElementById("depthInput"),
  beamInput: document.getElementById("beamInput"),
  candidateInput: document.getElementById("candidateInput"),
  maxMwInput: document.getElementById("maxMwInput"),
  avoidInput: document.getElementById("avoidInput"),
  evaluateBtn: document.getElementById("evaluateBtn"),
  optimizeBtn: document.getElementById("optimizeBtn"),
  decisionBadge: document.getElementById("decisionBadge"),
  evaluationSummary: document.getElementById("evaluationSummary"),
  riskList: document.getElementById("riskList"),
  candidateCount: document.getElementById("candidateCount"),
  candidateTable: document.getElementById("candidateTable"),
  selectedBadge: document.getElementById("selectedBadge"),
  candidateDetail: document.getElementById("candidateDetail"),
  treeCount: document.getElementById("treeCount"),
  treeView: document.getElementById("treeView"),
  chatMode: document.getElementById("chatMode"),
  chatMessages: document.getElementById("chatMessages"),
  promptChips: document.getElementById("promptChips"),
  chatForm: document.getElementById("chatForm"),
  chatInput: document.getElementById("chatInput"),
  chatSendBtn: document.getElementById("chatSendBtn"),
  toast: document.getElementById("toast"),
};

function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}

function fmtSigned(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  const number = Number(value);
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(digits)}`;
}

function cssClassForDecision(decision) {
  if (!decision) return "muted";
  if (decision === "pass") return "pass";
  if (decision === "uncertain") return "uncertain";
  return decision;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  window.setTimeout(() => els.toast.classList.remove("show"), 3200);
}

function setBusy(isBusy, action = null) {
  els.evaluateBtn.disabled = isBusy;
  els.optimizeBtn.disabled = isBusy;
  els.evaluateBtn.classList.toggle("is-loading", isBusy && action === "evaluate");
  els.optimizeBtn.classList.toggle("is-loading", isBusy && action === "optimize");

  if (isBusy) {
    els.apiStatus.textContent = action === "optimize" ? "Optimizing" : "Evaluating";
    els.apiStatus.className = "status-pill busy";
  } else {
    checkHealth();
  }
}

function setChatBusy(isBusy) {
  els.chatInput.disabled = isBusy;
  els.chatSendBtn.disabled = isBusy;
  if (isBusy) {
    els.chatMode.textContent = "Thinking";
    els.chatMode.className = "badge muted";
  }
}

async function postJson(url, payload) {
  let response;

  try {
    response = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
  } catch {
    const error = new Error("The API is not reachable.");
    error.kind = "network";
    throw error;
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : null;
    const error = new Error(detail || `Request failed with ${response.status}`);
    error.status = response.status;
    throw error;
  }

  return data;
}

function friendlyErrorMessage(error, fallback) {
  if (error?.kind === "network") {
    return "The API is not reachable. Check that the app server is running.";
  }

  if (error?.status === 400 && error.message) {
    return error.message;
  }

  if (error?.status === 422) {
    return "Check the input values and try again.";
  }

  return fallback;
}

function parseAvoidSubstructures() {
  return els.avoidInput.value
    .split(/[,\n;]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildConstraintsPayload() {
  const constraints = {};
  const maxMw = Number(els.maxMwInput.value);
  const avoidSubstructures = parseAvoidSubstructures();

  if (Number.isFinite(maxMw) && maxMw > 0) {
    constraints.max_mw = maxMw;
  }

  if (avoidSubstructures.length) {
    constraints.avoid_substructures = avoidSubstructures;
  }

  return constraints;
}

function requestPayload() {
  return {
    smiles: els.smilesInput.value.trim(),
    max_depth: Number(els.depthInput.value),
    beam_width: Number(els.beamInput.value),
    max_candidates_per_node: Number(els.candidateInput.value),
    include_full_tree: false,
    constraints: buildConstraintsPayload(),
  };
}

function renderEvaluation(evaluation) {
  state.evaluation = evaluation;
  const decision = evaluation?.final_decision?.decision || evaluation?.decision || "n/a";
  const assessment = evaluation?.overall_assessment || {};
  const rules = evaluation?.rules || evaluation || {};
  const admet = evaluation?.admet_predictions || {};

  els.decisionBadge.textContent = decision;
  els.decisionBadge.className = `badge ${cssClassForDecision(decision)}`;

  const metrics = [
    ["QED", rules.qed],
    ["Lipinski", rules.lipinski?.passed],
    ["Veber", rules.veber?.passed],
    ["PAINS", rules.pains?.passed],
    ["Brenk", rules.brenk?.passed],
    ["Solubility", admet.solubility?.prediction],
    ["Lipophilicity", admet.lipophilicity?.prediction],
    ["AMES risk", admet.ames?.probability_positive],
    ["hERG risk", admet.herg?.probability_positive],
    ["CYP3A4 risk", admet.cyp3a4?.probability_positive],
  ];

  els.evaluationSummary.innerHTML = metrics
    .map(([label, value]) => {
      const display = typeof value === "boolean" ? (value ? "Pass" : "Fail") : fmtNumber(value);
      return `
        <div class="metric">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(display)}</strong>
        </div>
      `;
    })
    .join("");

  const risks = assessment.main_risks || [];
  els.riskList.innerHTML = risks.length
    ? risks.map((risk) => `<span class="tag risk">${escapeHtml(risk)}</span>`).join("")
    : `<span class="tag">no major risk flags</span>`;
}

function renderRulePreview(rules, smiles) {
  state.evaluation = null;

  if (!rules?.valid) {
    els.decisionBadge.textContent = "Invalid";
    els.decisionBadge.className = "badge reject";
    els.evaluationSummary.innerHTML = "";
    els.riskList.innerHTML = `<span class="tag risk">${escapeHtml(
      rules?.reason || "invalid SMILES",
    )}</span>`;
    return;
  }

  els.decisionBadge.textContent = `Rules ${rules.decision}`;
  els.decisionBadge.className = `badge ${cssClassForDecision(rules.decision)}`;

  const metrics = [
    ["QED", rules.qed],
    ["Lipinski", rules.lipinski?.passed],
    ["Veber", rules.veber?.passed],
    ["PAINS", rules.pains?.passed],
    ["Brenk", rules.brenk?.passed],
  ];

  els.evaluationSummary.innerHTML = metrics
    .map(([label, value]) => {
      const display = typeof value === "boolean" ? (value ? "Pass" : "Fail") : fmtNumber(value);
      return `
        <div class="metric">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(display)}</strong>
        </div>
      `;
    })
    .join("");

  els.riskList.innerHTML = `
    <span class="tag">ADMET not refreshed</span>
    <span class="tag">${escapeHtml(smiles)}</span>
  `;
}

function markEvaluationStale(smiles) {
  state.evaluation = null;
  state.optimization = null;
  state.selectedCandidateId = null;

  els.decisionBadge.textContent = "Stale";
  els.decisionBadge.className = "badge stale";
  els.riskList.innerHTML = `
    <span class="tag">drug input changed</span>
    <span class="tag">${escapeHtml(smiles)}</span>
  `;
  els.candidateCount.textContent = "0";
  els.candidateTable.innerHTML = `
    <tr>
      <td colspan="6">Run Optimize to generate candidates for the current molecule.</td>
    </tr>
  `;
  els.treeCount.textContent = "0 nodes";
  els.treeView.innerHTML = "";
  renderCandidateDetail(null);
}

function setParentChatContext(evaluation, autoPrompt = true) {
  const smiles =
    evaluation?.canonical_smiles ||
    evaluation?.original_smiles ||
    els.smilesInput.value.trim();
  const context = {
    type: "parent",
    smiles,
    evaluation,
    decision: evaluation?.final_decision?.decision,
    main_risks: evaluation?.overall_assessment?.main_risks || [],
  };

  setCurrentChatContext(
    context,
    `parent:${smiles}:${context.decision || "unknown"}`,
    `Drug changed to parent molecule ${smiles}.`,
    "Explain the current parent molecule, its main ADMET risks, and what the agent should consider next.",
    autoPrompt,
  );
}

function reviewLabel(status) {
  const labels = {
    pending: "Pending",
    approved: "Approved",
    rejected: "Rejected",
    needs_review: "Needs Review",
  };
  return labels[status] || "Pending";
}

function renderCandidates(result, preferredNodeId = state.selectedCandidateId) {
  state.optimization = result;
  const candidates = result.best_candidate_nodes || [];
  els.candidateCount.textContent = String(candidates.length);

  if (!candidates.length) {
    els.candidateTable.innerHTML = `
      <tr>
        <td colspan="6">No candidates generated.</td>
      </tr>
    `;
    renderCandidateDetail(null);
    return;
  }

  state.selectedCandidateId = candidates.some(
    (candidate) => candidate.node_id === preferredNodeId,
  )
    ? preferredNodeId
    : candidates[0].node_id;

  els.candidateTable.innerHTML = candidates
    .map((candidate) => {
      const selected = candidate.node_id === state.selectedCandidateId ? " selected" : "";
      const decision = candidate.decision || "n/a";
      const reviewStatus = candidate.human_status || "pending";
      return `
        <tr
          class="candidate-row${selected}"
          data-node-id="${escapeHtml(candidate.node_id)}"
          tabindex="0"
          role="button"
          aria-label="Select candidate ${escapeHtml(candidate.smiles)}"
        >
          <td class="smiles-cell">${escapeHtml(candidate.smiles)}</td>
          <td>${escapeHtml(candidate.transformation || "root")}</td>
          <td>${fmtNumber(candidate.scalar_score)}</td>
          <td>${fmtSigned(candidate.delta_vs_parent)}</td>
          <td><span class="badge ${cssClassForDecision(decision)}">${escapeHtml(decision)}</span></td>
          <td><span class="badge ${escapeHtml(reviewStatus)}">${escapeHtml(reviewLabel(reviewStatus))}</span></td>
        </tr>
      `;
    })
    .join("");

  document.querySelectorAll(".candidate-row").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedCandidateId = row.dataset.nodeId;
      document
        .querySelectorAll(".candidate-row")
        .forEach((item) => item.classList.toggle("selected", item === row));
      renderCandidateDetail(findCandidate(row.dataset.nodeId));
    });
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        row.click();
      }
    });
  });

  renderCandidateDetail(findCandidate(state.selectedCandidateId));
}

function findCandidate(nodeId) {
  return (state.optimization?.best_candidate_nodes || []).find(
    (candidate) => candidate.node_id === nodeId,
  );
}

function renderCandidateDetail(candidate) {
  if (!candidate) {
    els.selectedBadge.textContent = "None";
    els.selectedBadge.className = "badge muted";
    els.candidateDetail.className = "detail-empty";
    els.candidateDetail.textContent = "Select a candidate row.";
    return;
  }

  els.selectedBadge.textContent = candidate.transformation || "Candidate";
  els.selectedBadge.className = "badge muted";
  els.candidateDetail.className = "detail-body";

  const scoreVector = candidate.score_vector || {};
  const scaffold = candidate.scaffold || {};
  const sa = candidate.synthetic_accessibility || {};
  const constraints = candidate.constraints || {};
  const constraintStatus =
    constraints.passed === undefined
      ? "not applied"
      : constraints.passed
        ? "passed"
        : "violated";
  const reviewStatus = candidate.human_status || "pending";

  els.candidateDetail.innerHTML = `
    <div>
      <div class="mono">${escapeHtml(candidate.smiles)}</div>
    </div>

    <div class="summary-list">
      <div class="metric">
        <span>Score</span>
        <strong>${fmtNumber(candidate.scalar_score)}</strong>
      </div>
      <div class="metric">
        <span>Delta</span>
        <strong>${fmtSigned(candidate.delta_vs_parent)}</strong>
      </div>
      <div class="metric">
        <span>Scaffold</span>
        <strong>${escapeHtml(scaffold.interpretation || "n/a")}</strong>
      </div>
      <div class="metric">
        <span>Synthesis</span>
        <strong>${escapeHtml(sa.interpretation || "n/a")}</strong>
      </div>
      <div class="metric">
        <span>Review</span>
        <strong>${escapeHtml(reviewLabel(reviewStatus))}</strong>
      </div>
    </div>

    <div class="review-actions" aria-label="Human review actions">
      <button type="button" data-review-status="approved">Approve</button>
      <button type="button" data-review-status="needs_review">Needs Review</button>
      <button type="button" data-review-status="rejected">Reject</button>
    </div>

    <div class="score-grid">
      ${Object.entries(scoreVector)
        .map(([key, value]) => renderScoreLine(key, value))
        .join("")}
    </div>

    <div class="list-block">
      <h3>Improvements</h3>
      ${renderList(candidate.improvements)}
    </div>

    <div class="list-block">
      <h3>Tradeoffs</h3>
      ${renderList(candidate.tradeoffs)}
    </div>

    <div class="list-block">
      <h3>Scaffold</h3>
      <ul>
        <li>Murcko preserved: ${escapeHtml(String(Boolean(scaffold.murcko_preserved)))}</li>
        <li>Fingerprint similarity: ${fmtNumber(scaffold.fingerprint_similarity)}</li>
        <li>Preservation score: ${fmtNumber(scaffold.preservation_score)}</li>
      </ul>
    </div>

    <div class="list-block">
      <h3>Synthetic Accessibility Proxy</h3>
      <ul>
        <li>Score: ${fmtNumber(sa.score)}</li>
        <li>MW: ${fmtNumber(sa.features?.molecular_weight, 1)}</li>
        <li>Rotatable bonds: ${escapeHtml(sa.features?.rotatable_bonds ?? "n/a")}</li>
        <li>Rings: ${escapeHtml(sa.features?.ring_count ?? "n/a")}</li>
      </ul>
    </div>

    <div class="list-block">
      <h3>Constraints</h3>
      <ul>
        <li>Status: ${escapeHtml(constraintStatus)}</li>
        <li>Violations: ${escapeHtml((constraints.violations || []).join(", ") || "none")}</li>
      </ul>
    </div>
  `;

  wireReviewActions(candidate.node_id);
  setCandidateChatContext(candidate);
}

function wireReviewActions(nodeId) {
  els.candidateDetail
    .querySelectorAll("[data-review-status]")
    .forEach((button) => {
      button.addEventListener("click", () => {
        setCandidateReviewStatus(nodeId, button.dataset.reviewStatus);
      });
    });
}

function setCandidateReviewStatus(nodeId, status) {
  const candidate = findCandidate(nodeId);
  if (!candidate) {
    return;
  }

  candidate.human_status = status;

  const treeNode = state.optimization?.tree_summary?.nodes?.[nodeId];
  if (treeNode) {
    treeNode.human_status = status;
  }

  renderCandidates(state.optimization, nodeId);
  showToast(`Candidate marked ${reviewLabel(status)}.`);
}

function setCandidateChatContext(candidate) {
  const context = {
    type: "candidate",
    smiles: candidate.smiles,
    candidate,
    decision: candidate.decision,
    main_risks: candidate.main_risks || [],
    optimization: {
      n_nodes: state.optimization?.n_nodes,
      n_candidate_nodes: state.optimization?.n_candidate_nodes,
      last_strategy: state.optimization?.last_strategy,
    },
  };

  setCurrentChatContext(
    context,
    `candidate:${candidate.node_id}`,
    `Drug changed to candidate ${candidate.smiles}.`,
    "Explain this selected candidate compared with its parent. Focus on score delta, improvements, tradeoffs, scaffold preservation, and synthesis proxy.",
    true,
  );
}

function setCurrentChatContext(context, key, systemMessage, prompt, autoPrompt) {
  state.currentChatContext = context;

  if (state.currentChatContextKey === key) {
    return;
  }

  state.currentChatContextKey = key;
  addChatMessage("system", systemMessage);

  if (autoPrompt) {
    sendChatPrompt(prompt, {showUser: false});
  }
}

function renderScoreLine(key, value) {
  const numeric = Math.max(0, Math.min(1, Number(value) || 0));
  return `
    <div class="score-line">
      <span>${escapeHtml(key.replaceAll("_", " "))}</span>
      <strong>${fmtNumber(value)}</strong>
      <div class="bar"><span style="width: ${numeric * 100}%"></span></div>
    </div>
  `;
}

function renderList(items) {
  if (!items || !items.length) {
    return "<ul><li>none</li></ul>";
  }
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function renderTree(result) {
  const nodes = result.tree_summary?.nodes || {};
  const nodeList = Object.values(nodes).sort((a, b) => {
    if (a.depth !== b.depth) return a.depth - b.depth;
    return Number(b.scalar_score || 0) - Number(a.scalar_score || 0);
  });

  els.treeCount.textContent = `${nodeList.length} nodes`;

  if (!nodeList.length) {
    els.treeView.innerHTML = "";
    return;
  }

  els.treeView.innerHTML = nodeList
    .map((node) => `
      <div class="tree-node depth-${Number(node.depth) || 0}">
        <strong>${escapeHtml(node.transformation || "parent")} - ${fmtNumber(node.scalar_score)}</strong>
        <span class="mono">${escapeHtml(node.smiles)}</span>
        <span>
          depth ${escapeHtml(node.depth)}
          | ${escapeHtml(node.decision || "n/a")}
          | ${escapeHtml(reviewLabel(node.human_status || "pending"))}
        </span>
      </div>
    `)
    .join("");
}

async function handleEvaluate() {
  const smiles = els.smilesInput.value.trim();
  if (!smiles) {
    showToast("Enter a SMILES string.");
    return;
  }

  setBusy(true, "evaluate");
  try {
    const evaluation = await postJson("/evaluate", {smiles});
    renderEvaluation(evaluation);
    setParentChatContext(evaluation, true);
    showToast("Evaluation complete.");
  } catch (error) {
    showToast(
      friendlyErrorMessage(
        error,
        "Evaluation failed. Check the molecule and try again.",
      ),
    );
  } finally {
    setBusy(false);
  }
}

async function handleOptimize() {
  const payload = requestPayload();
  if (!payload.smiles) {
    showToast("Enter a SMILES string.");
    return;
  }

  setBusy(true, "optimize");
  try {
    const result = await postJson("/optimize", payload);
    state.optimization = result;
    renderEvaluation(result.parent_evaluation || state.evaluation || {});
    setParentChatContext(result.parent_evaluation || state.evaluation || {}, false);
    renderCandidates(result);
    renderTree(result);
    showToast("Optimization complete.");
  } catch (error) {
    showToast(
      friendlyErrorMessage(
        error,
        "Optimization failed. Check the molecule and settings, then try again.",
      ),
    );
  } finally {
    setBusy(false);
  }
}

function addChatMessage(role, content) {
  state.chatMessages.push({role, content});

  if (state.chatMessages.length > 30) {
    state.chatMessages = state.chatMessages.slice(-30);
  }

  renderChatMessages();
}

function renderChatMessages() {
  if (!state.chatMessages.length) {
    els.chatMessages.innerHTML = `
      <div class="chat-message system">
        <strong>Copilot</strong>
        <p>Evaluate or optimize a molecule and I will follow the current drug context.</p>
      </div>
    `;
    return;
  }

  els.chatMessages.innerHTML = state.chatMessages
    .map((message) => `
      <div class="chat-message ${escapeHtml(message.role)}">
        <strong>${escapeHtml(labelForChatRole(message.role))}</strong>
        <p>${escapeHtml(message.content)}</p>
      </div>
    `)
    .join("");
  els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function labelForChatRole(role) {
  if (role === "assistant") return "Copilot";
  if (role === "user") return "You";
  return "Context";
}

function renderPromptChips(prompts) {
  const items = prompts || [];
  els.promptChips.innerHTML = items
    .map((prompt) => `
      <button type="button" class="prompt-chip" data-prompt="${escapeHtml(prompt)}">
        ${escapeHtml(prompt)}
      </button>
    `)
    .join("");

  document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      sendChatPrompt(chip.dataset.prompt, {showUser: true});
    });
  });
}

function chatHistoryForApi() {
  return state.chatMessages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .slice(-8);
}

async function sendChatPrompt(prompt, options = {}) {
  const trimmed = String(prompt || "").trim();

  if (!trimmed) {
    return;
  }

  if (options.showUser !== false) {
    addChatMessage("user", trimmed);
  }

  setChatBusy(true);
  try {
    const response = await postJson("/chat", {
      smiles: state.currentChatContext?.smiles || els.smilesInput.value.trim(),
      prompt: trimmed,
      context: state.currentChatContext || {
        type: "input",
        smiles: els.smilesInput.value.trim(),
      },
      messages: chatHistoryForApi(),
    });
    addChatMessage("assistant", response.reply);
    renderPromptChips(response.suggested_prompts);
    els.chatMode.textContent = response.mode === "gemini" ? "Gemini" : "Fallback";
    els.chatMode.className = "badge muted";
  } catch (error) {
    addChatMessage(
      "assistant",
      friendlyErrorMessage(
        error,
        "I could not answer that request right now.",
      ),
    );
  } finally {
    setChatBusy(false);
  }
}

function handleChatSubmit(event) {
  event.preventDefault();
  const prompt = els.chatInput.value.trim();
  els.chatInput.value = "";
  sendChatPrompt(prompt, {showUser: true});
}

let smilesChangeTimer = null;
let rulesPreviewRequestId = 0;

async function validateRulesPreview(smiles, key) {
  const requestId = ++rulesPreviewRequestId;

  try {
    const rules = await postJson("/rules", {smiles});
    if (requestId !== rulesPreviewRequestId || state.currentChatContextKey !== key) {
      return;
    }
    renderRulePreview(rules, smiles);
  } catch {
    if (requestId === rulesPreviewRequestId && state.currentChatContextKey === key) {
      els.decisionBadge.textContent = "Check failed";
      els.decisionBadge.className = "badge uncertain";
    }
  }
}

function handleSmilesChange() {
  window.clearTimeout(smilesChangeTimer);
  smilesChangeTimer = window.setTimeout(() => {
    const smiles = els.smilesInput.value.trim();

    if (!smiles) {
      return;
    }

    const key = `input:${smiles}`;
    if (state.currentChatContextKey === key) {
      return;
    }

    markEvaluationStale(smiles);
    state.currentChatContext = {
      type: "input",
      smiles,
    };
    state.currentChatContextKey = key;
    addChatMessage(
      "system",
      `Drug input changed to ${smiles}. Run Evaluate or Optimize to refresh ADMET context.`,
    );
    renderPromptChips([
      "Explain this SMILES before evaluation.",
      "What should I check first?",
      "Run evaluation then summarize the risk profile.",
    ]);
    validateRulesPreview(smiles, key);
    sendChatPrompt("Explain this SMILES before evaluation.", {showUser: false});
  }, 900);
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("health check failed");
    els.apiStatus.textContent = "API online";
    els.apiStatus.className = "status-pill ok";
  } catch {
    els.apiStatus.textContent = "API offline";
    els.apiStatus.className = "status-pill error";
  }
}

els.evaluateBtn.addEventListener("click", handleEvaluate);
els.optimizeBtn.addEventListener("click", handleOptimize);
els.chatForm.addEventListener("submit", handleChatSubmit);
els.smilesInput.addEventListener("input", handleSmilesChange);

checkHealth();
renderChatMessages();
handleEvaluate();
