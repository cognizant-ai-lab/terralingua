import { state, humanPendingPrompts, humanWaiting, escHtml } from './state.js';

function buildParamInputs(actionName, spec){
  if(!spec || !Object.keys(spec).length) return "";
  let html = ``;
  for(const [pname, pval] of Object.entries(spec)){
    if(actionName==="create_artifact" && pname==="type") continue;
    const choices = pval?.choices;
    if(actionName==="move" && pname==="direction"){
      html += `<div class="human-dir-grid">
        <span></span>
        <button class="human-dir-btn" data-param="direction" data-value="up" type="button">↑</button>
        <span></span>
        <button class="human-dir-btn" data-param="direction" data-value="left" type="button">←</button>
        <button class="human-dir-btn" data-param="direction" data-value="stay" type="button">·</button>
        <button class="human-dir-btn" data-param="direction" data-value="right" type="button">→</button>
        <span></span>
        <button class="human-dir-btn" data-param="direction" data-value="down" type="button">↓</button>
        <span></span>
      </div>`;
    } else if(choices && choices.length){
      html += `<div class="human-param-row">
        <label class="human-param-label">${escHtml(pname)}</label>
        <select class="human-param-input" data-param="${escHtml(pname)}">
          ${choices.map(c=>`<option value="${escHtml(c)}">${escHtml(c)}</option>`).join("")}
        </select>
      </div>`;
    } else if(pname==="amount"||pname==="energy"||pname==="lifespan"){
      html += `<div class="human-param-row">
        <label class="human-param-label">${escHtml(pname)}</label>
        <input type="number" class="human-param-input" data-param="${escHtml(pname)}" value="1" min="0" />
      </div>`;
    } else {
      html += `<div class="human-param-row">
        <label class="human-param-label">${escHtml(pname)}</label>
        <input type="text" class="human-param-input" data-param="${escHtml(pname)}" />
      </div>`;
    }
  }
  return html;
}

function collectParamValues(card, actionName){
  const params = {};
  if(actionName==="move"){
    const btn = card.querySelector(".human-dir-btn.selected");
    params.direction = btn ? btn.dataset.value : "stay";
  }
  for(const el of card.querySelectorAll(".human-param-input")){
    const n = el.dataset.param;
    params[n] = el.type==="number" ? (el.value!==""?parseInt(el.value,10):0) : el.value.trim();
  }
  if(actionName==="create_artifact") params.type = "text";
  return params;
}

function buildHumanCard(tag, payload, ws){
  const card = document.createElement("div");
  card.className = "human-card";
  card.dataset.tag = tag;
  const messages = (payload.observation_raw || {}).message || {};
  const availableActions = payload.available_actions || {};
  const step = payload.step ?? "?";

  const msgEntries = Object.entries(messages);
  let msgsHtml = "";
  if(msgEntries.length){
    msgsHtml = `<div class="human-card-msgs">${
      msgEntries.map(([s,t])=>`<div class="human-card-msg"><b>${escHtml(s)}:</b> ${escHtml(t)}</div>`).join("")
    }</div>`;
  }

  const actionNames = Object.keys(availableActions).filter(a => a !== "reproduce");
  card.innerHTML = `
    ${msgsHtml}
    <div class="human-card-action-row">
      <select class="human-card-action-select">${actionNames.map(a=>`<option value="${escHtml(a)}">${escHtml(a)}</option>`).join("")}</select>
      <span class="human-card-step">step ${step}</span>
    </div>
    <div class="human-card-params"></div>
    <input class="human-card-message" type="text" placeholder="Message (optional)…" autocomplete="one-time-code" style="margin-top:0.35rem;" />
    <button class="human-card-submit" type="button">Submit</button>
  `;

  const selectEl = card.querySelector(".human-card-action-select");
  const paramsDiv = card.querySelector(".human-card-params");

  function refreshParams(){
    const an = selectEl.value;
    const spec = availableActions[an]?.params || {};
    paramsDiv.innerHTML = buildParamInputs(an, spec);
    paramsDiv.querySelectorAll(".human-dir-btn").forEach(btn=>{
      btn.addEventListener("click",()=>{
        paramsDiv.querySelectorAll(".human-dir-btn").forEach(b=>b.classList.remove("selected"));
        btn.classList.add("selected");
      });
    });
  }
  refreshParams();
  selectEl.addEventListener("change", refreshParams);

  card.querySelector(".human-card-submit").addEventListener("click",()=>{
    const an = selectEl.value;
    const params = collectParamValues(card, an);
    if(an === "create_artifact" && !params.name?.trim()){
      const nameInput = card.querySelector('.human-param-input[data-param="name"]');
      if(nameInput){ nameInput.style.outline="2px solid red"; nameInput.focus(); }
      return;
    }
    const message = card.querySelector(".human-card-message").value.trim();
    ws.send(JSON.stringify({action:an, params, message}));
    humanWaiting.set(tag, humanPendingPrompts.get(tag));
    humanPendingPrompts.delete(tag);
    card.classList.add("human-card--waiting");
    card.querySelectorAll("input, select, button, textarea").forEach(el => {
      el.disabled = true;
      if(el.classList.contains("human-card-submit")) el.textContent = "Waiting for next step…";
    });
  });

  return card;
}

export function updateHumanCardObs(card, payload){
  const messages = (payload.observation_raw || {}).message || {};
  const stepEl = card.querySelector(".human-card-step");
  if(stepEl) stepEl.textContent = `step ${payload.step ?? "?"}`;
  const availableActions = payload.available_actions || {};
  const actionNames = Object.keys(availableActions).filter(a => a !== "reproduce");

  // Update messages
  const oldMsgs = card.querySelector(".human-card-msgs");
  const msgEntries = Object.entries(messages);
  const selectEl = card.querySelector(".human-card-action-select");
  if(msgEntries.length){
    const html = `<div class="human-card-msgs">${msgEntries.map(([s,t])=>`<div class="human-card-msg"><b>${escHtml(s)}:</b> ${escHtml(t)}</div>`).join("")}</div>`;
    if(oldMsgs) oldMsgs.outerHTML = html;
    else if(selectEl) selectEl.insertAdjacentHTML("beforebegin", html);
  } else {
    if(oldMsgs) oldMsgs.remove();
  }

  const paramsDiv = card.querySelector(".human-card-params");
  if(selectEl){
    const prevAction = selectEl.value;
    selectEl.innerHTML = actionNames.map(a=>`<option value="${escHtml(a)}">${escHtml(a)}</option>`).join("");
    if(actionNames.includes(prevAction)) selectEl.value = prevAction;
    if(paramsDiv && selectEl.value !== prevAction){
      const an = selectEl.value;
      const spec = availableActions[an]?.params || {};
      paramsDiv.innerHTML = buildParamInputs(an, spec);
      paramsDiv.querySelectorAll(".human-dir-btn").forEach(btn=>{
        btn.addEventListener("click",()=>{
          paramsDiv.querySelectorAll(".human-dir-btn").forEach(b=>b.classList.remove("selected"));
          btn.classList.add("selected");
        });
      });
    }
    selectEl.onchange = () => {
      const an = selectEl.value;
      const spec = availableActions[an]?.params || {};
      paramsDiv.innerHTML = buildParamInputs(an, spec);
      paramsDiv.querySelectorAll(".human-dir-btn").forEach(btn=>{
        btn.addEventListener("click",()=>{
          paramsDiv.querySelectorAll(".human-dir-btn").forEach(b=>b.classList.remove("selected"));
          btn.classList.add("selected");
        });
      });
    };
    const submitBtn = card.querySelector(".human-card-submit");
    if(submitBtn){
      submitBtn.onclick = () => {
        const an = selectEl.value;
        const params = collectParamValues(card, an);
        const message = card.querySelector(".human-card-message").value.trim();
        const tag = card.dataset.tag;
        const pending = humanPendingPrompts.get(tag);
        if(!pending) return;
        pending.ws.send(JSON.stringify({action:an, params, message}));
        humanWaiting.set(tag, pending);
        humanPendingPrompts.delete(tag);
        card.classList.add("human-card--waiting");
        card.querySelectorAll("input, select, button, textarea").forEach(el => {
          el.disabled = true;
          if(el.classList.contains("human-card-submit")) el.textContent = "Waiting…";
        });
      };
    }
  }
  card.classList.remove("human-card--waiting");
  card.querySelectorAll("input, select, button, textarea").forEach(el => {
    el.disabled = false;
    if(el.classList.contains("human-card-submit")) el.textContent = "Submit";
  });
}

export function renderHumanControlPanel(){
  const body = document.getElementById("humanControlBody");
  const pending = state.selectedTag ? humanPendingPrompts.get(state.selectedTag) : null;
  const waiting = state.selectedTag ? humanWaiting.get(state.selectedTag) : null;
  if(!pending && !waiting){ body.style.display="none"; body.innerHTML=""; return; }
  body.style.display = "";
  const existing = body.querySelector(".human-card[data-tag]");
  const cardMatches = existing && existing.dataset.tag === state.selectedTag;

  if(pending){
    humanWaiting.delete(state.selectedTag);
    if(cardMatches){
      updateHumanCardObs(existing, pending.payload);
    } else {
      body.innerHTML = "";
      body.appendChild(buildHumanCard(state.selectedTag, pending.payload, pending.ws));
    }
  } else {
    if(cardMatches){
      // Card already showing in waiting state — nothing to rebuild
    } else {
      body.innerHTML = "";
      const card = buildHumanCard(state.selectedTag, waiting.payload, waiting.ws);
      card.classList.add("human-card--waiting");
      const btn = card.querySelector(".human-card-submit");
      if(btn){ btn.textContent = "Waiting…"; btn.disabled = true; }
      body.appendChild(card);
    }
  }
}
