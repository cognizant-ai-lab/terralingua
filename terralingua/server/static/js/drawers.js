import { state, escHtml, agentName } from './state.js';
import { loadApiKey, saveApiKey, getAnonId } from './crypto.js';
import { myArtifacts, myArtifactVersions, _saveMyArtifacts } from './artifacts.js';
import { agentStore } from './agents.js';
import { drawGrid } from './grid.js';
import { _confirm } from './modals.js';

let _wsRequest = null;
let _deselectAgent = null;
export function initDrawers(wsReq, deselectAgentFn) { _wsRequest = wsReq; _deselectAgent = deselectAgentFn; }

let _artEditMode = false;
let _artEditName = null;

// ── Genome ────────────────────────────────────────────────────────────────────
// Caches of user-entered genome values, keyed by genome type. Let the user
// switch between genotypes without losing what they typed/tweaked. The add and
// edit drawers each have their own cache so values don't bleed across them;
// the edit cache is also cleared when applyEditGenomeData runs so values from
// one agent's edit session don't appear in the next.
const _genomeCache = {};
let _lastGenomeType = "no_traits";
const _editGenomeCache = {};
let _lastEditGenomeType = "no_traits";

function _captureGenomeFrom(type, sentenceId, sliderPrefix, cache){
  if(type==="sentence_mutate" || type==="sentence_directed"){
    const el = document.getElementById(sentenceId);
    if(el) cache[type] = el.value;
  } else if(type==="ocean_5"){
    const fields = state.genomeInfo?.types?.ocean_5?.fields || [];
    const data = {};
    for(const f of fields){
      const el = document.getElementById(`${sliderPrefix}${f.name}`);
      if(el) data[f.name] = parseFloat(el.value);
    }
    cache.ocean_5 = data;
  }
}

function _restoreGenomeTo(type, sentenceId, sliderPrefix, valuePrefix, cache){
  const cached = cache[type];
  if(cached==null) return;
  if(type==="sentence_mutate" || type==="sentence_directed"){
    const el = document.getElementById(sentenceId);
    if(el) el.value = cached;
  } else if(type==="ocean_5"){
    for(const [name,val] of Object.entries(cached)){
      const sl = document.getElementById(`${sliderPrefix}${name}`);
      const vl = document.getElementById(`${valuePrefix}${name}`);
      if(sl) sl.value = val;
      if(vl) vl.textContent = parseFloat(val).toFixed(2);
    }
  }
}

function _captureGenome(type){ _captureGenomeFrom(type, "dGenomeSentence", "gr_",  _genomeCache); }
function _restoreGenome(type){ _restoreGenomeTo  (type, "dGenomeSentence", "gr_",  "gv_",  _genomeCache); }
function _captureEditGenome(type){ _captureGenomeFrom(type, "eGenomeSentence", "egr_", _editGenomeCache); }
function _restoreEditGenome(type){ _restoreGenomeTo  (type, "eGenomeSentence", "egr_", "egv_", _editGenomeCache); }

async function loadGenomeInfo(){
  try{ state.genomeInfo = await _wsRequest("get_genome_info"); }
  catch(_){ state.genomeInfo = {active:"no_traits", types:{no_traits:{fields:[]},sentence_mutate:{fields:[]},sentence_directed:{fields:[]},ocean_5:{fields:[]}}}; }
  // Set dropdown to active type
  document.getElementById("dGenomeType").value = "no_traits";
  _lastGenomeType = "no_traits";
  renderGenomeFields();
}

function renderGenomeFields(){
  const sec  = document.getElementById("dGenomeSection");
  const type = document.getElementById("dGenomeType").value;
  const randomBtn = document.getElementById("dGenomeRandomBtn");
  const fields = state.genomeInfo?.types?.[type]?.fields || [];

  randomBtn.style.display = type==="ocean_5" ? "" : "none";

  if(type==="no_traits" || fields.length===0){
    sec.innerHTML=`<p style="font-size:0.72rem;color:var(--muted);margin-bottom:0.25rem;">No configurable traits.</p>`;
    return;
  }

  if(type==="sentence_mutate"){
    sec.innerHTML=`
      <input id="dGenomeSentence" type="text" placeholder="e.g. curious bold idealist…" />
      <p style="font-size:0.68rem;color:var(--muted);margin-top:0.3rem;line-height:1.4;">Each generation the text is slightly mutated — words may be added, changed, or removed. Leave blank for a random starting genome.</p>`;
    return;
  }

  if(type==="sentence_directed"){
    sec.innerHTML=`
      <input id="dGenomeSentence" type="text" placeholder="e.g. you are an idealist who wants to change the world" />
      <p style="font-size:0.68rem;color:var(--muted);margin-top:0.3rem;line-height:1.4;">The parent agent writes a custom description for each offspring. If the parent passes an empty string, the offspring inherits a mutated version instead. Leave blank to start with no personality.</p>`;
    return;
  }

  if(type==="ocean_5"){
    const groups={};
    for(const f of fields){ const g=f.trait_type||"other"; (groups[g]||(groups[g]=[])).push(f); }
    let html="";
    for(const [gname,gfields] of Object.entries(groups)){
      html+=`<div class="genome-group-label">${gname}</div>`;
      for(const f of gfields){
        const [lo,hi]=f.range;
        html+=`<div class="genome-row">
          <div class="genome-row-hdr">
            <span class="genome-trait-name">${f.name.replace(/_/g," ")}</span>
            <span class="genome-trait-val" id="gv_${f.name}">${f.default.toFixed(2)}</span>
          </div>
          <input type="range" class="genome-slider" id="gr_${f.name}"
            min="${lo}" max="${hi}" step="0.01" value="${f.default}"
            oninput="document.getElementById('gv_${f.name}').textContent=parseFloat(this.value).toFixed(2)" />
          <p class="genome-trait-descr">${escHtml(f.descr)}</p>
        </div>`;
      }
    }
    sec.innerHTML=html;
    return;
  }
}

function randomizeGenome(){
  const type=document.getElementById("dGenomeType").value;
  const fields=state.genomeInfo?.types?.[type]?.fields||[];
  if(type==="ocean_5"){
    for(const f of fields){
      const [lo,hi]=f.range;
      const val=Math.random()*(hi-lo)+lo;
      const sl=document.getElementById(`gr_${f.name}`);
      const vl=document.getElementById(`gv_${f.name}`);
      if(sl){sl.value=val;vl.textContent=val.toFixed(2);}
    }
  }
}

function collectGenomeData(){
  const type=document.getElementById("dGenomeType").value;
  const fields=state.genomeInfo?.types?.[type]?.fields||[];
  if(type==="no_traits" || fields.length===0) return null;
  if(type==="sentence_mutate"){
    const raw=document.getElementById("dGenomeSentence")?.value?.trim();
    return raw ? {words: raw.split(/\s+/).filter(Boolean)} : null;
  }
  if(type==="sentence_directed"){
    const sentence=document.getElementById("dGenomeSentence")?.value?.trim();
    return sentence ? {sentence} : null;
  }
  if(type==="ocean_5"){
    const data={};
    for(const f of fields){
      const el=document.getElementById(`gr_${f.name}`);
      data[f.name]=el?parseFloat(el.value):f.default;
    }
    return data;
  }
  return null;
}

document.getElementById("dGenomeType").addEventListener("change", function(){
  _captureGenome(_lastGenomeType);
  renderGenomeFields();
  const newType = this.value;
  _restoreGenome(newType);
  _lastGenomeType = newType;
});
document.getElementById("dGenomeRandomBtn").addEventListener("click", randomizeGenome);

// ── Actions ───────────────────────────────────────────────────────────────────

async function loadActionsInfo(){
  try{ state.actionsInfo = (await _wsRequest("get_actions")).actions || []; }
  catch(_){ state.actionsInfo = []; }
  renderActionChips();
}

function renderActionChips(){
  const sec = document.getElementById("dActionsSection");
  if(!state.actionsInfo.length){ sec.innerHTML=""; return; }
  sec.innerHTML = `<div class="action-chips">${
    state.actionsInfo.map(a=>`<span class="action-chip on" data-action="${escHtml(a.name)}" title="${escHtml(a.description)}">${escHtml(a.name.replace(/_/g," "))}</span>`).join("")
  }</div>`;
  sec.querySelectorAll(".action-chip").forEach(chip=>{
    chip.addEventListener("click",()=>chip.classList.toggle("on"));
  });
}

function setAllActions(on){
  document.querySelectorAll("#dActionsSection .action-chip").forEach(c=>c.classList.toggle("on",on));
}

function collectExcludedActions(){
  const excluded=[];
  document.querySelectorAll("#dActionsSection .action-chip").forEach(c=>{
    if(!c.classList.contains("on")) excluded.push(c.dataset.action);
  });
  return excluded.length ? excluded : null;
}

document.getElementById("dActionsAllBtn").addEventListener("click",()=>setAllActions(true));
document.getElementById("dActionsNoneBtn").addEventListener("click",()=>setAllActions(false));

// ── Agent type toggle ─────────────────────────────────────────────────────────
document.getElementById("dTypeLLM").addEventListener("click",()=>{
  state._humanAgentType="remote_llm";
  document.getElementById("dTypeLLM").classList.add("active");
  document.getElementById("dTypeHuman").classList.remove("active");
  document.getElementById("dLLMFields").style.display="";
  document.getElementById("dGenomeFields").style.display="";
});
document.getElementById("dTypeHuman").addEventListener("click",()=>{
  state._humanAgentType="remote_human";
  document.getElementById("dTypeHuman").classList.add("active");
  document.getElementById("dTypeLLM").classList.remove("active");
  document.getElementById("dLLMFields").style.display="none";
  document.getElementById("dGenomeFields").style.display="none";
});

// ── Drawer ────────────────────────────────────────────────────────────────────
async function suggestAgentName(){
  const existing=[...state.tagToAgent.values()].map(a=>a.name).filter(Boolean).join(",");
  try{
    const r=await fetch(`/suggest_agent_name?existing=${encodeURIComponent(existing)}`);
    if(r.ok){const d=await r.json();document.getElementById("dName").value=d.name;}
  }catch(e){/* ignore */}
}
document.getElementById("dNameRefreshBtn").addEventListener("click",suggestAgentName);

document.getElementById("addAgentBtn").addEventListener("click",()=>{
  stopPicking();
  document.getElementById("artDrawer").classList.remove("open");
  // Reset to LLM type on open
  state._humanAgentType="remote_llm";
  document.getElementById("dTypeLLM").classList.add("active");
  document.getElementById("dTypeHuman").classList.remove("active");
  document.getElementById("dLLMFields").style.display="";
  document.getElementById("dGenomeFields").style.display="";
  document.getElementById("drawer").classList.add("open");
  suggestAgentName();
  loadGenomeInfo();loadActionsInfo();
});
document.getElementById("drawerCancelBtn").addEventListener("click",()=>{stopPicking();document.getElementById("drawer").classList.remove("open");});

// ── Artifact drawer ───────────────────────────────────────────────────────────
function openArtDrawerCreate(){
  _artEditMode=false; _artEditName=null;
  document.getElementById("artDrawerTitle").textContent="◈ Add Artifact";
  document.getElementById("artDrawer").classList.remove("edit-mode");
  document.getElementById("artDrawerCreateBtn").textContent="Create";
  document.getElementById("aName").value="";
  document.getElementById("aPayload").value="";
  document.getElementById("aLifespan").value="-1";
  document.getElementById("aMovable").checked=true;
  document.getElementById("artDrawerStatus").className="art-drawer-status";
  document.getElementById("artDrawerStatus").textContent="";
  stopPicking();
  document.getElementById("drawer").classList.remove("open");
  if(state.lastFrame && state.lastFrame.grid_state.grid_size != null){
    const gs=state.lastFrame.grid_state.grid_size;
    document.getElementById("aRow").max=gs-1;
    document.getElementById("aCol").max=gs-1;
    if(!document.getElementById("aRow").value||document.getElementById("aRow").value==="0"){
      document.getElementById("aRow").value=Math.floor(gs/2);
      document.getElementById("aCol").value=Math.floor(gs/2);
    }
  }
  document.getElementById("artDrawer").classList.add("open");
}
export function openArtDrawerEdit(art){
  _artEditMode=true; _artEditName=art.name;
  document.getElementById("artDrawerTitle").textContent=`✎ Edit: ${art.name}`;
  document.getElementById("artDrawer").classList.add("edit-mode");
  document.getElementById("artDrawerCreateBtn").textContent="Save";
  const raw=art.payload;
  document.getElementById("aPayload").value=typeof raw==="object"?JSON.stringify(raw):String(raw??"");
  const rem=art.remaining_time??art.lifespan??-1;
  document.getElementById("aLifespan").value=(rem==="inf"||rem===Infinity||rem===null)?"-1":String(rem);
  document.getElementById("artDrawerStatus").className="art-drawer-status";
  document.getElementById("artDrawerStatus").textContent="";
  stopPicking();
  document.getElementById("artDrawer").classList.add("open");
}

document.getElementById("addArtifactBtn").addEventListener("click", openArtDrawerCreate);
export function stopPicking(){
  state.pickingFor=null; state.hoverCell=null;
  for(const id of ["artPickBtn","agentPickBtn"]){
    const b=document.getElementById(id); if(b){b.textContent="⊕ Pick";b.classList.remove("on");}
  }
  document.getElementById("artPickHint").style.display="none";
  document.getElementById("agentPickHint").style.display="none";
  document.getElementById("grid-canvas").style.cursor="default";
  const banner=document.getElementById("pickBanner");
  if(banner){ banner.classList.add("hidden"); banner.classList.remove("pick-banner--artifact"); banner.innerHTML=""; }
  if(state.lastFrame) drawGrid(state.lastFrame.grid_state);
}

function startPicking(who){
  stopPicking();
  state.pickingFor=who;
  const btnId = who==="artifact"?"artPickBtn":"agentPickBtn";
  const hintId = who==="artifact"?"artPickHint":"agentPickHint";
  const btn=document.getElementById(btnId);
  btn.textContent="✕ Cancel"; btn.classList.add("on");
  document.getElementById(hintId).style.display="";
  document.getElementById("grid-canvas").style.cursor="crosshair";
  const banner=document.getElementById("pickBanner");
  if(banner){
    banner.classList.remove("hidden");
    banner.classList.toggle("pick-banner--artifact", who==="artifact");
    const what = who==="artifact" ? "artifact" : "agent";
    banner.innerHTML = `⊕ Click any cell to place the new ${what} — <kbd>Esc</kbd> to cancel`;
  }
}

document.addEventListener("keydown", (e)=>{
  if(e.key!=="Escape") return;
  // Picking takes precedence over drawers (it can co-exist with an open drawer).
  if(state.pickingFor){ stopPicking(); e.stopPropagation(); return; }
  // Otherwise, close whichever drawer is currently open.
  const drawer = document.getElementById("drawer");
  const artDrawer = document.getElementById("artDrawer");
  const editDrawer = document.getElementById("editDrawer");
  const settingsDrawer = document.getElementById("settingsDrawer");
  if(drawer.classList.contains("open")){ drawer.classList.remove("open"); return; }
  if(artDrawer.classList.contains("open")){
    artDrawer.classList.remove("open","edit-mode");
    _artEditMode=false; _artEditName=null;
    return;
  }
  if(editDrawer.classList.contains("open")){ closeEditDrawer(); return; }
  if(settingsDrawer?.classList.contains("open")){ settingsDrawer.classList.remove("open"); return; }
});

document.getElementById("artPickBtn").addEventListener("click",()=>{
  state.pickingFor==="artifact" ? stopPicking() : startPicking("artifact");
});
document.getElementById("agentPickBtn").addEventListener("click",()=>{
  state.pickingFor==="agent" ? stopPicking() : startPicking("agent");
});

document.getElementById("artDrawerCancelBtn").addEventListener("click",()=>{stopPicking();document.getElementById("artDrawer").classList.remove("open","edit-mode");_artEditMode=false;_artEditName=null;});

document.addEventListener("mousedown",(e)=>{
  if(state.pickingFor) return; // click is a grid-cell pick — don't close the drawer
  const drawer=document.getElementById("drawer");
  const artDrawer=document.getElementById("artDrawer");
  const editDrawer=document.getElementById("editDrawer");
  const settingsDrawer=document.getElementById("settingsDrawer");
  if(drawer.classList.contains("open")&&!drawer.contains(e.target)){stopPicking();drawer.classList.remove("open");}
  if(artDrawer.classList.contains("open")&&!artDrawer.contains(e.target)){stopPicking();artDrawer.classList.remove("open","edit-mode");_artEditMode=false;_artEditName=null;}
  if(editDrawer.classList.contains("open")&&!editDrawer.contains(e.target)) closeEditDrawer();
  // sdTopbarTrigger uses mousedown+stopPropagation so it never reaches this handler
  if(settingsDrawer?.classList.contains("open")&&!settingsDrawer.contains(e.target)) settingsDrawer.classList.remove("open");
});
document.getElementById("artDrawerCreateBtn").addEventListener("click",async()=>{
  const payload=document.getElementById("aPayload").value;
  const lifespan=parseInt(document.getElementById("aLifespan").value);
  const status=document.getElementById("artDrawerStatus");
  if(_artEditMode && _artEditName){
    status.className="art-drawer-status";status.textContent="Saving…";
    try{
      const res=await _wsRequest("modify_artifact",{name:_artEditName,payload,lifespan:isNaN(lifespan)?-1:lifespan});
      const newVer=(res?.version!=null)?res.version:(myArtifactVersions.get(_artEditName)??0)+1;
      myArtifactVersions.set(_artEditName,newVer); _saveMyArtifacts();
      status.className="art-drawer-status success";status.textContent="Saved.";
      setTimeout(()=>{document.getElementById("artDrawer").classList.remove("open","edit-mode");_artEditMode=false;_artEditName=null;status.className="art-drawer-status";status.textContent="";},1500);
    }catch(err){status.className="art-drawer-status error";status.textContent=err.message;}
    return;
  }
  const nameField=document.getElementById("aName");
  const name=nameField.value.trim();
  const row=parseInt(document.getElementById("aRow").value)||0;
  const col=parseInt(document.getElementById("aCol").value)||0;
  const movable=document.getElementById("aMovable").checked;
  if(!name){status.className="art-drawer-status error";status.textContent="Please fill in a name.";flagField(nameField);return;}
  status.className="art-drawer-status";status.textContent="Creating…";
  try{
    const created = await _wsRequest("create_artifact",{name,payload,row,col,lifespan:isNaN(lifespan)?-1:lifespan,movable});
    // The runner may have suffixed the name on collision ("X" -> "X_1"); the
    // ownership record is keyed on that final name, so we must track it too,
    // otherwise modify/destroy will fail authorization.
    const finalName = (created && created.name) || name;
    myArtifacts.add(finalName); myArtifactVersions.set(finalName,0); _saveMyArtifacts();
    status.className="art-drawer-status success";
    status.innerHTML=`◈ <strong>${escHtml(finalName)}</strong> created — will appear at the next timestep.`;
    document.getElementById("aName").value="";
    document.getElementById("aPayload").value="";
    setTimeout(()=>{stopPicking();document.getElementById("artDrawer").classList.remove("open");status.className="art-drawer-status";status.textContent="";},2000);
  }catch(err){status.className="art-drawer-status error";status.textContent=`${err.message}`;}
});

// ── Model list + provider detection ──────────────────────────────────────────
// Source of truth lives server-side in terralingua/utils/models.py and is served at
// /api/models. Fetched once at script load.
let MODEL_PROVIDERS = [];
const _modelsReady = fetch("/api/models")
  .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
  .then(data => { MODEL_PROVIDERS = data.providers || []; })
  .catch(err => { console.error("Failed to load model list:", err); MODEL_PROVIDERS = []; });

function providerFromKey(key){
  if(!key) return null;
  if(key.startsWith("sk-ant-")) return "anthropic";
  if(key.startsWith("sk-"))     return "openai";
  return null;
}

let _providerError = false;

function setProviderError(msg){
  _providerError = !!msg;
  const el = document.getElementById("dProviderError");
  if(msg){ el.textContent = msg; el.style.display = ""; }
  else   { el.textContent = ""; el.style.display = "none"; }
}

async function populateModelSelect(provider){
  await _modelsReady;
  const sel   = document.getElementById("dModel");
  const saved = localStorage.getItem("ogw-model");
  sel.innerHTML = "";
  const visible = provider
    ? MODEL_PROVIDERS.filter(p => p.id === provider)
    : MODEL_PROVIDERS;
  visible.forEach(p=>{
    const grp = document.createElement("optgroup");
    grp.label = p.label;
    p.models.forEach(m=>{
      const opt = document.createElement("option");
      opt.value = m.value;
      opt.textContent = m.label;
      grp.appendChild(opt);
    });
    sel.appendChild(grp);
  });
  if(saved && [...sel.options].some(o=>o.value===saved)) sel.value = saved;
}

// ── API key + model persistence ───────────────────────────────────────────────
(async function(){
  const keyEl = document.getElementById("dApiKey");

  if(!state.currentUser){
    const key = await loadApiKey();
    keyEl.value = key;
    keyEl.addEventListener("input", async function(){
      await saveApiKey(this.value);
      const p = providerFromKey(this.value);
      if(this.value && !p) setProviderError("Unsupported provider. Use an Anthropic (sk-ant-…) or OpenAI (sk-…) key.");
      else                 setProviderError(null);
      populateModelSelect(p);
    });
    const p = providerFromKey(key);
    if(key && !p) setProviderError("Unsupported provider. Use an Anthropic (sk-ant-…) or OpenAI (sk-…) key.");
    populateModelSelect(p);
  } else {
    document.getElementById("dApiKeyRow").classList.add("hidden");
    try{
      const {provider, has_key} = await _wsRequest("get_api_key_provider");
      if(has_key && !provider) setProviderError("Unsupported provider. Use an Anthropic (sk-ant-…) or OpenAI (sk-…) key.");
      populateModelSelect(provider);
    }catch{ populateModelSelect(null); }
  }

  document.getElementById("dModel").addEventListener("change", function(){
    localStorage.setItem("ogw-model", this.value);
  });
})();

// Mark a field as invalid (red border) and focus it. Field-error clears on
// the field's next input event (wired below, once per field).
function flagField(el){ el.classList.add("field-error"); el.focus(); }

document.getElementById("dName").addEventListener("input", function(){ this.classList.remove("field-error"); });
document.getElementById("dApiKey").addEventListener("input", function(){ this.classList.remove("field-error"); });
document.getElementById("aName").addEventListener("input", function(){ this.classList.remove("field-error"); });

document.getElementById("drawerConnectBtn").addEventListener("click",async()=>{
  const isHuman = state._humanAgentType==="remote_human";
  const nameField=document.getElementById("dName");
  const apiKeyField=document.getElementById("dApiKey");
  const name=nameField.value.trim();
  const motivation=isHuman?"":document.getElementById("dMotivation").value.trim();
  const status=document.getElementById("drawerStatus");
  if(!name){status.className="art-drawer-status error";status.textContent="Fill in agent name.";flagField(nameField);return;}
  if((state.lastFrame?.agents||[]).some(a=>a.name===name)){
    status.className="art-drawer-status error";status.textContent="An agent with this name already exists. Choose a different name.";flagField(nameField);return;
  }
  if(!isHuman){
    if(_providerError){status.className="art-drawer-status error";status.textContent="Fix the API key error before connecting.";flagField(apiKeyField);return;}
    if(!state.currentUser){
      const apiKey=apiKeyField.value.trim();
      if(!apiKey){status.className="art-drawer-status error";status.textContent="API key is required.";flagField(apiKeyField);return;}
    }
  }
  const rowVal=document.getElementById("dRow").value;
  const colVal=document.getElementById("dCol").value;
  const row=rowVal!==""?parseInt(rowVal):null;
  const col=colVal!==""?parseInt(colVal):null;
  const model=document.getElementById("dModel").value;
  const body={agent_name:name,motivation_prompt:motivation,agent_type:state._humanAgentType,model_info:model};
  if(!state.currentUser){
    body.anon_id = getAnonId();
    // Anonymous LLM agents: pass the user's key with the registration so the
    // server can spawn the bg loop without persisting it. Lives only in the
    // worker's RAM for the loop's lifetime.
    if(!isHuman) body.api_key = document.getElementById("dApiKey").value.trim();
  }
  if(row!==null&&col!==null){body.row=row;body.col=col;}
  const genomeType=document.getElementById("dGenomeType").value;
  body.genome_type=genomeType;
  const genomeData=collectGenomeData();
  if(genomeData) body.genome=genomeData;
  const excludedActions=collectExcludedActions();
  if(excludedActions) body.excluded_actions=excludedActions;
  status.className="art-drawer-status"; status.textContent="Registering…";
  try{
    const {agent_tag,token}=await _wsRequest("register",body);
    if(isHuman){
      agentStore.openHumanAgent(agent_tag,token,{genome_type:genomeType,genome:genomeData||null,excluded_actions:excludedActions||[]});
    } else {
      agentStore.trackAgent(agent_tag,token,{model,motivation,genome_type:genomeType,genome:genomeData||null,excluded_actions:excludedActions||[]});
    }
    status.className="art-drawer-status success";
    status.innerHTML=`◈ <strong>${escHtml(name)}</strong> connected — will appear at the next timestep.`;
    status.scrollIntoView({block:"nearest"});
    setTimeout(()=>{stopPicking();document.getElementById("drawer").classList.remove("open");status.className="art-drawer-status";status.textContent="";},2000);
  }catch(err){status.className="art-drawer-status error";status.textContent=err.message;status.scrollIntoView({block:"nearest"});}
});

// ── Edit-agent drawer ─────────────────────────────────────────────────────────
function renderEditGenomeFields(){
  const sec  = document.getElementById("eGenomeSection");
  const type = document.getElementById("eGenomeType").value;
  const randomBtn = document.getElementById("eGenomeRandomBtn");
  const fields = state.genomeInfo?.types?.[type]?.fields || [];
  randomBtn.style.display = type==="ocean_5" ? "" : "none";
  if(type==="no_traits" || !fields.length){
    sec.innerHTML=`<p style="font-size:0.72rem;color:var(--muted);margin-bottom:0.25rem;">No configurable traits.</p>`; return;
  }
  if(type==="sentence_mutate"){
    sec.innerHTML=`<input id="eGenomeSentence" type="text" placeholder="e.g. curious bold idealist…" />
      <p style="font-size:0.68rem;color:var(--muted);margin-top:0.3rem;line-height:1.4;">Each generation the text is slightly mutated — words may be added, changed, or removed.</p>`; return;
  }
  if(type==="sentence_directed"){
    sec.innerHTML=`<input id="eGenomeSentence" type="text" placeholder="e.g. you are an idealist who wants to change the world" />
      <p style="font-size:0.68rem;color:var(--muted);margin-top:0.3rem;line-height:1.4;">The parent agent writes a custom description for each offspring. If the parent passes an empty string, the offspring inherits a mutated version instead.</p>`; return;
  }
  if(type==="ocean_5"){
    const groups={};
    for(const f of fields){ const g=f.trait_type||"other"; (groups[g]||(groups[g]=[])).push(f); }
    let html="";
    for(const [gname,gfields] of Object.entries(groups)){
      html+=`<div class="genome-group-label">${gname}</div>`;
      for(const f of gfields){
        const [lo,hi]=f.range;
        html+=`<div class="genome-row">
          <div class="genome-row-hdr">
            <span class="genome-trait-name">${f.name.replace(/_/g," ")}</span>
            <span class="genome-trait-val" id="egv_${f.name}">${f.default.toFixed(2)}</span>
          </div>
          <input type="range" class="genome-slider" id="egr_${f.name}"
            min="${lo}" max="${hi}" step="0.01" value="${f.default}"
            oninput="document.getElementById('egv_${f.name}').textContent=parseFloat(this.value).toFixed(2)" />
          <p class="genome-trait-descr">${escHtml(f.descr)}</p>
        </div>`;
      }
    }
    sec.innerHTML=html;
  }
}

function applyEditGenomeData(genome_type, genome){
  // Fresh edit session — clear any stale cache from a previous agent's edit.
  for(const k of Object.keys(_editGenomeCache)) delete _editGenomeCache[k];
  _lastEditGenomeType = genome_type || "no_traits";
  document.getElementById("eGenomeType").value = _lastEditGenomeType;
  renderEditGenomeFields();
  if(!genome) return;
  if(genome_type==="sentence_mutate" && genome.words){
    const el=document.getElementById("eGenomeSentence"); if(el) el.value=genome.words.join(" ");
  } else if(genome_type==="sentence_directed" && genome.sentence!=null){
    const el=document.getElementById("eGenomeSentence"); if(el) el.value=genome.sentence;
  } else if(genome_type==="ocean_5"){
    for(const [k,v] of Object.entries(genome)){
      const sl=document.getElementById(`egr_${k}`), vl=document.getElementById(`egv_${k}`);
      if(sl){ sl.value=v; if(vl) vl.textContent=parseFloat(v).toFixed(2); }
    }
  }
}

function collectEditGenomeData(){
  const type=document.getElementById("eGenomeType").value;
  const fields=state.genomeInfo?.types?.[type]?.fields||[];
  if(type==="no_traits"||!fields.length) return {type:"no_traits",genome:null};
  if(type==="sentence_mutate"){
    const raw=document.getElementById("eGenomeSentence")?.value?.trim();
    return {type, genome: raw ? {words: raw.split(/\s+/).filter(Boolean)} : null};
  }
  if(type==="sentence_directed"){
    const sentence=document.getElementById("eGenomeSentence")?.value?.trim();
    return {type, genome: sentence ? {sentence} : null};
  }
  if(type==="ocean_5"){
    const data={};
    for(const f of fields){ const el=document.getElementById(`egr_${f.name}`); data[f.name]=el?parseFloat(el.value):f.default; }
    return {type,genome:data};
  }
  return {type:"no_traits",genome:null};
}

function renderEditActionChips(excludedActions){
  const sec = document.getElementById("eActionsSection");
  if(!state.actionsInfo.length){ sec.innerHTML=""; return; }
  const excluded=new Set(excludedActions||[]);
  sec.innerHTML=`<div class="action-chips">${
    state.actionsInfo.map(a=>`<span class="action-chip${excluded.has(a.name)?"":" on"}" data-action="${escHtml(a.name)}" title="${escHtml(a.description)}">${escHtml(a.name.replace(/_/g," "))}</span>`).join("")
  }</div>`;
  sec.querySelectorAll(".action-chip").forEach(chip=>{
    chip.addEventListener("click",()=>chip.classList.toggle("on"));
  });
}

function collectEditExcludedActions(){
  const excluded=[];
  document.querySelectorAll("#eActionsSection .action-chip").forEach(c=>{
    if(!c.classList.contains("on")) excluded.push(c.dataset.action);
  });
  return excluded;
}

async function openEditDrawer(tag){
  state.editingTag=tag;
  if(!state.genomeInfo) await loadGenomeInfo();
  if(!state.actionsInfo.length) await loadActionsInfo();
  document.getElementById("editDrawerTitle").textContent=`✎ Edit Agent — ${agentName(tag)}`;
  const data=agentStore.get(tag)||{};
  document.getElementById("eMotivation").value=data.motivation||state.tagToAgent.get(tag)?.motivation||"";
  applyEditGenomeData(data.genome_type||"no_traits", data.genome);
  renderEditActionChips(data.excluded_actions||[]);
  document.getElementById("editDrawerStatus").className="art-drawer-status";
  document.getElementById("editDrawerStatus").textContent="";
  stopPicking();
  document.getElementById("drawer").classList.remove("open");
  document.getElementById("artDrawer").classList.remove("open");
  document.getElementById("editDrawer").classList.add("open");
}

function closeEditDrawer(){
  document.getElementById("editDrawer").classList.remove("open");
  state.editingTag=null;
}

document.getElementById("eGenomeType").addEventListener("change", function(){
  _captureEditGenome(_lastEditGenomeType);
  renderEditGenomeFields();
  const newType = this.value;
  _restoreEditGenome(newType);
  _lastEditGenomeType = newType;
});
document.getElementById("eGenomeRandomBtn").addEventListener("click",()=>{
  const fields=state.genomeInfo?.types?.ocean_5?.fields||[];
  for(const f of fields){
    const [lo,hi]=f.range, val=Math.random()*(hi-lo)+lo;
    const sl=document.getElementById(`egr_${f.name}`), vl=document.getElementById(`egv_${f.name}`);
    if(sl){ sl.value=val; if(vl) vl.textContent=val.toFixed(2); }
  }
});
document.getElementById("eActionsAllBtn").addEventListener("click",()=>{
  document.querySelectorAll("#eActionsSection .action-chip").forEach(c=>c.classList.add("on"));
});
document.getElementById("eActionsNoneBtn").addEventListener("click",()=>{
  document.querySelectorAll("#eActionsSection .action-chip").forEach(c=>c.classList.remove("on"));
});
document.getElementById("editDrawerCancelBtn").addEventListener("click", closeEditDrawer);
document.getElementById("editDrawerSaveBtn").addEventListener("click", async()=>{
  if(!state.editingTag) return;
  const data=agentStore.get(state.editingTag); if(!data) return;
  const motivation=document.getElementById("eMotivation").value.trim();
  const {type:genome_type, genome}=collectEditGenomeData();
  const excluded_actions=collectEditExcludedActions();
  const status=document.getElementById("editDrawerStatus");
  status.className="art-drawer-status"; status.textContent="Saving…";
  try{
    const body={token:data.token, motivation_prompt:motivation, excluded_actions, genome_type};
    if(genome) body.genome=genome;
    await _wsRequest("update_agent",{tag:state.editingTag,...body});
    // Update local cache so next openEditDrawer pre-fills correctly
    agentStore.update(state.editingTag,{ motivation, genome_type, genome:genome||null, excluded_actions });
    status.className="art-drawer-status success"; status.textContent="Saved — takes effect next step.";
    setTimeout(closeEditDrawer, 1500);
  }catch(err){ status.className="art-drawer-status error"; status.textContent=err.message; }
});

document.getElementById("followEditBtn").addEventListener("click",()=>{ if(state.selectedTag) openEditDrawer(state.selectedTag); });
document.getElementById("followKillBtn").addEventListener("click",async()=>{
  if(!state.selectedTag) return;
  const data=agentStore.get(state.selectedTag); if(!data) return;
  if(!await _confirm(`Kill agent ${agentName(state.selectedTag)}? It will die on the next simulation step.`, "Kill", "Cancel")) return;
  try{
    await _wsRequest("kill_agent",{tag:state.selectedTag,token:data.token});
    _deselectAgent();
  }catch(err){ await _confirm(`Kill failed: ${err.message}`, "OK", ""); }
});
