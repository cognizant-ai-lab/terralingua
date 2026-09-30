import { state, agentHistory, familyTags, agentName, escHtml, seenAgentNames } from './state.js';
import { snapshotElement } from './screenshot.js';
import { agentStore } from './agents.js';
import { dirArrow, selectAgent } from './roster.js';
import { wsRequest } from './ws.js';
import { openArtDetail } from './artifacts.js';
import { renderAnalysisInto } from './anthropologist.js';

function _renderRadar(genome){
  const entries=Object.entries(genome);
  const n=entries.length;
  if(!n) return "";
  const cx=110,cy=110,R=75;
  const angle=i=>-Math.PI/2+i*(2*Math.PI/n);
  let grid="",axes="",labels="";
  for(const r of [0.25,0.5,0.75,1.0]){
    grid+=`<circle cx="${cx}" cy="${cy}" r="${R*r}" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>`;
  }
  for(let i=0;i<n;i++){
    const a=angle(i);
    const x=(cx+R*Math.cos(a)).toFixed(1), y=(cy+R*Math.sin(a)).toFixed(1);
    axes+=`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>`;
    const lx=(cx+(R+16)*Math.cos(a)).toFixed(1), ly=(cy+(R+16)*Math.sin(a)).toFixed(1);
    const ca=Math.cos(a), anchor=ca>0.1?"start":ca<-0.1?"end":"middle";
    labels+=`<text x="${lx}" y="${ly}" text-anchor="${anchor}" dominant-baseline="middle" font-size="8.5" fill="rgba(255,255,255,0.45)">${escHtml(entries[i][0].replace(/_/g," "))}</text>`;
  }
  const pts=entries.map(([,v],i)=>{
    const r=((Number(v)+1)/2)*R, a=angle(i);
    return `${(cx+r*Math.cos(a)).toFixed(1)},${(cy+r*Math.sin(a)).toFixed(1)}`;
  }).join(" ");
  const poly=`<polygon points="${pts}" fill="rgba(167,139,250,0.18)" stroke="rgba(167,139,250,0.85)" stroke-width="1.5" stroke-linejoin="round"/>`;
  return `<svg viewBox="0 0 220 220" width="100%" style="max-width:190px;display:block;margin:0.25rem auto;">${grid}${axes}${poly}${labels}</svg>`;
}

export function _computeFamilyTags(tag, agents) {
  const liveSet = new Set(agents.map(a => a.tag));
  // Use full genealogy so we can traverse through dead intermediaries
  const parentOf = new Map(Object.entries(state.lastFrame?.genealogy || {}));
  for(const a of agents) if(a.parent_tag) parentOf.set(a.tag, a.parent_tag);
  const childrenOf = new Map();
  for (const [child, parent] of parentOf) {
    if (!childrenOf.has(parent)) childrenOf.set(parent, []);
    childrenOf.get(parent).push(child);
  }
  const result = new Set();
  // live ancestors
  let cur = parentOf.get(tag);
  while (cur) { if(liveSet.has(cur)) result.add(cur); cur = parentOf.get(cur); }
  // live descendants (BFS through full tree, including dead intermediaries)
  const queue = [tag];
  const visited = new Set([tag]);
  while (queue.length) {
    const t = queue.shift();
    for (const child of (childrenOf.get(t) || [])) {
      if (!visited.has(child)) {
        visited.add(child);
        queue.push(child);
        if(liveSet.has(child)) result.add(child);
      }
    }
  }
  return result;
}

function formatAction(action, params){
  if(!action) return "-";
  params=params||{};
  if(action==="modify_artifact")
    return params.name?`Modify Artifact ${params.name}`:"Modify Artifact";
  if(action==="destroy_artifact")
    return params.name?`Destroy Artifact ${params.name}`:"Destroy Artifact";
  if(action.startsWith("modify_artifact_"))
    return `Modify Artifact ${action.slice("modify_artifact_".length)}`;
  if(action.startsWith("destroy_artifact_"))
    return `Destroy Artifact ${action.slice("destroy_artifact_".length)}`;
  switch(action){
    case "move":
      if(params.direction==="stay") return "Stay";
      return params.direction?`Move ${dirArrow(params.direction)}`:"Move";
    case "create_artifact":  return params.name?`Create Artifact ${params.name}`:"Create Artifact";
    case "pickup_artifact":  return params.name?`Pick Up Artifact ${params.name}`:"Pick Up Artifact";
    case "drop_artifact":    return params.name?`Drop Artifact ${params.name}`:"Drop Artifact";
    case "give_artifact": {
      const artName=params.artifact_name||params.name;
      let s=artName?`Give Artifact ${artName}`:"Give Artifact";
      if(params.target_agent) s+=` to ${params.target_agent}`;
      return s;
    }
    case "give": {
      const gParts=["Give Energy"];
      if(params.amount) gParts.push(params.amount);
      if(params.target) gParts.push(`to ${params.target}`);
      return gParts.join(" ");
    }
    case "take": {
      const tParts=["Take Energy"];
      if(params.amount) tParts.push(params.amount);
      if(params.target) tParts.push(`from ${params.target}`);
      return tParts.join(" ");
    }
    case "reproduce":        return "Reproduce";
    case "set_color":        return params.color?`Set Color ${params.color}`:"Set Color";
    case "send_direct_message": return params.target?`Send Direct Message to ${params.target}`:"Send Direct Message";
    case "follow":              return params.target?`Follow ${params.target}`:"Follow";
    case "request_connection":  return params.target?`Request Connection to ${params.target}`:"Request Connection";
    case "accept_connection":   return params.target?`Accept Connection from ${params.target}`:"Accept Connection";
    case "reject_connection":   return params.target?`Reject Connection from ${params.target}`:"Reject Connection";
    case "disconnect":          return params.target?`Disconnect from ${params.target}`:"Disconnect";
    default: return action.split("_").map(w=>w.charAt(0).toUpperCase()+w.slice(1)).join(" ");
  }
}

export function renderHistoryList(){
  const hist=(agentHistory.get(state.selectedTag)||[]).slice().reverse();
  const list=document.getElementById("historyList");
  const atTop=list.scrollTop<10;
  list.innerHTML="";
  if(!hist.length){
    const empty=document.createElement("div"); empty.className="empty-note"; empty.textContent="No agent logs yet.";
    list.appendChild(empty); return;
  }
  for(const e of hist){
    const act=formatAction(e.action,e.params);
    const el=document.createElement("div"); el.className="hist-entry";
    const offset=Math.max(state.currentStep-1,state.historyLastStep)-e.step;
    const stepLabel=offset===0?"now":`${offset} step${offset===1?"":"s"} ago`;
    let html=`<div class="hist-step-hdr">${stepLabel}&nbsp;&nbsp;⚡ ${e.energy??"-"}&nbsp;&nbsp;⏱ ${e.time_left??"-"}</div>`;
    html+=`<div class="hist-action">${escHtml(act)}</div>`;
    if(e.action==="send_direct_message"&&e.params&&e.params.content)
      html+=`<div class="hist-dm">"${escHtml(e.params.content)}"</div>`;
    if(e.message){
      html+=`<div class="hist-msg-label">Broadcast</div>`;
      html+=`<div class="hist-msg">"${escHtml(e.message)}"</div>`;
    }
    if(e.inventory&&e.inventory.length) html+=`<div class="hist-inv">${e.inventory.map(i=>{
      const name=i.replace(/^A\([^)]+\):\s*/,"");
      return `<button class="inv-tag" data-art-name="${escHtml(name)}">${escHtml(name)}</button>`;
    }).join("")}</div>`;
    if(e.memory){
      const mem=typeof e.memory==="string"?e.memory:JSON.stringify(e.memory,null,2);
      html+=`<div class="hist-mem">${escHtml(mem)}</div>`;
    }
    el.innerHTML=html;
    el.querySelectorAll(".inv-tag[data-art-name]").forEach(btn=>{
      btn.addEventListener("click",()=>{
        const art=(state.lastFrame?.artifacts||[]).find(a=>a.name===btn.dataset.artName);
        if(art) openArtDetail(art);
      });
    });
    list.appendChild(el);
  }
  if(atTop) list.scrollTop=0;
}

export function renderAnalysis(){
  const tag = state.selectedTag;
  if(!tag) return;
  const agents = state.lastFrame?.agents || [];
  const ag = agents.find(a => a.tag === tag);
  const name = ag?.name || agentName(tag) || tag;
  renderAnalysisInto(document.getElementById("historyList"), tag, !!ag, name);
}

export function updateFollowDetail(agents){
  if(!state.selectedTag) return;
  let ag=agents.find(a=>a.tag===state.selectedTag);
  const isDead=!ag;
  if(isDead) ag=(state.lastFrame?.dead_agents||[]).find(a=>a.tag===state.selectedTag);
  const name=ag?.name||agentName(state.selectedTag)||state.selectedTag;
  document.getElementById("followTitle").textContent=`Following — ${name}${isDead?" (dead)":""}`;
  document.getElementById("followEmpty").style.display="none";
  document.getElementById("followDetail").style.display="flex";
  const followSnapshotBtn = document.getElementById("followSnapshotBtn");
  followSnapshotBtn.style.display = "";
  followSnapshotBtn.onclick = () => snapshotElement(document.getElementById("followSection"), followSnapshotBtn);

  const analysisBtn=document.getElementById("analysisBtn");
  analysisBtn.style.display="";
  analysisBtn.textContent=isDead?"Obituary":"Analysis";

  // Show Family button; keep family tags current on each frame
  document.getElementById("familyBtn").style.display="";
  if (state.familyHighlight) {
    familyTags.clear();
    for (const t of _computeFamilyTags(state.selectedTag, agents)) familyTags.add(t);
  }

  // Top: latest stats
  if(isDead){
    document.getElementById("fEnergy").textContent="—";
    document.getElementById("fTime").textContent="—";
    document.getElementById("fType").textContent="dead";
    document.getElementById("followAgentBtns").style.display="none";
  } else {
    document.getElementById("fEnergy").textContent=ag.energy??"-";
    document.getElementById("fTime").textContent=ag.time_left??"-";
    const typeLabel=ag.type==="remote_human"?"Human":ag.type==="remote_llm"?"LLM":ag.type.replace("Agent","");
    document.getElementById("fType").textContent=typeLabel;
    document.getElementById("followAgentBtns").style.display=agentStore.has(state.selectedTag)?"flex":"none";
  }

  // Build traits content (same code path for alive and dead)
  const motBlock=document.getElementById("fMotivation");
  const hasMot=!!ag?.motivation;
  if(hasMot) document.getElementById("fMotivationText").textContent=ag.motivation;

  const genomeBlock=document.getElementById("fGenome");
  const genomeEl=document.getElementById("fGenomeText");
  let genomeHtml="";
  if(ag?.genome_type==="sentence_mutate"&&ag.genome?.words?.length){
    genomeHtml=ag.genome.words.map(w=>`<span class="genome-chip">${escHtml(w)}</span>`).join("");
  } else if(ag?.genome_type==="sentence_directed"&&ag.genome?.sentence){
    genomeHtml=`<span class="detail-motivation">${escHtml(ag.genome.sentence)}</span>`;
  } else if(ag?.genome_type==="ocean_5"&&ag.genome){
    genomeHtml=_renderRadar(ag.genome);
  }
  if(genomeHtml) genomeEl.innerHTML=genomeHtml;
  const hasGenome=!!genomeHtml;

  // Show Traits toggle only if there's something to show
  const traitsBtn=document.getElementById("traitsBtn");
  const traitsPanel=document.getElementById("fTraitsPanel");
  traitsBtn.style.display=(hasMot||hasGenome)?"":"none";
  traitsBtn.classList.toggle("on", state.traitsExpanded);
  traitsPanel.style.display=state.traitsExpanded?"":"none";
  motBlock.style.display=(hasMot&&state.traitsExpanded)?"":"none";
  genomeBlock.style.display=(hasGenome&&state.traitsExpanded)?"":"none";
  if(state.analysisMode) renderAnalysis();
  else if(state.familyHighlight) renderFamilyTree();
  else renderHistoryList();
}

function _buildFamilyTree(list){
  if(!state.lastFrame||!state.selectedTag) return;
  const agents=state.lastFrame.agents;
  const agByTag=new Map(agents.map(a=>[a.tag,a]));
  // Use full genealogy (all agents ever, live + dead)
  const genealogy=state.lastFrame.genealogy||{};
  const parentOf=new Map(Object.entries(genealogy));
  for(const a of agents) if(a.parent_tag) parentOf.set(a.tag,a.parent_tag);
  const childrenOf=new Map();
  for(const [child,parent] of parentOf){
    if(!childrenOf.has(parent)) childrenOf.set(parent,[]);
    childrenOf.get(parent).push(child);
  }

  // Build direct ancestor chain: [root, ..., grandparent, parent]
  const ancestors=[];
  let cur=state.selectedTag;
  const visited=new Set([state.selectedTag]);
  while(parentOf.has(cur)){
    cur=parentOf.get(cur);
    if(visited.has(cur)) break; // cycle guard
    visited.add(cur);
    ancestors.unshift(cur);
  }
  const N=ancestors.length;

  function mkRow(tag,prefix,connector){
    const ag=agByTag.get(tag);
    const isDead=!ag;
    const name=ag?.name||agentName(tag)||tag;
    const isSelf=tag===state.selectedTag;
    const row=document.createElement("div");
    row.className="family-tree-row"+(isSelf?" family-tree-self":isDead?" family-tree-dead":"");
    const stats=ag?`<span class="family-tree-stat">⚡${ag.energy??"-"}</span>`
                  :`<span class="family-tree-stat">dead</span>`;
    row.innerHTML=`<span class="family-tree-pre">${escHtml(prefix+connector)}</span>`+
      `<button class="family-tree-name" data-tag="${escHtml(tag)}">${escHtml(name)}</button>${stats}`;
    row.querySelector(".family-tree-name").addEventListener("click",()=>selectAgent(tag));
    list.appendChild(row);
  }

  function renderDescendants(tag,prefix){
    const kids=childrenOf.get(tag)||[];
    for(let i=0;i<kids.length;i++){
      const isLast=i===kids.length-1;
      mkRow(kids[i],prefix,isLast?"└─ ":"├─ ");
      renderDescendants(kids[i],prefix+(isLast?"   ":"│  "));
    }
  }

  // Render ancestor chain as a straight line (direct lineage only, no siblings)
  // Position p: prefix="   ".repeat(p-1), connector="└─ " (root has neither)
  for(let i=0;i<N;i++)
    mkRow(ancestors[i], i===0?"":" ".repeat(3*(i-1)), i===0?"":"└─ ");
  // Render selected tag
  mkRow(state.selectedTag, N===0?"":" ".repeat(3*(N-1)), N===0?"":"└─ ");
  // Render all descendants from state.selectedTag downward
  renderDescendants(state.selectedTag," ".repeat(3*N));
}

export function renderFamilyTree(){
  const list=document.getElementById("historyList");
  list.innerHTML="";
  if(!state.lastFrame||!state.selectedTag) return;
  _buildFamilyTree(list);
  // Resolve names for any unknown tags then re-render
  const allTags=new Set();
  list.querySelectorAll(".family-tree-name[data-tag]").forEach(b=>allTags.add(b.dataset.tag));
  const unknown=[...allTags].filter(t=>!seenAgentNames.has(t)&&!state.tagToAgent.has(t));
  if(unknown.length){
    wsRequest("get_agent_names",{tags:unknown}).then(result=>{
      for(const [t,n] of Object.entries(result)) seenAgentNames.set(t,n);
      if(state.familyHighlight&&state.selectedTag){
        const list2=document.getElementById("historyList");
        list2.innerHTML="";
        _buildFamilyTree(list2);
      }
    }).catch(()=>{});
  }
}
