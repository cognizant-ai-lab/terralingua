import { state, allMessages } from './state.js';

export function accumulateMessages(groups){
  for(const g of groups){
    if(g.step>state.lastMsgStep){ allMessages.push(g); state.lastMsgStep=g.step; }
  }
  if(allMessages.length>200) allMessages.splice(0, allMessages.length-200);
}

export function renderMessages(){
  const list=document.getElementById("msgList");
  const wasAtTop=list.scrollTop<30;
  list.innerHTML="";
  const agentData=state.lastFrame?.agents.find(a=>a.tag===state.selectedTag);
  for(let i=allMessages.length-1;i>=0;i--){
    const group=allMessages[i];
    const msgs=group.messages;
    if(!msgs.length) continue;
    const recipients=group.recipients||[];
    const rendered=[];
    for(let j=0;j<msgs.length;j++){
      const msg=msgs[j];
      const isMine=agentData&&msg.startsWith(agentData.name+":");
      if(agentData && !isMine){
        const recv=recipients[j];
        if(!recv || !recv.includes(state.selectedTag)) continue;
      }
      rendered.push({msg, isMine});
    }
    if(!rendered.length) continue;
    const msgOffset=Math.max(state.currentStep-1,state.lastMsgStep)-group.step;
    const hdr=document.createElement("div"); hdr.className="msg-hdr"; hdr.textContent=msgOffset===0?"— now —":`— ${msgOffset} step${msgOffset===1?"":"s"} ago —`; list.appendChild(hdr);
    for(const {msg, isMine} of rendered){
      const el=document.createElement("div"); el.className="msg-line";
      if(isMine) el.classList.add("mine");
      const colon=msg.indexOf(":");
      if(colon>0){
        const b=document.createElement("b"); b.className="msg-sender"; b.textContent=msg.slice(0,colon+1);
        el.appendChild(b); el.appendChild(document.createTextNode(msg.slice(colon+1)));
      } else { el.textContent=msg; }
      list.appendChild(el);
    }
  }
  if(!list.children.length){
    const empty=document.createElement("div"); empty.className="empty-note"; empty.textContent="No messages to/from agent yet.";
    list.appendChild(empty);
  }
  if(wasAtTop) list.scrollTop=0;
}
