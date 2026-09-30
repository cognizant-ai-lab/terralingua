import { state } from './state.js';
import { drawGrid, getGraphNodePositions } from './grid.js';
import { canonicalSnapshots, parseRecording, stableLayout, displayEdges, graphChanges } from './graph_replay_data.js';

export function initGraphReplay(wsRequest) {
  const el = id => document.getElementById(id);
  const pane = el('graphReplayPane');
  const canvas = el('grid-canvas');
  const liveLegend = el('gridLegend').innerHTML;
  let snapshots = [], layout = {}, index = 0, timer = null, loading = false;
  let exportJob = null, source = '', loadedRun = null;
  const status = text => { el('graphReplayStatus').textContent = text; };
  const speed = () => Number(el('graphReplaySpeed').value);
  const draw = () => drawGrid(state.lastFrame?.grid_state);

  function syncControls() {
    const active = !!state.graphReplay;
    const busy = loading || !!exportJob;
    el('graphReplayPrev').disabled = !snapshots.length || busy || (active && index === 0);
    el('graphReplayNext').disabled = !snapshots.length || busy || (active && index === snapshots.length - 1);
    el('graphReplayPlay').disabled = !snapshots.length || busy;
    el('graphReplaySeek').disabled = !snapshots.length || busy;
    el('graphReplaySeek').max = Math.max(0, snapshots.length - 1);
    el('graphReplaySeek').value = index;
    el('graphReplayPlay').textContent = timer ? 'Pause' : 'Play';
    el('graphReplayLive').disabled = !active || busy;
    el('graphReplayLoad').disabled = busy;
    el('graphReplayOpen').disabled = busy;
    el('graphReplaySave').disabled = !snapshots.length || busy;
    el('graphReplaySpeed').disabled = !!exportJob;
    el('graphReplayExport').disabled = !snapshots.length || loading;
    el('graphReplayExport').textContent = exportJob ? 'Cancel export' : 'Export video';
    el('graphReplayMode').textContent = active ? 'Recorded graph' : 'Live graph';
    if (active) el('gridLegend').textContent = 'Dot = followee · dots at both ends = mutual · dashed = pending';
    else el('gridLegend').innerHTML = liveLegend;
    if (!active) {
      el('graphReplayStep').textContent = snapshots.length ? 'Live graph. The recording remains loaded.' : 'No recording loaded.';
      for (const id of ['graphReplayCounts', 'graphReplayChanges', 'graphReplayNode']) el(id).textContent = '';
    }
    el('graphReplayNote').textContent = active
      ? 'Graph replay only. The other panels and top counters remain live.'
      : 'Load a recording to inspect each step. This does not pause the simulation.';
  }

  function pause() {
    clearInterval(timer);
    timer = null;
    syncControls();
  }

  function nodeInfo(node) {
    if (!state.graphReplay) return;
    state.graphReplay.selectedNode = node;
    const snapshot = snapshots[index], graph = snapshot.graph;
    const agent = graph.agents.find(a => a.node === node);
    const incoming = graph.edges.filter(e => e.type === 'follow' && e.target === node).length;
    const outgoing = graph.edges.filter(e => e.type === 'follow' && e.source === node).length;
    el('graphReplayNode').textContent = node == null ? 'Click a node for its recorded details.' :
      `${agent?.name || node} · node ${node} · ${incoming} followers · follows ${outgoing}` +
      (agent?.energy != null ? ` · energy ${agent.energy}` : '') +
      (agent?.time_left != null ? ` · time left ${agent.time_left}` : '') +
      (agent?.role ? ` · role ${agent.role}` : '');
    draw();
  }

  function show(next) {
    if (!snapshots.length) return;
    index = Math.max(0, Math.min(snapshots.length - 1, next));
    const snapshot = snapshots[index], graph = snapshot.graph;
    const selected = state.graphReplay?.selectedNode;
    const follows = graph.edges.filter(e => e.type === 'follow').length;
    const pending = graph.edges.filter(e => e.type === 'pending').length;
    state.graphReplay = { step: snapshot.step, runId: snapshot.run_id,
      selectedNode: graph.nodes.includes(selected) ? selected : null,
      graph: { ...graph, env_type: 'graph', edges: displayEdges(graph.edges), node_layout: layout,
        layout_bounds: [-1, -1, 1, 1] } };
    el('graphReplayStep').textContent = `Step ${snapshot.step} · ${index + 1}/${snapshots.length} snapshots`;
    el('graphReplayCounts').textContent = `${graph.agents.length} agents · ${follows} directed links · ${pending} pending`;
    const changes = graphChanges(snapshots[index - 1]?.graph, graph);
    el('graphReplayChanges').textContent = index === 0 ? 'First recorded state.' :
      `Since previous snapshot: ${changes.born} nodes born, ${changes.died} gone; ${changes.added} links added, ${changes.removed} removed.`;
    nodeInfo(state.graphReplay.selectedNode);
    syncControls();
  }

  function play() {
    if (timer) { pause(); return; }
    if (!snapshots.length || loading || exportJob) return;
    if (index === snapshots.length - 1) index = 0;
    show(index);
    timer = setInterval(() => {
      if (index >= snapshots.length - 1) { pause(); return; }
      show(index + 1);
    }, 1000 / speed());
    syncControls();
  }

  function install(records, label, warning = '') {
    const next = canonicalSnapshots(records);
    if (!next.length) throw new Error('No graph snapshots were recorded for this run.');
    pause();
    snapshots = next;
    layout = stableLayout(snapshots);
    index = 0;
    source = label;
    loadedRun = snapshots.at(-1).run_id;
    state.mapZoom = 1; state.mapPanX = 0; state.mapPanY = 0;
    pane.hidden = false;
    pane.open = true;
    show(0);
    status(`${source} · ${snapshots.length} snapshots loaded.${warning ? ' ' + warning : ''}`);
  }

  async function loadCurrent() {
    pause(); loading = true; syncControls(); status('Loading the current run…');
    try {
      const first = await wsRequest('get_social_graph_history', { offset: 0, limit: 100 });
      if (!first.available) throw new Error(first.message || 'This run has no social graph recording.');
      const records = [...first.snapshots];
      const total = first.total;
      let offset = first.next_offset;
      while (records.length < total && offset != null) {
        const page = await wsRequest('get_social_graph_history', {
          offset, limit: Math.min(100, total - records.length),
          run_id: first.run_id, segment_id: first.segment_id,
        });
        if (page.run_id !== first.run_id || page.segment_id !== first.segment_id || !page.available) {
          throw new Error('The run changed while loading. Load the recording again.');
        }
        if (!page.snapshots.length || (page.next_offset != null && page.next_offset <= offset)) {
          throw new Error('The recording changed while loading. Load it again.');
        }
        records.push(...page.snapshots);
        offset = page.next_offset;
        status(`Loading snapshots: ${Math.min(records.length, total)}/${total}…`);
      }
      if (records.length < total) throw new Error('The recording changed while loading. Load it again.');
      install(records.slice(0, total), `Run ${first.run_id}`);
    } catch (error) { status(error.message); }
    finally { loading = false; syncControls(); }
  }

  function download(blob, name) {
    const url = URL.createObjectURL(blob), anchor = document.createElement('a');
    anchor.href = url; anchor.download = name; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  async function exportVideo() {
    if (exportJob) { exportJob.cancelled = true; exportJob.wake?.(); return; }
    if (!snapshots.length) return;
    if (!canvas.captureStream || !window.MediaRecorder) {
      status('Video export is unavailable in this browser. Try Chrome or Edge.'); return;
    }
    const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']
      .find(type => MediaRecorder.isTypeSupported(type));
    if (!mime) { status('This browser has no supported video encoder. Try Chrome or Edge.'); return; }
    pause();
    const savedIndex = index, savedReplay = state.graphReplay;
    const job = { cancelled: false, wake: null };
    exportJob = job; syncControls();
    let stream, recorder, finish, recordingError = null;
    try {
      // A fixed export canvas keeps video dimensions stable during window resizing.
      const output = document.createElement('canvas'); output.width = 1000; output.height = 1000;
      const paint = () => {
        const panX = state.mapPanX, panY = state.mapPanY;
        try {
          const scale = output.width / canvas.width;
          state.mapPanX *= scale; state.mapPanY *= scale;
          drawGrid(undefined, output);
        } finally { state.mapPanX = panX; state.mapPanY = panY; }
      };
      show(0); paint();
      stream = output.captureStream(30);
      recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 5000000 });
      const chunks = [];
      finish = new Promise(resolve => { recorder.onstop = resolve; });
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = event => { recordingError = event.error || new Error('Video recording failed.'); job.cancelled = true; job.wake?.(); };
      recorder.start();
      for (let i = 0; i < snapshots.length && !job.cancelled; i++) {
        show(i); paint();
        status(`Exporting step ${snapshots[i].step} (${i + 1}/${snapshots.length}). Keep this tab visible.`);
        await new Promise(resolve => {
          const timeout = setTimeout(resolve, 1000 / speed());
          job.wake = () => { clearTimeout(timeout); resolve(); };
        });
      }
      if (recorder.state !== 'inactive') recorder.stop();
      await finish;
      if (recordingError) throw recordingError;
      if (job.cancelled) status('Video export canceled.');
      else if (!chunks.length) throw new Error('The browser produced no video data.');
      else {
        download(new Blob(chunks, { type: mime }), `social_graph.${mime.startsWith('video/mp4') ? 'mp4' : 'webm'}`);
        status(`Video exported at ${speed()} steps per second.`);
      }
    } catch (error) { status(`Video export failed: ${error.message}`); }
    finally {
      if (recorder && recorder.state !== 'inactive') recorder.stop();
      stream?.getTracks().forEach(track => track.stop());
      exportJob = null;
      index = savedIndex;
      if (savedReplay) { state.graphReplay = savedReplay; show(savedIndex); }
      else { state.graphReplay = null; draw(); }
      syncControls();
    }
  }

  el('graphReplayLoad').addEventListener('click', loadCurrent);
  el('graphReplayOpen').addEventListener('click', () => el('graphReplayFile').click());
  el('graphReplayFile').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    pause(); loading = true; syncControls();
    try {
      const { snapshots: records, ignoredTail } = parseRecording(await file.text());
      install(records, file.name, ignoredTail ? 'The incomplete last line was ignored.' : '');
    } catch (error) { status(error.message); }
    finally { event.target.value = ''; loading = false; syncControls(); }
  });
  el('graphReplaySave').addEventListener('click', () => {
    download(new Blob([snapshots.map(s => JSON.stringify(s)).join('\n') + '\n'], { type: 'application/x-ndjson' }), 'social_graph.jsonl');
  });
  el('graphReplayPlay').addEventListener('click', play);
  el('graphReplayPrev').addEventListener('click', () => { pause(); show(index - 1); });
  el('graphReplayNext').addEventListener('click', () => { pause(); show(index + 1); });
  el('graphReplaySeek').addEventListener('input', event => { const value = Number(event.target.value); pause(); show(value); });
  el('graphReplaySpeed').addEventListener('change', () => { if (timer) { pause(); play(); } });
  el('graphReplayLive').addEventListener('click', () => {
    pause(); state.graphReplay = null;
    state.mapZoom = 1; state.mapPanX = 0; state.mapPanY = 0;
    draw(); syncControls(); status(state.lastFrame ? 'Showing the latest live graph.' : 'Waiting for a live frame. The recording remains loaded.');
  });
  el('graphReplayExport').addEventListener('click', exportVideo);
  pane.addEventListener('keydown', event => {
    if (['ArrowUp', 'ArrowDown'].includes(event.key)) event.stopPropagation();
    if (!snapshots.length || loading || exportJob || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'SUMMARY'].includes(event.target.tagName)) return;
    if (['ArrowLeft', 'ArrowRight', ' '].includes(event.key)) {
      event.preventDefault(); event.stopPropagation();
      if (event.key === ' ') play();
      else { pause(); show(index + (event.key === 'ArrowLeft' ? -1 : 1)); }
    }
  });
  canvas.addEventListener('click', event => {
    if (!state.graphReplay) return;
    event.stopImmediatePropagation();
    if (state.panHasMoved) { state.panHasMoved = false; return; }
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) * canvas.width / rect.width - state.mapPanX) / state.mapZoom;
    const y = ((event.clientY - rect.top) * canvas.height / rect.height - state.mapPanY) / state.mapZoom;
    const positions = getGraphNodePositions(state.graphReplay.graph, canvas.width, canvas.height);
    let nearest = null, distance = 28;
    for (const [node, position] of positions) {
      const delta = Math.hypot(position.x - x, position.y - y);
      if (delta < distance) { distance = delta; nearest = node; }
    }
    nodeInfo(nearest);
  }, true);
  document.addEventListener('dashboard-frame', event => {
    if (state.graphReplay && loadedRun && event.detail.run_id !== loadedRun) {
      status(`${source} remains loaded. The live panels now show a different run.`);
    }
  });
  syncControls();
}
