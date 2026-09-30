// Pure helpers for recorded social graph playback.
export function canonicalSnapshots(records) {
  let snapshots = [], run = null, segment = null;
  for (const record of records) {
    if (record?.schema_version !== 1 || !Number.isInteger(record.step) || record.step < 0 ||
        typeof record.run_id !== 'string' || !record.run_id || typeof record.segment_id !== 'string' || !record.segment_id ||
        typeof record.segment_start !== 'boolean' || record.graph?.env_type !== 'graph' || record.graph?.step !== record.step ||
        !record.graph || !Array.isArray(record.graph.nodes) || !Array.isArray(record.graph.edges) ||
        !Array.isArray(record.graph.agents)) throw new Error('This file is not a supported social graph recording.');
    const graph = record.graph;
    if (!graph.nodes.every(n => typeof n === 'string') ||
        !graph.edges.every(e => e && typeof e.source === 'string' && typeof e.target === 'string' && ['follow', 'pending'].includes(e.type)) ||
        !graph.agents.every(a => a && typeof a.node === 'string' && typeof a.tag === 'string')) {
      throw new Error('A recorded graph contains invalid nodes or connections.');
    }
    const nodes = new Set(graph.nodes);
    if (nodes.size !== graph.nodes.length || graph.edges.some(e => !nodes.has(e.source) || !nodes.has(e.target)) ||
        graph.agents.some(a => !nodes.has(a.node))) throw new Error('A recorded connection or agent has no node.');
    if (record.segment_start) {
      snapshots = run !== record.run_id ? [] : snapshots.filter(s => s.step < record.step);
      run = record.run_id; segment = record.segment_id;
    } else if (run !== record.run_id || segment !== record.segment_id) {
      throw new Error('A recording row has no matching segment start.');
    }
    const last = snapshots.at(-1);
    if (last && record.step <= last.step) throw new Error('The recording contains repeated or decreasing steps.');
    snapshots.push(record);
  }
  return snapshots;
}

export function parseRecording(text) {
  const lines = text.split('\n');
  const ignoredTail = lines.at(-1).length > 0;
  lines.pop(); // Only newline-terminated rows are complete.
  const records = lines.map((line, index) => {
    try { return JSON.parse(line); }
    catch (_) { throw new Error(`Invalid recording data on line ${index + 1}.`); }
  });
  const snapshots = canonicalSnapshots(records);
  if (!snapshots.length) throw new Error('The recording has no complete graph snapshots.');
  return { snapshots, ignoredTail };
}

export function stableLayout(snapshots) {
  // Order connected nodes together. Fix every position for the whole recording.
  const neighbors = new Map();
  for (const { graph } of snapshots) {
    for (const node of graph.nodes) if (!neighbors.has(node)) neighbors.set(node, new Set());
    for (const edge of graph.edges) {
      if (edge.type === 'pending') continue;
      neighbors.get(edge.source).add(edge.target);
      neighbors.get(edge.target).add(edge.source);
    }
  }
  const order = [], seen = new Set();
  const ranked = [...neighbors.keys()].sort((a, b) => neighbors.get(b).size - neighbors.get(a).size || a.localeCompare(b));
  for (const root of ranked) {
    if (seen.has(root)) continue;
    seen.add(root);
    const queue = [root];
    for (let i = 0; i < queue.length; i++) {
      const node = queue[i];
      order.push(node);
      for (const next of [...neighbors.get(node)].sort()) {
        if (!seen.has(next)) { seen.add(next); queue.push(next); }
      }
    }
  }
  return Object.fromEntries(order.map((node, i) => {
    const angle = -Math.PI / 2 + 2 * Math.PI * i / order.length;
    return [node, order.length === 1 ? [0, 0] : [Math.cos(angle), Math.sin(angle)]];
  }));
}

export function displayEdges(edges) {
  const follows = new Set(edges.filter(e => e.type === 'follow').map(e => JSON.stringify([e.source, e.target])));
  return edges.flatMap(e => {
    if (e.type !== 'follow' || e.source === e.target || !follows.has(JSON.stringify([e.target, e.source]))) return [e];
    return e.source < e.target ? [{ ...e, type: 'mutual' }] : [];
  });
}

export function graphChanges(previous, current) {
  if (!previous) return { born: current.nodes.length, died: 0, added: 0, removed: 0 };
  const beforeNodes = new Set(previous.nodes), afterNodes = new Set(current.nodes);
  const edgeSet = g => new Set(g.edges.filter(e => e.type === 'follow').map(e => JSON.stringify([e.source, e.target])));
  const beforeEdges = edgeSet(previous), afterEdges = edgeSet(current);
  const difference = (a, b) => [...a].filter(x => !b.has(x)).length;
  return { born: difference(afterNodes, beforeNodes), died: difference(beforeNodes, afterNodes),
    added: difference(afterEdges, beforeEdges), removed: difference(beforeEdges, afterEdges) };
}
