import assert from 'node:assert/strict';
import test from 'node:test';
import { canonicalSnapshots, parseRecording, stableLayout, displayEdges, graphChanges } from '../terralingua/server/static/js/graph_replay_data.js';

const frame = (step, options = {}) => ({ schema_version: 1, run_id: 'run', segment_id: 'first', step, segment_start: step === 0,
  ...options, graph: { env_type: 'graph', step, nodes: ['a', 'b'], agents: [{tag:'A', node:'a'}], edges: [], ...options.graph } });

test('resume removes abandoned future steps and preserves the prefix', () => {
  const records = [frame(0), frame(1), frame(2), frame(3), frame(2, { segment_id: 'second', segment_start: true }), frame(3, { segment_id: 'second' })];
  const result = canonicalSnapshots(records);
  assert.deepEqual(result.map(s => [s.step, s.segment_id]), [[0,'first'], [1,'first'], [2,'second'], [3,'second']]);
});

test('new run clears the former run; new segments replace the abandoned future', () => {
  const result = canonicalSnapshots([frame(0), frame(1), frame(0, {run_id:'new'}), frame(0, {run_id:'new', reason:'replacement'})]);
  assert.equal(result.length, 1); assert.equal(result[0].reason, 'replacement');
});

test('truncated final row is ignored; corruption inside the recording is rejected', () => {
  const line = JSON.stringify(frame(0));
  const result = parseRecording(line + '\n{"schema');
  assert.equal(result.snapshots.length, 1); assert.equal(result.ignoredTail, true);
  assert.throws(() => parseRecording(line + '\nBAD\n' + JSON.stringify(frame(1))), /line 2/);
  assert.throws(() => parseRecording('{broken'), /no complete/);
});

test('malformed graph data is rejected before rendering', () => {
  assert.throws(() => canonicalSnapshots([frame(0, {graph:{ nodes:['a'], edges:[{source:'a', target:'b', type:'follow'}], agents:[] }})]), /no node/);
  assert.throws(() => canonicalSnapshots([frame(0), frame(2), frame(1)]), /decreasing/);
});

test('one stable union layout contains births and deaths without moving nodes', () => {
  const records = [frame(0, {graph:{nodes:['a'],edges:[],agents:[]}}), frame(1), frame(2, {graph:{nodes:['b'],edges:[],agents:[]}})];
  const layout = stableLayout(records);
  assert.deepEqual(Object.keys(layout).sort(), ['a','b']);
  assert.deepEqual(layout, stableLayout(records));
  for (const position of Object.values(layout)) assert.ok(position.every(Number.isFinite));
});

test('reciprocal follow links collapse while pending requests keep direction', () => {
  const a = {source:'a',target:'b',type:'follow'}, b = {source:'b',target:'a',type:'follow'};
  const pending = {source:'b',target:'a',type:'pending'};
  assert.deepEqual(displayEdges([a,b,pending]), [{...a,type:'mutual'},pending]);
  assert.deepEqual(displayEdges([a]), [a]);
});

test('changes count directed links, births, and deaths', () => {
  const before = {nodes:['a','b'],edges:[{source:'a',target:'b',type:'follow'}]};
  const after = {nodes:['b','c'],edges:[{source:'b',target:'c',type:'follow'},{source:'c',target:'b',type:'follow'}]};
  assert.deepEqual(graphChanges(before,after), {born:1,died:1,added:2,removed:1});
});

test('segment boundaries and snapshot metadata are mandatory', () => {
  for (const options of [{run_id:''}, {segment_id:''}, {segment_start:undefined}, {graph:{env_type:'grid'}}, {graph:{step:99}}]) {
    assert.throws(() => canonicalSnapshots([frame(0, options)]));
  }
  assert.throws(() => canonicalSnapshots([frame(1)]), /segment start/);
  assert.throws(() => canonicalSnapshots([frame(0), frame(1, {segment_id:'new'})]), /segment start/);
  assert.throws(() => canonicalSnapshots([frame(0), frame(1), frame(1)]), /repeated/);
});

test('valid unterminated rows and blank complete rows follow server reader rules', () => {
  const first = JSON.stringify(frame(0));
  const result = parseRecording(first + '\n' + JSON.stringify(frame(1)));
  assert.equal(result.snapshots.length, 1); assert.equal(result.ignoredTail, true);
  assert.throws(() => parseRecording(first + '\n\n'), /line 2/);
});

test('screen positions stay fixed as nodes enter or leave', async () => {
  globalThis.window = { location: { origin: 'http://localhost' } };
  globalThis.document = { querySelector: () => null };
  const { getGraphNodePositions } = await import('../terralingua/server/static/js/grid.js');
  const node_layout = {a:[0,-1], b:[0,1]};
  const graph = {nodes:['a','b'], node_layout, layout_bounds:[-1,-1,1,1]};
  const both = getGraphNodePositions(graph, 300, 300);
  assert.deepEqual(getGraphNodePositions({...graph,nodes:['a']},300,300).get('a'), both.get('a'));
  assert.deepEqual(getGraphNodePositions({...graph,nodes:['b']},300,300).get('b'), both.get('b'));
  assert.ok(both.get('a').y >= 60);
});

test('live with no frame clears replay, and export draws on its own canvas', async () => {
  const labels = [], sizes = [];
  const context = new Proxy({}, { get(target, key) {
    if (key in target) return target[key];
    if (key === 'fillText') return text => labels.push(text);
    if (key === 'fillRect') return (...args) => sizes.push(args);
    return () => {};
  }});
  const visible = {width:300,height:300,getContext:()=>context};
  globalThis.document = { querySelector:()=>null, body:{classList:{contains:()=>false}}, getElementById:()=>visible };
  const { state } = await import('../terralingua/server/static/js/state.js');
  const { drawGrid } = await import('../terralingua/server/static/js/grid.js');
  state.graphReplay = null;
  drawGrid(undefined);
  assert.equal(labels.at(-1), 'No live frame available');
  assert.deepEqual(sizes.at(-1), [0,0,300,300]);
  state.graphReplay = {step:7,graph:{env_type:'graph',nodes:[],edges:[],agents:[]}};
  const output = {width:1000,height:1000,getContext:()=>context};
  drawGrid(undefined,output);
  assert.ok(sizes.some(args => args[2] === 1000 && args[3] === 1000));
  assert.equal(labels.at(-1), 'Recorded graph · step 7');
  assert.equal(visible.width,300);
  state.graphReplay = null;
});
