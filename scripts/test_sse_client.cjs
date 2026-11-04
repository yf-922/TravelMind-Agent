const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

async function dispatch(event) {
  const calls = [];
  const chunks = [new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`)];
  const context = vm.createContext({
    TextDecoder, AbortController,
    window: {},
    localStorage: { getItem: () => null },
    fetch: async () => ({ ok: true, body: { getReader: () => ({
      read: async () => chunks.length ? { done: false, value: chunks.shift() } : { done: true },
    }) } }),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/api.js'), 'utf8'), context);
  await context.streamPlan({}, {
    onResult: () => calls.push('result'),
    onMissingFields: () => calls.push('missing'),
    onError: () => calls.push('error'),
  });
  return calls;
}

test('unapproved drafts never invoke successful-plan callback', async () => {
  assert.deepEqual(await dispatch({ type: 'result', success: false, plan: { approved: false } }), ['error']);
});

test('missing information retains continuation flow', async () => {
  assert.deepEqual(await dispatch({ type: 'result', success: false, missing_fields: ['date'] }), ['missing']);
});

test('approved results invoke plan callback', async () => {
  assert.deepEqual(await dispatch({ type: 'result', success: true, plan: { approved: true } }), ['result']);
});

test('provider failure invokes error callback', async () => {
  assert.deepEqual(await dispatch({ type: 'error', message: 'timeout' }), ['error']);
});

test('old provider empty-array ticket prices are hidden, zero remains free', () => {
  const context = vm.createContext({ window: {} });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/api.js'), 'utf8'), context);
  const adapted = context.adaptPlan({destination:'Nanjing',days_count:1,days:[{day:1,timeline:[
    {type:'attraction',name:'Unknown ticket',cost:'[]'},
    {type:'attraction',name:'Free ticket',cost:'0'},
  ]}]}, null);
  assert.equal(adapted.days[0].items[0].cost, null);
  assert.equal(adapted.days[0].items[1].cost, 0);
});


async function fragmented(chunks, { readError = false } = {}) {
  const events = [], errors = [];
  let cancelled = false, released = false;
  const reader = {
    read: async () => {
      if (chunks.length) return { done: false, value: chunks.shift() };
      if (readError) throw new Error('reader disconnected');
      return { done: true };
    },
    cancel: async () => { cancelled = true; },
    releaseLock: () => { released = true; },
  };
  const context = vm.createContext({ TextDecoder, AbortController, window: {},
    localStorage: { getItem: () => null },
    fetch: async () => ({ ok: true, body: { getReader: () => reader } }),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend/api.js'), 'utf8'), context);
  await context.streamPlan({}, { onStage: e => events.push(e), onResult: e => events.push(e),
    onError: e => errors.push(e) });
  return { events, errors, cancelled, released, context };
}
const frame = event => new TextEncoder().encode(`data: ${JSON.stringify(event)}\n\n`);

test('event and UTF-8 Chinese bytes survive arbitrary chunk boundaries', async () => {
  const bytes = frame({ type: 'result', success: true, plan: { status: 'SUCCESS' }, message: '南京路线' });
  const result = await fragmented(Array.from(bytes, byte => Uint8Array.of(byte)));
  assert.equal(result.events[0].message, '南京路线');
  assert.equal(result.events.length, 1);
  assert.deepEqual(result.errors, []);
  assert.equal(result.released, true);
});

test('multiple data events in one chunk are dispatched in order', async () => {
  const result = await fragmented([frame({type:'stage', node:'planner'}), frame({type:'result',success:true})]
    .reduce((a,b) => [new Uint8Array([...(a[0] || []), ...b])], []));
  assert.deepEqual(result.events.map(e => e.type), ['stage', 'result']);
  assert.deepEqual(result.errors, []);
});

test('reader exceptions and incomplete EOF report errors and release stream state', async () => {
  for (const readError of [true, false]) {
    const result = await fragmented([new TextEncoder().encode('data: {"type":"result"')], {readError});
    assert.equal(result.events.length, 0);
    assert.equal(result.errors.length, 1);
    assert.equal(result.cancelled, true);
    assert.equal(result.released, true);
  }
});

test('partial routes are displayed with warnings but excluded from full-success counters', async () => {
  const result = await fragmented([new Uint8Array([
    ...frame({type:'result', success:true, status:'SUCCESS', plan:{status:'SUCCESS',days:[]}}),
    ...frame({type:'result', success:true, status:'PARTIAL_SUCCESS', plan:{status:'PARTIAL_SUCCESS',days:[],warnings:['通勤待修正']}}),
  ])]);
  assert.equal(result.events.filter(e => e.full_success).length, 1);
  assert.equal(result.events.filter(e => e.partial_success).length, 1);
  const adapted = result.context.adaptPlan(result.events[1].plan, null);
  assert.equal(adapted.status, 'PARTIAL_SUCCESS');
  assert.equal(adapted.full_success, false);
  assert.ok(adapted.tips.some(t => t.includes('部分完成')));
  assert.ok(adapted.tips.includes('通勤待修正'));
});
