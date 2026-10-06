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
