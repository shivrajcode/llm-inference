'use strict';
const $ = (id) => document.getElementById(id);
let status = { loaded: false };
let models = [];
let history = [];
let busy = false;
let controller = null;

function notice(message = '') {
  $('notice').textContent = message;
  $('notice').hidden = !message;
}

function controls() {
  $('load').disabled = busy || !$('model-select').value;
  $('unload').disabled = busy || !status.loaded;
  $('refresh').disabled = busy;
  $('model-select').disabled = busy;
  $('reset').disabled = busy;
  $('prompt').disabled = busy || !status.loaded;
  $('send').disabled = busy || !status.loaded || !$('prompt').value.trim();
  $('stop').hidden = !controller;
  $('max-tokens').disabled = busy;
  $('temperature').disabled = busy;
  $('conversation').setAttribute('aria-busy', String(busy));
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned an unreadable response (${response.status}).`); }
  if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function renderStatus() {
  $('status-text').textContent = status.loaded ? 'Model ready' : 'No model loaded';
  $('status-dot').classList.toggle('ready', status.loaded);
  $('context').textContent = status.context ?? '—';
  $('threads').textContent = status.threads ?? '—';
  $('batch').textContent = status.batch ?? '—';
  $('empty-detail').textContent = status.loaded
    ? 'Your model is ready. Start a conversation below; messages stay on this device.'
    : 'Load a local model to begin. Your messages are processed on this device.';
  $('generation-status').textContent = status.loaded ? `Ready · ${status.model || 'Local model'}` : 'Load a model to start chatting.';
  if (Number.isInteger(status.context)) $('max-tokens').max = Math.max(1, Math.min(2048, status.context - 1));
  controls();
}

function modelDetail() {
  const model = models.find((item) => item.path === $('model-select').value);
  $('model-detail').textContent = model
    ? `${(model.size_bytes / 1024 ** 3).toFixed(2)} GiB on disk · Runtime memory is additional.`
    : 'Add a .gguf file to the server’s models folder, then refresh.';
  controls();
}

async function refresh() {
  notice();
  try {
    const [nextStatus, result] = await Promise.all([api('/api/status'), api('/api/models')]);
    status = nextStatus;
    models = result.models;
    const selected = $('model-select').value;
    $('model-select').replaceChildren();
    if (!models.length) $('model-select').append(new Option('No GGUF models found', ''));
    for (const model of models) $('model-select').append(new Option(model.name, model.path));
    if (models.some((model) => model.path === selected)) $('model-select').value = selected;
    renderStatus();
    modelDetail();
  } catch (error) {
    status = { loaded: false };
    renderStatus();
    $('status-text').textContent = 'Server unavailable';
    notice(`${error.message} Check that the local server is running, then refresh.`);
  }
}

function reset() {
  history = [];
  $('messages').replaceChildren();
  $('empty').hidden = false;
  notice();
  renderStatus();
}

async function changeModel(unload) {
  busy = true;
  controls();
  notice();
  $('generation-status').textContent = unload ? 'Releasing model memory…' : 'Loading model…';
  try {
    await api(unload ? '/api/unload' : '/api/load', unload ? {} : { path: $('model-select').value });
    status = await api('/api/status');
    reset();
  } catch (error) {
    // A failed replacement may leave the previous model unloaded.
    try { status = await api('/api/status'); } catch { status = { loaded: false }; }
    renderStatus();
    notice(error.message);
  } finally { busy = false; controls(); }
}

function message(role, content) {
  $('empty').hidden = true;
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const label = document.createElement('p');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : 'Pocket';
  const text = document.createElement('p');
  text.className = 'message-content';
  text.textContent = content;
  article.append(label, text);
  $('messages').append(article);
  return text;
}

function scrollIfNearBottom(force = false) {
  const pane = $('conversation');
  if (force || pane.scrollHeight - pane.scrollTop - pane.clientHeight < 180) pane.scrollTop = pane.scrollHeight;
}

async function send(event) {
  event.preventDefault();
  const prompt = $('prompt').value.trim();
  if (!prompt || busy || !status.loaded || !$('max-tokens').reportValidity()) return;
  busy = true;
  controller = new AbortController();
  controls();
  notice();
  const nextHistory = [...history, { role: 'user', content: prompt }];
  message('user', prompt);
  const answer = message('assistant', '');
  $('prompt').value = '';
  $('generation-status').textContent = 'Reading your message…';
  scrollIfNearBottom(true);
  let fullText = '';
  let complete = false;
  let reader;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: nextHistory, max_tokens: Number($('max-tokens').value), temperature: Number($('temperature').value) }),
      signal: controller.signal
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.error || `Generation failed (${response.status}).`);
    }
    if (!response.body) throw new Error('Streaming is unavailable in this browser.');
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    function consume(frame) {
      const data = frame.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
      if (!data) return;
      const item = JSON.parse(data);
      if (item.error) throw new Error(item.error);
      if (typeof item.token === 'string') {
        fullText += item.token;
        answer.textContent = fullText;
        $('generation-status').textContent = 'Generating on your device…';
        scrollIfNearBottom();
      }
      if (item.done) {
        complete = true;
        const seconds = Number(item.seconds);
        const speed = seconds > 0 ? ` · ${(Number(item.generated_tokens) / seconds).toFixed(1)} tokens/s overall` : '';
        $('generation-status').textContent = `${item.prompt_tokens} prompt tokens · ${item.generated_tokens} reply tokens${speed}`;
      }
    }
    while (true) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      // Normalize only complete CRLF pairs; a CR may end a transport chunk.
      buffer = buffer.replace(/\r\n/g, '\n');
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        consume(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
      }
      if (done) { if (buffer.trim()) consume(buffer); break; }
    }
    if (!complete) throw new Error('The stream ended early. Try sending the message again.');
    history = [...nextHistory, { role: 'assistant', content: fullText }];
    if (!fullText) answer.textContent = '(The model returned an empty reply.)';
  } catch (error) {
    if (error.name === 'AbortError') {
      $('generation-status').textContent = 'Generation stopped.';
      if (fullText) history = [...nextHistory, { role: 'assistant', content: fullText }];
      else { answer.textContent = '(Stopped before a reply.)'; $('prompt').value = prompt; }
    } else {
      notice(error.message);
      $('generation-status').textContent = 'Message was not added to model history. You can retry.';
      if (!fullText) answer.textContent = '(No reply received.)';
      $('prompt').value = prompt;
    }
  } finally {
    if (reader) { try { await reader.cancel(); } catch {} reader.releaseLock(); }
    controller = null;
    busy = false;
    controls();
    $('prompt').focus();
  }
}

$('refresh').addEventListener('click', refresh);
$('model-select').addEventListener('change', modelDetail);
$('load').addEventListener('click', () => changeModel(false));
$('unload').addEventListener('click', () => changeModel(true));
$('reset').addEventListener('click', reset);
$('composer').addEventListener('submit', send);
$('stop').addEventListener('click', () => controller?.abort());
$('prompt').addEventListener('input', controls);
$('prompt').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); }
});
$('temperature').addEventListener('input', () => { $('temperature-value').textContent = $('temperature').value; });
document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => {
  if (busy) return;
  $('prompt').value = button.dataset.prompt;
  controls();
  if (status.loaded) $('prompt').focus();
  else notice('Choose and load a local model first. Your draft is ready below.');
}));
refresh();
