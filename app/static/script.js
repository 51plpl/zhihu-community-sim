const feed = document.getElementById('feed');
const questionInput = document.getElementById('questionInput');
const genBtn = document.getElementById('genBtn');
const loading = document.getElementById('loading');

let isLoading = false;
let totalAnswers = 0;
let lastQuestion = '';
let usedPersonaIds = [];

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function createAnswerCard(answer, index) {
  const card = document.createElement('div');
  card.className = 'card';

  const demo = answer.persona_demo || '未知身份';
  const impression = answer.persona_impression || '';
  const content = answer.answer || '';

  card.innerHTML = `
    <div class="card-persona">${escapeHtml(demo)}</div>
    <div class="card-impression">${escapeHtml(impression)}</div>
    <div class="card-content">
      <div>${escapeHtml(content)}</div>
    </div>
    <div class="card-footer">
      <span class="card-index">#${index}</span>
    </div>
  `;

  return card;
}

function appendLoadMore() {
  // Remove old "load more" if exists
  const old = document.getElementById('loadMore');
  if (old) old.remove();

  const btn = document.createElement('button');
  btn.id = 'loadMore';
  btn.className = 'btn-load-more';
  btn.textContent = '再生成 5 个';
  btn.addEventListener('click', generateMore);
  feed.appendChild(btn);
}

async function generate() {
  const question = questionInput.value.trim();
  if (!question) { questionInput.focus(); return; }
  if (isLoading) return;
  isLoading = true;

  feed.innerHTML = '';
  loading.style.display = 'flex';
  genBtn.disabled = true;
  totalAnswers = 0;
  lastQuestion = question;
  usedPersonaIds = [];

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, count: 5, chunks: 3 }),
    });

    if (!data.answers || data.answers.length === 0) {
      feed.innerHTML = '<div class="empty">没有生成成功</div>';
      return;
    }

    feed.innerHTML = '';
    data.answers.forEach((a, i) => {
      totalAnswers = i + 1;
      if (a.persona_id) usedPersonaIds.push(a.persona_id);
      feed.appendChild(createAnswerCard(a, i + 1));
    });
    appendLoadMore();
  } catch (err) {
    feed.innerHTML = `<div class="empty">生成失败: ${err.message}</div>`;
  } finally {
    isLoading = false;
    loading.style.display = 'none';
    genBtn.disabled = false;
  }
}

async function generateMore() {
  if (isLoading) return;
  isLoading = true;
  loading.style.display = 'flex';

  // Remove the load more button while loading
  const old = document.getElementById('loadMore');
  if (old) old.remove();

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: lastQuestion, count: 5, chunks: 3, exclude_ids: usedPersonaIds }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    if (data.answers && data.answers.length > 0) {
      data.answers.forEach((a) => {
        totalAnswers++;
        if (a.persona_id) usedPersonaIds.push(a.persona_id);
        feed.appendChild(createAnswerCard(a, totalAnswers));
      });
      appendLoadMore();
    }
  } catch (err) {
    // Re-add the button on error so user can retry
    appendLoadMore();
  } finally {
    isLoading = false;
    loading.style.display = 'none';
  }
}

genBtn.addEventListener('click', generate);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') generate();
});
