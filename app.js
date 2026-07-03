let topics = [];
let currentTopicId = null;
let currentDepth = 'concise';
let showAnswers = false;

async function init() {
  topics = await loadTopics();
  if (topics.length > 0) {
    currentTopicId = topics[0].id;
  }
  render();
}

async function loadTopics() {
  try {
    const resp = await fetch('/topics-list');
    const files = await resp.json();
    const ts = await Promise.all(files.map(f =>
      fetch(`/topics/${f}`).then(r => r.json())
    ));
    return ts;
  } catch {
    const resp = await fetch('/topics/topic_1.json');
    const data = await resp.json();
    return [data];
  }
}

function getTopic() {
  return topics.find(t => t.id === currentTopicId);
}

function switchTopic(id) {
  currentTopicId = id;
  render();
}

function switchDepth(depth) {
  currentDepth = depth;
  render();
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderMarkdown(text) {
  let html = escapeHtml(text);

  const codeBlocks = [];
  html = html.replace(/```\n?([\s\S]*?)```/g, (_, code) => {
    const idx = codeBlocks.length;
    codeBlocks[idx] = code;
    return `%%%CB_${idx}%%%`;
  });

  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = '<p>' + html + '</p>';
  html = html.replace(/<p><\/p>/g, '');
  html = html.replace(/<br><\/li>/g, '</li>');
  html = html.replace(/<\/ul><br>/g, '</ul>');
  html = html.replace(/<p><ul>/g, '<ul>');
  html = html.replace(/<\/ul><\/p>/g, '</ul>');
  html = html.replace(/<p><li>/g, '<li>');
  html = html.replace(/<\/li><\/p>/g, '</li>');
  html = html.replace(/<p><h3>/g, '<h3>');
  html = html.replace(/<\/h3><\/p>/g, '</h3>');

  html = html.replace(/<p>%%%CB_(\d+)%%%<\/p>/g, (_, i) =>
    `<pre><code>${codeBlocks[parseInt(i)].trim()}</code></pre>`);
  html = html.replace(/%%%CB_(\d+)%%%/g, (_, i) =>
    `<pre><code>${codeBlocks[parseInt(i)].trim()}</code></pre>`);

  return html;
}

function render() {
  const main = document.getElementById('main');
  const nav = document.getElementById('tab-bar');
  const control = document.getElementById('control-bar');

  const shortNames = {
    1: '1  Introduction',
    2: '2  Dataset Collection',
    3: '3  Visual Preprocessing',
    4: '4  Statistical DS',
    5: '5  Outliers & Missing',
    6: '6  Feature Engineering',
    7: '7  Dim Reduction',
    8: '8  Clustering & Class.',
    9: '9  Pattern Mining',
    10: '10 Evaluation',
    11: '11  Old Exam \'24'
  };

  nav.innerHTML = topics.map(t =>
    `<button class="tab-btn${t.id === currentTopicId ? ' active' : ''}"
      onclick="switchTopic(${t.id})">${shortNames[t.id] || escapeHtml(t.title)}</button>`
  ).join('');

  const topic = getTopic();
  if (!topic) {
    control.innerHTML = '';
    main.innerHTML = '<p class="empty">No topic loaded.</p>';
    return;
  }

  const depthLabels = { concise: 'Concise', medium: 'Medium', full: 'Full' };

  if (topic.type === 'exam') {
    control.innerHTML = `
      <div class="answer-toggle-bar">
        <button class="answer-toggle-btn" disabled
          title="Answers not yet available">
          <span class="toggle-track">
            <span class="toggle-thumb"></span>
          </span>
          Show Answers
        </button>
      </div>
    `;
    main.innerHTML = `
      <div class="topic-header">
        <span class="topic-icon">${escapeHtml(topic.icon)}</span>
        <span class="topic-title">${escapeHtml(topic.title)}</span>
      </div>
      <div class="sections">
        ${topic.questions.map(q => `
          <div class="section-card">
            <div class="section-heading">
              <h2>${escapeHtml(q.heading)}</h2>
            </div>
            <div class="section-content">${renderMarkdown(q.content)}</div>
          </div>
        `).join('')}
      </div>
    `;
    return;
  }

  const allDepths = ['concise', 'medium', 'full'];

  control.innerHTML = `
    <div class="depth-switcher">
      ${allDepths.map(d => {
        const isAvail = topic.depths[d] && topic.depths[d].available;
        return `<button class="depth-btn${d === currentDepth ? ' active' : ''}"
          ${isAvail ? `onclick="switchDepth('${d}')"` : 'disabled'}
          ${!isAvail ? 'title="Not yet available"' : ''}>
          ${depthLabels[d]}
        </button>`;
      }).join('')}
    </div>
  `;

  const depthObj = topic.depths[currentDepth];
  if (!depthObj || !depthObj.available) {
    main.innerHTML = `
      <div class="topic-header">
        <span class="topic-icon">${escapeHtml(topic.icon)}</span>
        <span class="topic-title">${escapeHtml(topic.title)}</span>
      </div>
      <p class="empty">Content for <strong>${depthLabels[currentDepth]}</strong> depth is not yet available.</p>
    `;
    return;
  }

  main.innerHTML = `
    <div class="topic-header">
      <span class="topic-icon">${escapeHtml(topic.icon)}</span>
      <span class="topic-title">${escapeHtml(topic.title)}</span>
    </div>
    <div class="sections">
      ${depthObj.sections.map(s => `
        <div class="section-card">
          <div class="section-heading">
            <h2>${escapeHtml(s.heading)}</h2>
          </div>
          <div class="section-content">${renderMarkdown(s.content)}</div>
        </div>
      `).join('')}
    </div>
  `;
}

init();
