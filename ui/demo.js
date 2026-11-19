"use strict";

const $ = (s) => document.querySelector(s);
const SID_KEY = "lyra_demo_sid";

let sid = localStorage.getItem(SID_KEY) || "";
let stage = "name";
let lastPrompt = "";
let rec = null;
let recKind = null;
let recChunks = [];
let recStart = 0;
let recTimer = null;
let autoStop = null;
let analyser = null;
let audioCtx = null;
let meterFrame = null;
let busy = false;
let toastTimer = null;

// ----------------------------------------------------------------- helpers
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = {};
  try { data = await r.json(); } catch (_) { /* empty body */ }
  if (!r.ok) throw new Error(data.reason || `request failed (${r.status})`);
  return data;
}

function toast(msg, bad) {
  const el = $("#toast");
  el.textContent = msg;
  el.className = "toast show" + (bad ? " bad" : "");
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.className = "toast";
    setTimeout(() => { el.hidden = true; }, 300);
  }, 4200);
}

function setStatus(label, cls) {
  const el = $("#status");
  el.textContent = label;
  el.className = "status-pill" + (cls ? " " + cls : "");
}

function saveSid(next) {
  if (!next || next === sid) return;
  sid = next;
  localStorage.setItem(SID_KEY, sid);
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] || "");
    reader.onerror = () => reject(new Error("could not read the recording"));
    reader.readAsDataURL(blob);
  });
}

function playClip(b64, fallbackText) {
  if (b64) {
    const audio = new Audio("data:audio/wav;base64," + b64);
    audio.play().catch(() => { /* autoplay blocked; the text is on screen */ });
    return;
  }
  if (fallbackText && "speechSynthesis" in window) {
    try { speechSynthesis.speak(new SpeechSynthesisUtterance(fallbackText)); } catch (_) {}
  }
}

// --------------------------------------------------------------- rendering
function renderLine(prompt) {
  const box = $("#prompt-line");
  box.innerHTML = "";
  for (const word of String(prompt || "").split(/\s+/)) {
    if (!word) continue;
    const el = document.createElement("span");
    el.className = "word pending";
    el.textContent = word + " ";
    box.appendChild(el);
  }
}

function colorLine(words) {
  const els = document.querySelectorAll("#prompt-line .word");
  (words || []).forEach((w, i) => {
    if (els[i]) els[i].className = "word " + (w.ok ? "ok" : "bad");
  });
}

function renderConversation(items) {
  const box = $("#conversation");
  box.innerHTML = "";
  if (!items || !items.length) {
    box.innerHTML = '<div class="empty">Tap the microphone, then say <b>“Hey Lyra, …”</b></div>';
    return;
  }
  for (const m of items) {
    const el = document.createElement("div");
    el.className = "msg " + (m.role === "bot" ? "bot" : m.role === "ignored" ? "ignored" : "user");
    const who = m.role === "bot" ? "Lyra" : m.role === "ignored" ? "ignored" : "you";
    el.innerHTML = `<div class="role">${who}</div>`;
    el.appendChild(document.createTextNode(m.text || ""));
    box.appendChild(el);
  }
  box.scrollTop = box.scrollHeight;
}

function updateSteps(which) {
  const active = which === "name" ? 1 : which === "prompts" ? 2 : 3;
  document.querySelectorAll(".step").forEach((el) => {
    const n = Number(el.dataset.step);
    el.classList.toggle("done", n < active);
    el.classList.toggle("current", n === active);
  });
}

function applyState(s) {
  saveSid(s.sid);
  stage = s.stage || "name";

  $("#enrol-card").hidden = stage === "chat";
  $("#chat-card").hidden = stage !== "chat";
  $("#name-stage").hidden = stage !== "name";
  $("#prompt-stage").hidden = stage !== "prompts";

  if (stage === "name") {
    $("#btn-enrol-mic").textContent = "● Tap to answer";
    $("#enrol-pill").textContent = "step 1 of 2";
    $("#enrol-progress").style.width = "0%";
    setStatus("ready");
  } else if (stage === "prompts") {
    $("#btn-enrol-mic").textContent = "● Tap to record";
    $("#enrol-pill").textContent = "step 2 of 2";
    $("#enrol-progress").style.width = ((s.index || 0) / Math.max(1, s.total || 3)) * 100 + "%";
    $("#prompt-count").textContent = `Line ${Math.min((s.index || 0) + 1, s.total || 3)} of ${s.total || 3}`;
    if (s.prompt && s.prompt !== lastPrompt) {
      lastPrompt = s.prompt;
      renderLine(s.prompt);
    }
    setStatus("ready");
  } else {
    setStatus("enrolled");
    $("#chat-pill").textContent = s.name ? `enrolled as ${s.name}` : "enrolled";
    $("#talk-hint").innerHTML = s.name
      ? `Say <b>“Hey Lyra”</b> to start, and she will greet you by name. Follow-ups work until you say “stop”.`
      : "Say “Hey Lyra” to start. Follow-ups work without the wake word.";
  }

  renderConversation(s.conversation || []);
  updateSteps(stage);
}

// --------------------------------------------------------------- recording
function pickMime() {
  const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  for (const type of options) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "";
}

function setRecordingUI(on) {
  const enrolBtn = $("#btn-enrol-mic");
  const talkBtn = $("#btn-talk");
  if (on) {
    enrolBtn.classList.add("recording");
    talkBtn.classList.add("recording");
    enrolBtn.textContent = "■ Stop & send";
    talkBtn.textContent = "■ Stop & send";
    setStatus("listening", "listening");
  } else {
    enrolBtn.classList.remove("recording");
    talkBtn.classList.remove("recording");
    enrolBtn.textContent = stage === "name" ? "● Tap to answer" : "● Tap to record";
    talkBtn.textContent = "🎤 Tap to speak";
    $("#rec-time").textContent = "0.0 s";
    $("#enrol-time").textContent = "";
    $("#level").style.width = "0%";
    $("#enrol-level").style.width = "0%";
    $("#speech-dot").className = "speech-dot";
  }
}

function stopMeter() {
  clearInterval(recTimer);
  clearTimeout(autoStop);
  if (meterFrame) cancelAnimationFrame(meterFrame);
  meterFrame = null;
  analyser = null;
  if (audioCtx) {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }
}

function startMeter(stream) {
  try {
    audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    const buffer = new Float32Array(analyser.fftSize);
    const tick = () => {
      if (!analyser) return;
      analyser.getFloatTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
      const rms = Math.sqrt(sum / buffer.length);
      const pct = Math.min(100, Math.max(0, rms * 900));
      $("#level").style.width = pct + "%";
      $("#enrol-level").style.width = pct + "%";
      $("#speech-dot").className = "speech-dot" + (rms > 0.012 ? " on" : "");
      meterFrame = requestAnimationFrame(tick);
    };
    tick();
  } catch (_) {
    // no meter; recording still works
  }
}

async function startRecording(kind) {
  if (busy || rec) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("This browser cannot record audio — try Chrome, Edge or Safari.", true);
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    toast("Microphone blocked — allow mic access for this site and try again.", true);
    return;
  }

  const mime = pickMime();
  try {
    rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    toast("This browser cannot record audio — try Chrome, Edge or Safari.", true);
    return;
  }

  recKind = kind;
  recChunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
  rec.start();
  recStart = performance.now();
  setRecordingUI(true);
  startMeter(stream);
  $("#verify-note").hidden = true;

  recTimer = setInterval(() => {
    const s = ((performance.now() - recStart) / 1000).toFixed(1);
    $("#rec-time").textContent = s + " s";
    $("#enrol-time").textContent = s + " s";
  }, 100);

  autoStop = setTimeout(() => stopRecording(), kind === "ask" ? 20000 : 15000);
}

async function stopRecording() {
  if (!rec) return;
  const kind = recKind;
  const mime = rec.mimeType || pickMime() || "audio/webm";
  stopMeter();

  const blob = await new Promise((resolve) => {
    rec.onstop = () => resolve(new Blob(recChunks, { type: mime }));
    try { rec.stop(); } catch (_) { resolve(new Blob(recChunks, { type: mime })); }
  });
  try { rec.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
  rec = null;
  recKind = null;
  setRecordingUI(false);

  if (blob.size < 1200) {
    toast("That was too short — tap, speak a full sentence, then tap again.", true);
    return;
  }

  busy = true;
  setStatus("thinking", "thinking");
  try {
    const audio = await blobToBase64(blob);
    const path = kind === "ask" ? "/api/demo/ask" : "/api/demo/enrol";
    const res = await api(path, { sid, audio, mime });
    handleResult(res, kind);
  } catch (e) {
    toast(e.message || "something went wrong", true);
    setStatus(stage === "chat" ? "enrolled" : "ready");
  } finally {
    busy = false;
  }
}

// ----------------------------------------------------------------- results
function handleResult(res, kind) {
  if (kind === "enrol") {
    if (!res.ok && res.stage === "name") {
      $("#enrol-result").textContent = "✗ " + (res.reason || "say your name again");
      $("#enrol-result").className = "result bad";
      setStatus("ready");
      return;
    }

    if (res.stage === "prompts" && res.ok && !res.done) {
      $("#enrol-result").textContent = res.name
        ? `✓ Nice to meet you, ${res.name}. Read line ${res.index + 1}.`
        : "✓ good — next line";
      $("#enrol-result").className = "result ok";
      if (res.words && res.words.length) {
        if (res.prompt) { lastPrompt = res.prompt; renderLine(res.prompt); }
        colorLine(res.words);
      }
      applyState(res);
      playClip(res.prompt_audio, res.prompt);
      return;
    }

    if (res.done && res.ok) {
      $("#enrol-result").textContent = "";
      applyState(res);
      toast(`✓ Enrolled${res.name ? " as " + res.name : ""} — say “Hey Lyra”.`);
      playClip(res.prompt_audio, `Nice to meet you, ${res.name}. Ask me anything.`);
      return;
    }

    if (res.stage === "prompts" && !res.ok) {
      $("#enrol-result").textContent = "✗ " + (res.reason || "try again");
      $("#enrol-result").className = "result bad";
      setStatus("ready");
      return;
    }

    applyState(res);
    return;
  }

  // a question
  const note = $("#verify-note");
  if (res.owner === false) {
    const pct = res.score != null ? `${Math.round(res.score * 100)}% match` : "no match";
    note.textContent = res.decision === "locked"
      ? `Paused — ${res.reason || "too many unknown attempts"}.`
      : `That didn't sound like the enrolled voice (${pct}). Lyra only answers its owner.`;
    note.className = "verify-note bad";
    note.hidden = false;
    setStatus("not you", "ignored");
    return;
  }

  if (res.wake_required) {
    note.textContent = res.reply || "Say “Hey Lyra” to start.";
    note.className = "verify-note";
    note.hidden = false;
    setStatus("ready");
    return;
  }

  note.hidden = true;
  applyState(res);
  if (res.score != null) {
    $("#score-note").textContent = `${Math.round(res.score * 100)}% voice match`;
  }
  if (res.stop) {
    setStatus("ready");
    $("#talk-hint").innerHTML = "Conversation closed. Say <b>“Hey Lyra”</b> to start again.";
  } else {
    setStatus("speaking", "speaking");
  }
  if (res.audio) {
    const audio = new Audio("data:audio/wav;base64," + res.audio);
    audio.onended = () => setStatus("enrolled");
    audio.play().catch(() => setStatus("enrolled"));
  } else {
    playClip(null, res.reply);
    setStatus("enrolled");
  }
}

// ------------------------------------------------------------------- boot
$("#btn-enrol-mic").addEventListener("click", () => {
  if (rec) stopRecording();
  else startRecording("enrol");
});

$("#btn-talk").addEventListener("click", () => {
  if (rec) stopRecording();
  else startRecording("ask");
});

$("#btn-restart").addEventListener("click", async () => {
  if (rec) return;
  try {
    busy = true;
    const res = await api("/api/demo/start", { sid });
    lastPrompt = "";
    $("#enrol-result").textContent = "";
    $("#enrol-result").className = "result";
    $("#verify-note").hidden = true;
    $("#conversation").innerHTML = "";
    applyState(res);
    playClip(res.audio, res.question);
    toast("Started over — say your name.");
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy = false;
  }
});

(async function boot() {
  try {
    const url = "/api/demo/session" + (sid ? "?sid=" + encodeURIComponent(sid) : "");
    const r = await fetch(url);
    applyState(await r.json());
  } catch (e) {
    toast("Could not reach Lyra — is the demo running?", true);
  }
})();
