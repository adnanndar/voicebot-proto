"use strict";

// Live (browser-mic) driver for the approved Lyra UI.
// Same look and same behaviour: click "Start listening" once and Lyra keeps
// listening, detecting when you stop speaking (1 s of silence) before it
// answers. Audio is captured with MediaRecorder — the SAME path used during
// enrolment — so the voice check stays consistent.

const $ = (s) => document.querySelector(s);
const SID_KEY = "lyra_live_sid";

let sid = localStorage.getItem(SID_KEY) || "";
let stage = "name";
let name = "";
let total = 10;
let lastPrompt = "";
let toastTimer = null;

// ---- enrolment recording (push-to-record, same as the approved flow) ----
let rec = null, recKind = null, recChunks = [], recStart = 0, recTimer = null,
    autoStop = null, busy = false, starting = false;

// ---- continuous conversation capture (MediaRecorder + level-based VAD) ----
let listening = false;
let processing = false;
let miccheckMode = false;
let convStream = null, convAudioCtx = null, convScriptNode = null, convMute = null;
let convRec = null, convChunks = [], convRecStart = 0;
let vadState = "idle";          // idle | speech
let speechMs = 0, quietMs = 0;
let noiseFloor = 0.015;
let lastFrameTime = 0;
let uttStartMs = 0;

const VAD = { start_ms: 300, end_ms: 1000, max_sec: 30 };

const STATUS_LABELS = {
  idle: "Idle", listening: "Listening", verifying: "Checking your voice…",
  thinking: "Thinking…", speaking: "Speaking…", standby: "Sleeping — say “Hey Lyra”",
  ignored: "Not the owner",
};

// ----------------------------------------------------------------- helpers
async function apiGet(path) {
  const r = await fetch(path);
  return r.json();
}
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = {};
  try { data = await r.json(); } catch (_) {}
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
function setStatus(key, cls) {
  const el = $("#status");
  el.textContent = STATUS_LABELS[key] || key;
  el.className = "status-pill" + (cls ? " " + cls : "");
  $("#orb").className = "orb" + (key === "listening" ? " listening" : "");
  $("#orb-core").textContent = key === "speaking" ? "🔊" : key === "listening" ? "🎤" : "🎙";
}
function saveSid(next) {
  if (!next || next === sid) return;
  sid = next;
  localStorage.setItem(SID_KEY, sid);
}
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1] || "");
    r.onerror = () => reject(new Error("could not read the recording"));
    r.readAsDataURL(blob);
  });
}
function playClip(b64, text) {
  if (b64) {
    const a = new Audio("data:audio/wav;base64," + b64);
    a.play().catch(() => {});
    return a;
  }
  if (text && "speechSynthesis" in window) {
    try { speechSynthesis.speak(new SpeechSynthesisUtterance(text)); } catch (_) {}
  }
  return null;
}

// Speak a reply in the browser (instant) and call onDone when it finishes.
// Doing this in the browser keeps the wait after you stop talking near 1 s —
// the server no longer has to synthesise audio before it can answer.
function speakReply(text, onDone) {
  let finished = false;
  let guard = null;
  let started = false;
  const finish = () => {
    if (finished) return;
    finished = true;
    if (guard) clearTimeout(guard);
    onDone();
  };
  const clean = String(text || "").trim();
  if (!clean || !("speechSynthesis" in window)) {
    setTimeout(finish, 30);
    return;
  }
  try { speechSynthesis.cancel(); } catch (_) {}
  const u = new SpeechSynthesisUtterance(clean);
  u.lang = "en-US";
  u.rate = 1.0;
  u.onstart = () => {
    started = true;
    // Now that speech has begun, allow enough time for the whole sentence.
    if (guard) clearTimeout(guard);
    guard = setTimeout(finish, Math.min(30000, 3000 + clean.length * 110));
  };
  u.onend = finish;
  u.onerror = finish;
  // If the engine never starts (blocked, muted or no voice), don't hang.
  guard = setTimeout(() => { if (!started) finish(); }, 1200);
  try { speechSynthesis.speak(u); } catch (_) { finish(); }
}
function levelPct(rms) {
  const db = 20 * Math.log10(rms + 1e-9);
  return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
}

// --------------------------------------------------------------- rendering
function renderLine(prompt) {
  const box = $("#prompt-line");
  box.innerHTML = "";
  for (const w of String(prompt || "").split(/\s+/)) {
    if (!w) continue;
    const el = document.createElement("span");
    el.className = "word pending";
    el.textContent = w + " ";
    box.appendChild(el);
  }
}
function colorLine(words) {
  const els = document.querySelectorAll("#prompt-line .word");
  (words || []).forEach((w, i) => { if (els[i]) els[i].className = "word " + (w.ok ? "ok" : "bad"); });
}
function renderConversation(items) {
  const box = $("#conversation");
  box.innerHTML = "";
  if (!items || !items.length) {
    box.innerHTML = '<div class="empty">Enrol your voice, then press <b>Start listening</b> and say “Hey Lyra”.</div>';
    return;
  }
  for (const m of items) {
    const el = document.createElement("div");
    el.className = "msg " + (m.role === "bot" ? "bot" : m.role === "ignored" ? "ignored" : "user");
    const r = document.createElement("div");
    r.className = "role";
    r.textContent = m.role === "bot" ? "assistant" : m.role === "ignored" ? "ignored" : "you";
    el.appendChild(r);
    el.appendChild(document.createTextNode(m.text || ""));
    box.appendChild(el);
  }
  box.scrollTop = box.scrollHeight;
}
function updateSteps(which) {
  const active = which === "name" ? 1 : which === "prompts" ? 2 : 3;
  document.querySelectorAll("#steps .step").forEach((el) => {
    const n = Number(el.dataset.step);
    el.classList.toggle("done", n < active);
    el.classList.toggle("active", n === active);
  });
}
function showScore(score, owner) {
  if (score == null) { $("#score-block").hidden = true; return; }
  $("#score-block").hidden = false;
  $("#score-fill").style.width = (Math.max(0, Math.min(1, score)) * 100) + "%";
  $("#score-fill").className = "score-fill " + (owner ? "owner" : "other");
  $("#score-thresh").style.left = "35%";
  $("#score-label").textContent =
    (owner ? "✓ it's you — " : "✗ not you — ") + (score * 100).toFixed(0) + "% match";
}

function applyState(s) {
  saveSid(s.sid);
  stage = s.stage || "name";
  name = s.name || "";
  total = s.total || 10;

  $("#vp-pill").textContent = stage === "chat" ? "enrolled ✓" : "not enrolled";
  $("#vp-pill").className = "pill" + (stage === "chat" ? " ok" : "");
  $("#vp-meta").textContent = stage === "chat"
    ? `${name ? name + " · " : ""}${total} samples · answers only to you`
    : "Enrol once so the assistant only answers you.";

  $("#enrol-card").hidden = stage === "chat";
  $("#name-stage").hidden = stage !== "name";
  $("#prompt-stage").hidden = stage !== "prompts";

  if (stage === "name") {
    $("#enrol-progress").style.width = "0%";
    $("#btn-record").textContent = "● Tap to answer";
    $("#name-question").textContent = "🎙 “What is your name?”";
  } else if (stage === "prompts") {
    $("#enrol-progress").style.width = ((s.index || 0) / Math.max(1, total)) * 100 + "%";
    $("#btn-record").textContent = "● Tap to record";
    $("#prompt-count").textContent = `Prompt ${Math.min((s.index || 0) + 1, total)} of ${total}`;
    if (s.prompt && s.prompt !== lastPrompt) {
      lastPrompt = s.prompt;
      renderLine(s.prompt);
    }
    const dots = $("#enrol-dots");
    dots.innerHTML = "";
    for (let i = 0; i < total; i++) {
      const d = document.createElement("i");
      d.className = i < (s.index || 0) ? "on" : i === (s.index || 0) ? "current" : "";
      dots.appendChild(d);
    }
  }

  if (stage === "chat") {
    $("#hero-title").textContent = name ? `Ready when you are, ${name}` : "Ready when you are";
    $("#hero-sub").textContent = "Press Start listening, then say “Hey Lyra”. I decide 1 second after you stop speaking.";
    $("#btn-listen").textContent = listening ? "⏹ Stop" : "▶ Start listening";
  } else if (stage === "name") {
    $("#hero-title").textContent = "Welcome!";
    $("#hero-sub").textContent = "Step 1: enrol your voice — your name first, then 10 spoken lines.";
  } else {
    $("#hero-title").textContent = name ? `Keep going, ${name}` : "Keep going";
    $("#hero-sub").textContent = "Read each line aloud to finish enrolment.";
  }

  renderConversation(s.conversation || []);
  updateSteps(stage);
}

// --------------------------------------------------------------- enrolment
function pickMime() {
  const opts = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  for (const t of opts) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}
function setRecUI(on) {
  const b = $("#btn-record");
  if (on) {
    b.classList.add("recording");
    b.textContent = "■ Stop & send";
    $("#speech-dot").className = "speech-dot on";
    $("#speech-text").textContent = "recording…";
    setStatus("listening", "listening");
  } else {
    b.classList.remove("recording");
    b.textContent = stage === "name" ? "● Tap to answer" : "● Tap to record";
    $("#record-time").textContent = "";
    $("#record-level").style.width = "0%";
    $("#speech-dot").className = "speech-dot";
    $("#speech-text").textContent = "microphone off";
  }
}
async function startRec(kind) {
  if (busy || rec || starting) return;
  starting = true;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("This browser cannot record audio — try Chrome, Edge or Safari.", true);
      return;
    }
    let s;
    try { s = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { toast("Microphone blocked — allow mic access and try again.", true); return; }
    const mime = pickMime();
    try { rec = new MediaRecorder(s, mime ? { mimeType: mime } : undefined); }
    catch (e) { s.getTracks().forEach((t) => t.stop()); toast("Cannot record audio in this browser.", true); return; }
    recKind = kind; recChunks = [];
    rec.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
    rec.start();
    recStart = performance.now();
    setRecUI(true);
    recTimer = setInterval(() => {
      $("#record-time").textContent = ((performance.now() - recStart) / 1000).toFixed(1) + "s";
    }, 100);
    autoStop = setTimeout(() => stopRec(), 30000);
  } finally { starting = false; }
}
async function stopRec() {
  if (!rec) return;
  const kind = recKind;
  const mime = rec.mimeType || pickMime() || "audio/webm";
  clearInterval(recTimer); clearTimeout(autoStop);
  const blob = await new Promise((resolve) => {
    rec.onstop = () => resolve(new Blob(recChunks, { type: mime }));
    try { rec.stop(); } catch (_) { resolve(new Blob(recChunks, { type: mime })); }
  });
  try { rec.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
  rec = null; recKind = null;
  setRecUI(false);
  if (blob.size < 1200) { toast("Too short — tap, speak a full sentence, then tap again.", true); return; }
  busy = true;
  setStatus("thinking", "thinking");
  try {
    const audio = await blobToBase64(blob);
    const res = await api("/api/demo/enrol", { sid, audio, mime });
    handleEnrolResult(res);
  } catch (e) {
    toast(e.message || "something went wrong", true);
    setStatus("idle");
  } finally { busy = false; }
}

function handleEnrolResult(res) {
  if (!res.ok && res.stage === "name") {
    $("#enrol-result").textContent = "✗ " + (res.reason || "say your name again");
    $("#enrol-result").className = "result bad";
    setStatus("idle");
    return;
  }
  if (res.stage === "prompts" && res.ok && !res.done) {
    $("#enrol-result").textContent = res.name
      ? `✓ Nice to meet you, ${res.name}. Now read the line.`
      : "✓ good — next line";
    $("#enrol-result").className = "result ok";
    if (res.words && res.words.length) {
      colorLine(res.words);
      setTimeout(() => { lastPrompt = ""; applyState(res); playClip(res.prompt_audio, res.prompt); }, 1500);
    } else {
      applyState(res);
      playClip(res.prompt_audio, res.prompt);
    }
    return;
  }
  if (res.done && res.ok) {
    $("#enrol-result").textContent = `✓ all done! I'll call you ${res.name || ""}. ${res.num_samples || total} samples saved.`;
    $("#enrol-result").className = "result ok";
    applyState(res);
    toast(`✓ Voice enrolled${res.name ? " as " + res.name : ""} — press “Start listening”.`);
    playClip(res.prompt_audio, `Nice to meet you, ${res.name}. Ask me anything.`);
    return;
  }
  if (res.stage === "prompts" && !res.ok) {
    $("#enrol-result").textContent = "✗ " + (res.reason || "try again");
    $("#enrol-result").className = "result bad";
    setStatus("idle");
    return;
  }
  applyState(res);
}

// --------------------------------------------- continuous conversation (VAD)
function rmsOf(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

function startConvRec() {
  if (!convStream || convRec) return;
  const mime = pickMime();
  try {
    convRec = new MediaRecorder(convStream, mime ? { mimeType: mime } : undefined);
    convChunks = [];
    convRec.ondataavailable = (e) => { if (e.data && e.data.size) convChunks.push(e.data); };
    convRec.start();
    convRecStart = performance.now();
  } catch (e) {
    convRec = null;
    convChunks = [];
    convRecStart = 0;
    toast("Could not start recording — reload the page and allow the microphone.", true);
    stopListening();
  }
}

function stopConvRec() {
  if (convRec) {
    try { convRec.ondataavailable = null; } catch (_) {}
    if (convRec.state !== "inactive") { try { convRec.stop(); } catch (_) {} }
  }
  convRec = null;
  convChunks = [];
  convRecStart = 0;
}

function vadStep(rms, dtMs) {
  const thr = Math.max(noiseFloor * 2.5, 0.01);
  if (vadState === "idle") {
    noiseFloor = 0.98 * noiseFloor + 0.02 * rms;
    if (rms > thr) {
      speechMs += dtMs;
      if (speechMs >= VAD.start_ms) {
        vadState = "speech";
        speechMs = 0; quietMs = 0;
        uttStartMs = performance.now();
        $("#capture").hidden = false;
        $("#capture").classList.remove("quiet");
        $("#capture-title").textContent = "Capturing your speech…";
        setStatus("listening", "listening");
      }
    } else {
      speechMs = 0;
    }
  } else {
    if (rms < thr) {
      quietMs += dtMs;
      const q = Math.min(quietMs, VAD.end_ms);
      $("#capture").classList.add("quiet");
      $("#capture-title").textContent = `Speaker paused — ${(q / 1000).toFixed(1)} of ${(VAD.end_ms / 1000).toFixed(1)} s of silence`;
      $("#capture-fill").style.width = (q / VAD.end_ms) * 100 + "%";
      if (quietMs >= VAD.end_ms) { endUtterance(); }
    } else {
      quietMs = 0;
      $("#capture").classList.remove("quiet");
      $("#capture-title").textContent = "Capturing your speech…";
    }
    if (performance.now() - uttStartMs >= VAD.max_sec * 1000) endUtterance();
  }
}

function processLevel(samples) {
  if (!listening) return;
  const rms = rmsOf(samples);
  $("#live-level").style.width = levelPct(rms) + "%";
  $("#mic-db").textContent = (20 * Math.log10(rms + 1e-9)).toFixed(0) + " dB";
  $("#speech-dot").className = "speech-dot" + (rms > 0.012 ? " on" : "");
  $("#speech-text").textContent = vadState === "speech" ? "hearing speech" : "listening…";
  if (vadState === "speech" && uttStartMs) {
    $("#capture-time").textContent = ((performance.now() - uttStartMs) / 1000).toFixed(1) + " s";
  }

  const now = performance.now();
  const dt = lastFrameTime ? now - lastFrameTime : 0;
  lastFrameTime = now;

  // Recycle the recorder during long silences so its buffer can't grow without
  // bound (important for long listening sessions).
  if (!processing && vadState === "idle" && convRecStart && now - convRecStart > 20000) {
    stopConvRec();
    startConvRec();
  }
  if (!processing && dt > 0) vadStep(rms, dt);
}

async function endUtterance() {
  if (vadState !== "speech") return;
  vadState = "idle";
  $("#capture").hidden = true;
  speechMs = 0; quietMs = 0;

  const mime = convRec ? (convRec.mimeType || pickMime() || "audio/webm") : "audio/webm";
  let blob = new Blob(convChunks, { type: mime });
  if (convRec && convRec.state !== "inactive") {
    blob = await new Promise((resolve) => {
      convRec.onstop = () => resolve(new Blob(convChunks, { type: mime }));
      try { convRec.stop(); } catch (_) { resolve(new Blob(convChunks, { type: mime })); }
    });
  }
  convRec = null;
  convChunks = [];
  convRecStart = 0;

  if (blob.size < 800) {
    // Too short to be a real utterance — resume listening with a fresh recorder.
    if (listening) { startConvRec(); setStatus("listening", "listening"); }
    return;
  }

  processing = true;
  setStatus("verifying", "verifying");
  try {
    const audio = await blobToBase64(blob);
    const res = await api("/api/demo/ask", { sid, audio, mime, miccheck: miccheckMode });
    handleAskResult(res);
  } catch (e) {
    toast(e.message || "something went wrong", true);
    processing = false;
    setStatus("listening", "listening");
  }
}

function handleAskResult(res) {
  if (miccheckMode) {
    miccheckMode = false;
    const out = $("#miccheck-out");
    out.hidden = false;
    const pct = res.score != null ? Math.round(res.score * 100) + "%" : "no match";
    const transcript = (res.transcript || "").trim();
    const heard = transcript ? `You said: “${transcript}”` : "Didn't make out the words.";
    if (res.owner === false) {
      out.className = "miccheck-out bad";
      out.innerHTML = `<b>Not your voice</b> — ${pct}.<br>${heard}`;
    } else {
      out.className = "miccheck-out ok";
      out.innerHTML = `<b>It's you</b> — ${pct}.<br>${heard}`;
    }
    stopListening();
    return;
  }

  if (res.owner === false) {
    showScore(res.score, false);
    setStatus("ignored", "ignored");
    const pct = res.score != null ? `${Math.round(res.score * 100)}% match` : "no match";
    toast(res.decision === "locked"
      ? `Paused — ${res.reason || "too many unknown attempts"}.`
      : `That didn't sound like the enrolled voice (${pct}).`, true);
    processing = false;
    startConvRec();
    setStatus("listening", "listening");
    return;
  }

  if (res.wake_required) {
    showScore(res.score, true);
    toast(res.reply || "Say “Hey Lyra” to start.");
    processing = false;
    startConvRec();
    setStatus("listening", "listening");
    return;
  }

  applyState(res);
  showScore(res.score, true);
  if (res.stop) {
    toast("Conversation closed. Say “Hey Lyra” to start again.");
    setStatus("speaking", "speaking");
    speakReply(res.reply, () => {
      processing = false;
      if (listening) { startConvRec(); setStatus("standby"); }
    });
  } else {
    setStatus("speaking", "speaking");
    let resumed = false;
    const done = () => {
      if (resumed) return;
      resumed = true;
      processing = false;
      if (listening) { startConvRec(); setStatus("listening", "listening"); }
    };
    if (res.audio) {
      const a = new Audio("data:audio/wav;base64," + res.audio);
      a.onended = done;
      a.onerror = done;
      a.play().catch(done);
    } else {
      speakReply(res.reply, done);
    }
  }
}

// ------------------------------------------------------- conversation start/stop
async function startListening(miccheck) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("This browser cannot record audio — try Chrome, Edge or Safari.", true);
    return;
  }
  try {
    convStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    toast("Microphone blocked — allow mic access and try again.", true);
    return;
  }
  try {
    convAudioCtx = new AudioContext({ sampleRate: 16000 });
    if (convAudioCtx.state === "suspended") convAudioCtx.resume().catch(() => {});
    const src = convAudioCtx.createMediaStreamSource(convStream);
    convScriptNode = convAudioCtx.createScriptProcessor(2048, 1, 1);
    convScriptNode.onaudioprocess = (e) => processLevel(e.inputBuffer.getChannelData(0));
    convMute = convAudioCtx.createGain();
    convMute.gain.value = 0;
    src.connect(convScriptNode);
    convScriptNode.connect(convMute);
    convMute.connect(convAudioCtx.destination);
  } catch (_) {}

  listening = true;
  processing = false;
  miccheckMode = !!miccheck;
  vadState = "idle";
  speechMs = 0; quietMs = 0;
  lastFrameTime = 0;
  startConvRec();
  if (!convRec) return;  // recording failed — startConvRec already reset the UI
  setStatus("listening", "listening");
  $("#speech-text").textContent = "listening…";
  $("#btn-listen").textContent = "⏹ Stop";
}

function stopListening() {
  listening = false;
  processing = false;
  vadState = "idle";
  $("#capture").hidden = true;
  stopConvRec();
  if ("speechSynthesis" in window) { try { speechSynthesis.cancel(); } catch (_) {} }
  if (convScriptNode) { try { convScriptNode.disconnect(); } catch (_) {} convScriptNode = null; }
  if (convStream) { try { convStream.getTracks().forEach((t) => t.stop()); } catch (_) {} convStream = null; }
  if (convAudioCtx) { try { convAudioCtx.close().catch(() => {}); } catch (_) {} convAudioCtx = null; }
  convMute = null;
  setStatus("idle");
  $("#speech-dot").className = "speech-dot";
  $("#speech-text").textContent = "microphone off";
  $("#live-level").style.width = "0%";
  $("#mic-db").textContent = "— dB";
  $("#btn-listen").textContent = "▶ Start listening";
}

// -------------------------------------------------------------------- wiring
$("#btn-enrol").addEventListener("click", async () => {
  try {
    busy = true;
    const res = await api("/api/demo/start", { sid });
    lastPrompt = "";
    $("#enrol-result").textContent = "";
    $("#enrol-result").className = "result";
    $("#conversation").innerHTML = "";
    applyState(res);
    playClip(res.audio, res.question || "What is your name?");
  } catch (e) { toast(e.message, true); }
  finally { busy = false; }
});

$("#btn-record").addEventListener("click", () => {
  if (stage === "chat") return;
  if (rec) stopRec();
  else startRec("enrol");
});

$("#btn-listen").addEventListener("click", () => {
  if (stage !== "chat") { toast("Enrol your voice first — tap “Enrol my voice”.", true); return; }
  if (listening) stopListening();
  else startListening(false);
});

$("#btn-miccheck").addEventListener("click", () => {
  if (stage !== "chat") { toast("Enrol your voice first, then tap “Mic check”.", true); return; }
  const out = $("#miccheck-out");
  if (listening && miccheckMode) { stopListening(); return; }
  out.hidden = false;
  out.className = "miccheck-out";
  out.textContent = "🎤 Listening — say a full sentence now…";  if (listening) stopListening();
  startListening(true);
});

$("#btn-cancel").addEventListener("click", async () => {
  try {
    await api("/api/demo/start", { sid });
    applyState(await apiGet("/api/demo/session?sid=" + encodeURIComponent(sid)));
  } catch (e) { toast(e.message, true); }
});

$("#btn-delete").addEventListener("click", async () => {
  if (!confirm("Start over? Your voice and conversation for this session will be cleared.")) return;
  try {
    if (listening) stopListening();
    const res = await api("/api/demo/start", { sid });
    localStorage.removeItem(SID_KEY);
    sid = "";
    lastPrompt = "";
    $("#conversation").innerHTML = "";
    $("#enrol-result").textContent = "";
    $("#enrol-result").className = "result";
    applyState(res);
    toast("Started over.");
  } catch (e) { toast(e.message, true); }
});

// --------------------------------------------------------------------- boot
(async function boot() {
  // The shared (approved) HTML says "2 seconds"; the live detector uses 1 s.
  const note = $("#capture-note");
  if (note) note.textContent = "The bot decides 1 second after you stop talking.";
  // Warm the browser's speech voices so the first reply is spoken instantly.
  if ("speechSynthesis" in window) {
    try {
      speechSynthesis.getVoices();
      speechSynthesis.onvoiceschanged = () => { try { speechSynthesis.getVoices(); } catch (_) {} };
    } catch (_) {}
  }
  try {
    const url = "/api/demo/session" + (sid ? "?sid=" + encodeURIComponent(sid) : "");
    applyState(await apiGet(url));
    setStatus("idle");
  } catch (e) { toast("Could not reach Lyra.", true); }
})();
