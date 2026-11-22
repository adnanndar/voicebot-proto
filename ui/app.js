"use strict";

const $ = (s) => document.querySelector(s);

let state = { status: "idle", listening: false, live: { level: 0, speech: false } };
let renderedConv = "";
let dragging = false;
let micCheckOk = false;   // true once a mic check has matched your voice
let lastPrompt = "";
let holdPromptUntil = 0;  // keep the just-scored line visible for a moment
let toastTimer = null;

function showToast(msg, bad) {
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

const STATUS_LABELS = {
  idle: "Idle",
  listening: "Listening",
  verifying: "Checking your voice…",
  thinking: "Thinking…",
  speaking: "Speaking…",
  standby: "Sleeping — say “Hey Lyra”",
  ignored: "Not the owner",
};

// ---------------------------------------------------------------- helpers
async function post(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const j = await r.json().catch(() => ({ ok: false, reason: "bad response" }));
  if (!r.ok) throw new Error(j.reason || `HTTP ${r.status}`);
  return j;
}

function levelPct(rms) {
  const db = 20 * Math.log10(rms + 1e-9);
  return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
}

// ---------------------------------------------------------------- render
function render(s) {
  state = s;
  $("#conn-dot").className = "dot on";

  // status
  const pill = $("#status");
  let label = STATUS_LABELS[s.status] || s.status;
  if (s.listening && s.status === "listening" && !s.conversation_open) {
    label = "Say “Hey Lyra”";
  }
  pill.textContent = label;
  pill.className = "status-pill " + s.status;

  // hero
  const ownerName = s.owner_name && s.owner_name !== "the owner" ? s.owner_name : "";
  const waitingForWake = s.listening && !s.conversation_open;
  if (s.listening) {
    if (s.status === "standby" || waitingForWake) {
      $("#hero-title").textContent = s.status === "standby" ? "Sleeping…" : "Listening for “Hey Lyra”";
      $("#hero-sub").textContent = ownerName
        ? `Say “Hey Lyra” and I'll greet you, ${ownerName}.`
        : "Say “Hey Lyra” to start.";
    } else {
      $("#hero-title").textContent = "Listening…";
      $("#hero-sub").textContent = "Speak naturally. I decide 1 second after you stop talking.";
    }
  } else if (s.voiceprint.enrolled) {
    $("#hero-title").textContent = ownerName ? `Ready when you are, ${ownerName}` : "Ready when you are";
    $("#hero-sub").textContent = "Press Start listening, then say “Hey Lyra”. I decide 1 second after you stop.";
  } else {
    $("#hero-title").textContent = "Welcome!";
    $("#hero-sub").textContent = "Step 1: enrol your voice — your name first, then 10 spoken lines.";
  }
  $("#btn-listen").textContent = s.listening ? "⏹ Stop" : "▶ Start listening";

  // voice orb reacts to your level and the assistant's state
  const orb = $("#orb");
  const orbCore = $("#orb-core");
  const pct = levelPct(s.live.level || 0);
  orbCore.style.transform = `scale(${1 + Math.min(0.28, pct / 360)})`;
  orb.className = "orb" +
    (s.status === "standby" ? " sleeping" : "") +
    (s.live.speech ? " hearing" : "") +
    (s.listening && s.status !== "standby" ? " listening" : "") +
    (s.error && /mic/i.test(s.error) ? " denied" : "");
  orbCore.textContent = s.status === "standby" ? "💤" : s.live.speech ? "🎤" : "🎙";

  // 3-step guide
  const step1 = !!s.voiceprint.enrolled;
  const step2 = micCheckOk;
  const step3 = !!s.listening;
  const active = !step1 ? "1" : !step2 ? "2" : "3";
  document.querySelectorAll("#steps .step").forEach((el) => {
    const n = el.dataset.step;
    el.classList.toggle("done", n === "1" ? step1 : n === "2" ? step2 : step3);
    el.classList.toggle("active", n === active && !el.classList.contains("done"));
  });

  // live capture: show the utterance being collected and the 2 s silence countdown
  const cap = s.capture || {};
  const closeAfter = s.close_after_sec || 5;
  $("#capture").hidden = !(s.listening && cap.active);
  if (cap.active) {
    const silence = cap.silence_seconds || 0;
    const quietFor = Math.min(silence, closeAfter);
    $("#capture-time").textContent = `${(cap.seconds || 0).toFixed(1)} s`;
    $("#capture-fill").style.width = (quietFor / closeAfter * 100) + "%";
    $("#capture").classList.toggle("quiet", silence > 0);
    $("#capture-title").textContent = silence > 0
      ? `Speaker paused — ${quietFor.toFixed(1)} of ${closeAfter} s of silence`
      : "Capturing your speech…";
    $("#capture-note").textContent = silence > 0
      ? `I will decide if the speaker stays silent for ${closeAfter} seconds.`
      : `I decide after the speaker stops talking for ${closeAfter} seconds.`;
  }

  // mic
  $("#speech-dot").className = "speech-dot" + (s.live.speech ? " on" : "");
  $("#speech-text").textContent = s.listening ? (s.live.speech ? "hearing speech" : "listening…") : "microphone off";
  const rms = s.live.level || 0;
  $("#live-level").style.width = levelPct(rms) + "%";
  $("#record-level").style.width = levelPct(rms) + "%";
  const db = 20 * Math.log10(rms + 1e-9);
  $("#mic-db").textContent = db.toFixed(0) + " dB";
  $("#mic-warning").hidden = !(s.listening && rms < 0.005);

  // score
  const d = s.last_decision;
  if (d && d.score != null) {
    $("#score-block").hidden = false;
    $("#score-fill").style.width = (Math.max(0, Math.min(1, d.score)) * 100) + "%";
    const owner = d.decision === "owner";
    $("#score-fill").className = "score-fill " + (owner ? "owner" : "other");
    $("#score-thresh").style.left = (s.settings.accept_threshold * 100) + "%";
    $("#score-label").textContent =
      (owner ? "✓ it's you — " : "✗ not you — ") + (d.score * 100).toFixed(0) + "% match";
  } else {
    $("#score-block").hidden = true;
  }

  // conversation
  const sig = JSON.stringify(s.conversation);
  if (sig !== renderedConv) {
    renderedConv = sig;
    const box = $("#conversation");
    box.innerHTML = "";
    if (!s.conversation.length) {
      const e = document.createElement("div");
      e.className = "empty";
      e.innerHTML = "Nothing yet — press <b>Start listening</b> and speak.";
      box.appendChild(e);
    }
    for (const m of s.conversation) {
      const el = document.createElement("div");
      if (m.role === "ignored") {
        el.className = "msg ignored";
        const pct = m.score != null ? `${(m.score * 100).toFixed(0)}% match` : "too short to score";
        el.textContent = m.decision === "locked"
          ? `paused — ${m.reason || "repeated unknown voices"}`
          : `ignored — ${(m.duration_s ?? 0).toFixed(1)}s heard · ${pct}`;
      } else {
        el.className = "msg " + m.role;
        const r = document.createElement("div");
        r.className = "role";
        r.textContent = m.role === "user" ? "you" : "assistant";
        el.appendChild(r);
        el.appendChild(document.createTextNode(m.text || "(couldn't make out the words)"));
      }
      box.appendChild(el);
    }
    box.scrollTop = box.scrollHeight;
  }

  // profile
  const vp = $("#vp-pill");
  vp.textContent = s.voiceprint.enrolled ? "enrolled ✓" : "not enrolled";
  vp.className = "pill" + (s.voiceprint.enrolled ? " ok" : "");
  const m = s.voiceprint.meta || {};
  $("#vp-meta").textContent = s.voiceprint.enrolled
    ? `${ownerName ? ownerName + " · " : ""}${m.num_samples ?? "?"} samples · answers only to you`
    : "Enrol once so the assistant only answers you.";
  $("#btn-restore").hidden = !s.voiceprint.has_original;

  // settings (skip while user is dragging)
  if (!dragging) {
    setSlider("#timeout", s.settings.timeout_minutes);
    $("#timeout-enabled").checked = !!s.settings.timeout_enabled;
    $("#saveutt").checked = !!s.settings.save_utterances;
  }

  // enrolment
  $("#enrol-card").hidden = !s.enrol.active;
  setRecordingUI(s.enrol.recording, s.enrol.stage);
  if (s.enrol.active) {
    const nameStage = s.enrol.stage === "name";
    $("#name-stage").hidden = !nameStage;
    $("#prompt-stage").hidden = nameStage;
    $("#enrol-progress").style.width = (s.enrol.total ? (s.enrol.index / s.enrol.total) * 100 : 0) + "%";

    const dots = $("#enrol-dots");
    dots.innerHTML = "";
    if (!nameStage) {
      if (Date.now() >= holdPromptUntil) {
        $("#prompt-count").textContent = `Prompt ${Math.min(s.enrol.index + 1, s.enrol.total)} of ${s.enrol.total}`;
        $("#enrol-tone").textContent = s.enrol.tone ? `(${s.enrol.tone} delivery)` : "";
        if (s.enrol.prompt && s.enrol.prompt !== lastPrompt) {
          lastPrompt = s.enrol.prompt;
          renderPromptLine(s.enrol.prompt);
        }
      }
      for (let i = 0; i < s.enrol.total; i++) {
        const d = document.createElement("i");
        d.className = i < s.enrol.index ? "on" : i === s.enrol.index ? "current" : "";
        dots.appendChild(d);
      }
    } else {
      lastPrompt = "";
      holdPromptUntil = 0;
    }

    const result = $("#enrol-result");
    if (!result.textContent.startsWith("✓") && !result.textContent.startsWith("✗")) {
      result.className = "result";
      result.textContent = nameStage
        ? "tap record and say your name"
        : "tap record and read the line aloud";
    }
    // Briefly lock the button while the just-scored words stay on screen.
    $("#btn-record").disabled = !s.enrol.recording && !nameStage && Date.now() < holdPromptUntil;
  }
}

function renderPromptLine(prompt) {
  const box = $("#prompt-line");
  box.innerHTML = "";
  for (const w of prompt.split(/\s+/)) {
    if (!w) continue;
    const el = document.createElement("span");
    el.className = "word pending";
    el.textContent = w + " ";
    box.appendChild(el);
  }
}

function colorPromptLine(words) {
  const els = document.querySelectorAll("#prompt-line .word");
  words.forEach((w, i) => {
    if (els[i]) els[i].className = "word " + (w.ok ? "ok" : "bad");
  });
}

function setSlider(id, value) {
  const el = $(id);
  if (el && Math.abs(parseFloat(el.value) - value) > 0.001) {
    el.value = value;
    $(id + "-out").textContent = value;
  }
}

// ---------------------------------------------------------------- websocket
function connectWS() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = (e) => { try { render(JSON.parse(e.data)); } catch (_) {} };
  ws.onclose = () => {
    $("#conn-dot").className = "dot off";
    setTimeout(connectWS, 1500);
  };
  ws.onerror = () => ws.close();
}

// ---------------------------------------------------------------- profile
$("#btn-enrol").addEventListener("click", () => startEnrol("enrol"));
$("#btn-add").addEventListener("click", () => startEnrol("add"));
$("#btn-cancel").addEventListener("click", async () => { await post("/api/enrol/cancel"); });

async function startEnrol(mode) {
  if (state.privacy && state.privacy.require_consent && !$("#enrol-consent").checked) {
    alert("Please tick the consent box first — your voice stays on this device and is never uploaded.");
    return;
  }
  try {
    await post("/api/enrol/start", { mode, consent: $("#enrol-consent").checked });
    micCheckOk = false;
    lastPrompt = "";
    holdPromptUntil = 0;
    $("#enrol-result").textContent = "";
    $("#enrol-result").className = "result";
    $("#enrol-card").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (e) {
    alert("Could not start enrolment: " + e.message);
  }
}

$("#btn-delete").addEventListener("click", async () => {
  if (!confirm("Delete your voice? The assistant will no longer recognise you.")) return;
  await post("/api/profile/delete");
  micCheckOk = false;
  render(await (await fetch("/api/state")).json());
});

$("#btn-restore").addEventListener("click", async () => {
  if (!confirm("Restore the original enrolment (undo automatic updates)?")) return;
  await post("/api/profile/restore");
  render(await (await fetch("/api/state")).json());
});

// ---------------------------------------------------------------- listening
$("#btn-listen").addEventListener("click", async () => {
  await post("/api/listen", { on: !state.listening });
});

// ---------------------------------------------------------------- settings
function bindSlider(id, key) {
  const el = $(id), out = $(id + "-out");
  el.addEventListener("input", () => { dragging = true; out.textContent = el.value; });
  el.addEventListener("change", async () => {
    dragging = false;
    await post("/api/settings", { [key]: parseFloat(el.value) });
  });
}
bindSlider("#timeout", "timeout_minutes");

$("#timeout-enabled").addEventListener("change", (e) => post("/api/settings", { timeout_enabled: e.target.checked }));
$("#saveutt").addEventListener("change", (e) => post("/api/settings", { save_utterances: e.target.checked }));

// ---------------------------------------------------------------- recording
// Audio is captured on the SERVER (same PortAudio mic as live listening), so
// enrolment and live recognition use the identical capture path.
let recordTimer = null;

function setRecordingUI(recording, stage) {
  const btn = $("#btn-record");
  if (recording) {
    btn.classList.add("recording");
    btn.textContent = "■ Stop";
  } else {
    btn.classList.remove("recording");
    btn.textContent = stage === "name" ? "● Tap to answer" : "● Tap to record";
    clearInterval(recordTimer);
    $("#record-time").textContent = "";
  }
}

function handleEnrolResult(res) {
  const el = $("#enrol-result");
  if (res.words && res.words.length) {
    if (res.prompt) renderPromptLine(res.prompt);
    colorPromptLine(res.words);
    holdPromptUntil = Date.now() + 1600;  // let the green/red words be seen
  }
  if (res.done) {
    if (res.ok) {
      const name = res.owner_name ? ` ${res.owner_name}` : "";
      el.textContent = `✓ all done!${name ? " I'll call you" + name + "." : ""} ${res.num_samples} samples saved.`;
      el.className = "result ok";
      showToast(`✓ Voice enrolled — ${res.num_samples} samples saved. You are ready to listen.`);
    } else {
      el.textContent = "✗ " + (res.reason || "could not finish enrolment");
      el.className = "result bad";
    }
  } else if (res.stage === "prompts" && res.name) {
    el.textContent = `✓ Nice to meet you, ${res.name}. Now read the first line.`;
    el.className = "result ok";
    showToast(`✓ I'll call you ${res.name}.`);
  } else if (res.ok) {
    el.textContent = "✓ good — next line";
    el.className = "result ok";
  } else {
    el.textContent = "✗ " + (res.reason || "try again");
    el.className = "result bad";
  }
  fetch("/api/state").then(r => r.json()).then(render).catch(() => {});
}

$("#btn-record").addEventListener("click", async () => {
  const recording = state.enrol && state.enrol.recording;
  const stage = (state.enrol && state.enrol.stage) || "prompts";
  const el = $("#enrol-result");
  try {
    if (recording) {
      const res = await post("/api/enrol/record/stop");
      setRecordingUI(false, stage);
      handleEnrolResult(res);
    } else {
      await post("/api/enrol/record/start");
      el.textContent = stage === "name"
        ? "recording… say your name"
        : "recording… speak the prompt clearly";
      el.className = "result";
      const t0 = performance.now();
      clearInterval(recordTimer);
      recordTimer = setInterval(() => {
        $("#record-time").textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s";
      }, 100);
    }
  } catch (e) {
    el.textContent = "error: " + e.message;
    el.className = "result bad";
  }
});

// ---------------------------------------------------------------- mic check
$("#btn-miccheck").addEventListener("click", async () => {
  const out = $("#miccheck-out");
  const btn = $("#btn-miccheck");
  out.hidden = false;
  out.className = "miccheck-out";
  out.textContent = "🎤 Recording 4 seconds — say a full sentence now…";
  btn.disabled = true;
  try {
    const r = await post("/api/miccheck", { seconds: 4 });
    if (!r.speech_found) {
      out.className = "miccheck-out bad";
      out.innerHTML = `<b>No speech heard.</b> Level ${r.db} dB. Check your input device, speak closer, or raise the input volume.`;
    } else {
      const u = r.utterances[0];
      const match = u.score == null ? "no voiceprint yet" : `${(u.score * 100).toFixed(0)}% match`;
      const threshold = (state.settings && state.settings.accept_threshold) ?? 0.35;
      const rule = Math.round(threshold * 100);
      const ok = u.score != null && u.score >= threshold;
      micCheckOk = ok;
      out.className = "miccheck-out " + (ok ? "ok" : "bad");
      out.innerHTML = `<b>Heard you</b> (${u.duration_s}s, ${r.db} dB) — ${match}.<br>“${u.transcript || "…"}”` +
        (ok ? `<br>Above the ${rule}% rule — I will answer you.` : `<br>Below the ${rule}% rule — re-enrol on this same mic, or move closer.`);
    }
  } catch (e) {
    out.className = "miccheck-out bad";
    out.textContent = "Mic check failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------- boot
connectWS();
fetch("/api/state").then(r => r.json()).then(render).catch(() => {});
