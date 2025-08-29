#!/usr/bin/env python3
"""
Agent↔HTML-Diagnose (Headless)
- Lädt eine Seite (index.html oder agent_test.html), prüft Agent-Hooks, triggert Aktionen,
  beobachtet DOM-Mutationen, sammelt Konsole/Netzwerk, Screenshots, und erzeugt agent_html_report.json.
- Ändert im Projekt NICHTS.

CLI:
  --base-url URL        Basis-URL deines Webservers (Default: http://127.0.0.1:8000)
  --path PATH           Pfad relativ zur Base-URL (Default: /web/agent_test.html; Fallback /web/index.html)
  --timeout SEC         Max. Zeit pro Schritt (Default: 20)
  --prompt "TEXT"       Test-Prompt an den Agent (Default: 'Bitte färbe alle H2-Überschriften blau.')
"""
from __future__ import annotations
import json, time, argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROMPT = "Bitte färbe alle H2-Überschriften blau und füge am Seitenende eine Notiz hinzu."

JS_SETUP = r"""
(() => {
  // Diagnose-Namespace
  window.__AgentDiag = { events: [], errors: [] };

  // Konsole abgreifen (im Browserkontext)
  const _log = console.log, _warn = console.warn, _err = console.error;
  console.log = (...a) => { window.__AgentDiag.events.push({t: Date.now(), type:'log', msg:String(a.join(' '))}); _log(...a); };
  console.warn = (...a) => { window.__AgentDiag.events.push({t: Date.now(), type:'warn', msg:String(a.join(' '))}); _warn(...a); };
  console.error = (...a) => { window.__AgentDiag.events.push({t: Date.now(), type:'error', msg:String(a.join(' '))}); _err(...a); };

  // Mutation Observer (DOM Änderungen)
  const mutations = [];
  const obs = new MutationObserver(list => {
    list.forEach(m => {
      mutations.push({
        t: Date.now(),
        type: m.type,
        target: (m.target && m.target.nodeName) || '?',
        added: Array.from(m.addedNodes||[]).map(n => n.nodeName),
        removed: Array.from(m.removedNodes||[]).map(n => n.nodeName),
        attrName: m.attributeName || null
      });
    });
  });
  obs.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: false });

  // Agent-Wrapper (wenn vorhanden)
  function wrapAgent(agent) {
    if (!agent || agent.__wrapped) return agent;
    const handler = {
      get(target, prop, recv) {
        const val = Reflect.get(target, prop, recv);
        if (typeof val === 'function') {
          return function(...args) {
            window.__AgentDiag.events.push({t: Date.now(), type:'agent_call', method:String(prop), args});
            try {
              const ret = val.apply(this, args);
              // Promise? dann Ergebnis loggen
              if (ret && typeof ret.then === 'function') {
                return ret.then(out => {
                  window.__AgentDiag.events.push({t: Date.now(), type:'agent_resolve', method:String(prop), out});
                  return out;
                }).catch(e => {
                  window.__AgentDiag.events.push({t: Date.now(), type:'agent_reject', method:String(prop), error:String(e)});
                  throw e;
                });
              } else {
                window.__AgentDiag.events.push({t: Date.now(), type:'agent_return', method:String(prop), out: ret});
                return ret;
              }
            } catch (e) {
              window.__AgentDiag.events.push({t: Date.now(), type:'agent_error', method:String(prop), error:String(e)});
              throw e;
            }
          }
        }
        return val;
      }
    };
    const proxy = new Proxy(agent, handler);
    proxy.__wrapped = true;
    return proxy;
  }

  // Versuche Andio.agent / Andio.Agent / window.Agent etc.
  window.__AgentDiag.exposed = {};
  const candidates = [
    ['Andio','agent'],
    ['Andio','Agent'],
    ['Agent'],
    ['andio','agent'],
  ];

  for (const path of candidates) {
    try {
      let ctx = window;
      for (const p of path) { ctx = ctx?.[p]; }
      if (ctx) {
        const wrapped = wrapAgent(ctx);
        // zurückschreiben wenn wir was ersetzen können
        try {
          let w = window;
          for (let i=0;i<path.length-1;i++) w = w[path[i]];
          if (w && path.length>1) w[path[path.length-1]] = wrapped;
        } catch(_) {}
        window.__AgentDiag.exposed[path.join('.')] = true;
      }
    } catch (e) {}
  }

  window.__AgentDiag.getSnapshot = () => ({
    htmlLen: document.documentElement.outerHTML.length,
    events: window.__AgentDiag.events.slice(),
    mutations: mutations.slice(),
    exposed: window.__AgentDiag.exposed
  });

  // Hilfsfunktion, um einen Standard-"HTML Edit" zu triggern (für generische Agenten)
  window.__AgentDiag.trySimpleEdit = async function(promptText) {
    // 1) Wenn Andio.agent.command existiert → verwenden
    try {
      if (window.Andio && window.Andio.agent && typeof window.Andio.agent.command === 'function') {
        return await window.Andio.agent.command({kind:'html-edit', prompt: promptText});
      }
    } catch(e) {
      console.warn('Andio.agent.command Fehler:', e);
    }

    // 2) Wenn es ein generisches agent.run gibt
    try {
      if (window.Agent && typeof window.Agent.run === 'function') {
        return await window.Agent.run({task:'html-edit', prompt: promptText});
      }
    } catch(e) {
      console.warn('Agent.run Fehler:', e);
    }

    // 3) Fallback: direkte DOM-Modifikation, um die Pipeline zu demonstrieren
    try {
      document.querySelectorAll('h2').forEach(el => el.style.color = 'dodgerblue');
      const note = document.createElement('div');
      note.textContent = 'Hinzugefügt vom Agent-Diagnosetest.';
      note.style = 'margin:12px 0;padding:8px;border:1px dashed #48f;background:#001025;';
      document.body.appendChild(note);
      return {ok:true, fallback:true};
    } catch(e) {
      return {ok:false, error:String(e)};
    }
  };
})();
"""

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--path", default="/web/agent_test.html")
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + args.path
    alt_url = args.base_url.rstrip("/") + "/web/index.html"
    report = {
        "page_url": None,
        "agent_exposed": {},
        "before": {},
        "after": {},
        "errors": [],
        "screenshots": {},
        "summary": []
    }

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            record_video_dir=str(OUTDIR / "video"),
            viewport={"width": 1280, "height": 900}
        )
        page = ctx.new_page()

        # Browser-Konsolenlog sammeln
        logs = []
        page.on("console", lambda m: logs.append({"type": m.type, "text": m.text()}))
        page.on("pageerror", lambda e: report["errors"].append(f"pageerror: {e}"))

        # Seite laden (mit Fallback)
        try:
            page.goto(url, timeout=args.timeout * 1000)
            report["page_url"] = url
        except Exception:
            page.goto(alt_url, timeout=args.timeout * 1000)
            report["page_url"] = alt_url

        # Setup-Skript injizieren
        page.evaluate(JS_SETUP)

        # Vorher-Snapshot
        before = page.evaluate("() => window.__AgentDiag.getSnapshot()")
        report["before"] = before
        page.screenshot(path=str(OUTDIR / "agent_before.png"))

        # Trigger: versuche eine einfache HTML-Edit-Aktion
        page.evaluate(f'(async()=>{{await window.__AgentDiag.trySimpleEdit({json.dumps(args.prompt)});}})()')

        # kurz warten und Nachher-Snapshot sammeln
        page.wait_for_timeout(1500)
        after = page.evaluate("() => window.__AgentDiag.getSnapshot()")
        report["after"] = after
        page.screenshot(path=str(OUTDIR / "agent_after.png"))

        # Agent-Expose-Status & Logs
        report["agent_exposed"] = after.get("exposed", {})
        report["screenshots"] = {
            "before": str((OUTDIR / "agent_before.png").resolve()),
            "after": str((OUTDIR / "agent_after.png").resolve())
        }
        report["logs"] = logs

        # Minimale Auswertung
        html_delta = (after.get("htmlLen", 0) - before.get("htmlLen", 0))
        mut_count = len(after.get("mutations", [])) - len(before.get("mutations", []))
        report["summary"].append(f"HTML-Längenänderung: {html_delta}")
        report["summary"].append(f"Mutationen beobachtet (Δ): {mut_count}")
        report["summary"].append(f"Agent-Hooks gefunden: {', '.join(k for k,v in report['agent_exposed'].items() if v) or 'keine'}")

        # Speichern
        out = OUTDIR / "agent_html_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n=== Agent HTML Diagnose ===")
        for line in report["summary"]:
            print("•", line)
        print(f"\nReport: {out}")
        print(f"Screenshots: {report['screenshots']['before']} | {report['screenshots']['after']}")

        ctx.close()
        browser.close()

if __name__ == "__main__":
    run()
