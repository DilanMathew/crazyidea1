# crazyidea2: Agents are the new attack surface (Nov 2026)

**Lane:** agents and LLM tooling · **Voice:** ML practitioner · **Goal:** reach and followers

## 1. Why this topic, right now

Agents moved from demos to production in 2026. Daily use of agents at work went from 14% to 37% in a year (Stack Overflow, 2026), and **MCP** became the common way to plug tools into them. Security did not keep up. This year brought tool-poisoning and "rug pull" research on MCP servers, prompt-injection CVEs in AI IDEs, and official MCP security guidance. Meanwhile the builder conversation on LinkedIn and X is about **coding-agent harnesses and memory**, **token and cost efficiency**, **cheap or small models for simple steps**, and **evals that don't lie**.

The tension worth playing with: **"Give agents more tools" vs. "every tool is a door."** Most people have read *about* prompt injection. Very few have *watched* one work. That gap is the hook.

## 2. Ideas, ranked by stop-the-scroll power

| # | Idea | What you build (ML/AI work) | Hook |
|---|------|-----------------------------|------|
| 1 | **Agent Heist** ✅ built | Interactive simulator of 8 prompt-injection / MCP attacks vs. 7 defenses, exact odds, and a Pareto front of all 128 defense combos (`demo/index.html`) | "One sentence in an email made my AI assistant mail out my API key. Here are the 7 defenses I tried, ranked." |
| 2 | **Real-model Heist** (sequel to #1) | Replace the toy numbers with measurements: run the same 8 payloads × N trials against 3–4 real models (one frontier, one open-weight, one small) through a minimal agent loop. Report attack success with confidence intervals. | "I attacked 4 LLMs 3,200 times. The smallest model wasn't the easiest to fool." *(only if the data says so)* |
| 3 | **Judge Flip** | LLM-as-judge bias experiment: same pair of answers, swap the order, pad one with fluff, and count verdict flips. Add a simple fix (swap-and-average, rubric scoring) and show the flip rate drop. | "I swapped the order of two answers. My AI judge changed its mind X% of the time." |
| 4 | **The Agent Receipt** | Instrument a coding agent on one real bug fix. Itemise tokens, tool calls, retries, dead ends and $ cost, and render it as a printed supermarket receipt. | "My AI agent's receipt for fixing one bug." (very shareable visual) |
| 5 | **Distil My Agent** | Log 1–2k traces of a frontier model doing one narrow task. Fine-tune a 1–3B open model with LoRA on those traces, then plot accuracy vs. cost vs. latency. | "A model 100× smaller does my agent's job at Y% accuracy for Z% of the cost." |
| 6 | **MCP Census** | Pull public MCP server registries, embed every tool description, cluster them, and flag risky capabilities (shell, filesystem, network, credentials) with a small classifier. | "I read 5,000 AI tool descriptions. N% can run shell commands." |
| 7 | **Scoreboard Trap** | A toy RL agent in a small game where the reward can be gamed. Show the reward going up while the real goal gets worse (reward hacking, live). | "My agent's score went up 40%. It got worse at the game." |
| 8 | **Agent Memory Poisoning** | Agent with long-term memory. One poisoned chat plants a "fact" that surfaces in a later, unrelated session. Measure persistence over N sessions. | "I told my AI one lie in March. It repeated it in June." |

**Suggested series:** #1 now (interactive toy) → #3 the following week (quick, data-driven) → #2 as the big finale with real measurements. #4 and #5 are good to have in reserve if the series takes off.

## 3. The demo: Agent Heist

Open `crazyidea2/demo/index.html` in any browser. There is nothing to install and no API key, and no real model is called.

- **Scenario:** a user asks an assistant to *"Summarise my unread email"* (plus the weather, for the MCP attacks). One email or tool is hostile. The inbox contains a fake secret (`DEMO-KEY-1234`).
- **8 attacks:** a polite ask, hidden HTML text, a fake tool output, a Base64 smuggle, invisible Unicode tag characters, a markdown image pixel, a poisoned MCP tool and an MCP rug pull.
- **7 defenses:** pinned tool hashes, an injection classifier, plan-then-execute (CaMeL-style), spotlighting, least-privilege tools, human approval and an egress allowlist. Each has a friction cost.
- **Live run:** an animated agent trace and an attacker "listener" that shows exactly what leaked.
- **Exact odds:** stacked bars showing *which stage* stops each attack.
- **Pareto chart:** all 128 defense combinations, breach rate vs. friction. Click a dot to apply it.
- **Gullibility slider:** from a hardened model to an eager one.

**What the model shows (default settings):**
- No defenses: **60%** average breach rate.
- **Human approval alone, the most expensive single defense (friction 15), still lets 36% through.** Reads and image rendering are not "side effects", so nobody is asked.
- **Pin + least privilege + egress allowlist (friction 11): 4.9%.** That's three cheap layers.
- **Everything on (friction 43): 1.5%.** The **poisoned MCP tool still gets through ~11% of the time**, because the tool is both trusted and needed.
- Pinning tool hashes stops rug pulls almost completely but does nothing against a tool that was malicious from day one.

> These are illustrative assumptions in a toy model, not measurements. The *structure* is the lesson: every defense covers some exfiltration routes and is blind to others. Say this in the post. Idea #2 is how you replace the assumptions with real data.

### Recording it for LinkedIn
1. Open full-screen and choose **Markdown Pixel**. Press **Run heist** with no defenses so the listener lights up red.
2. Turn on **Human approval** and run again. It still leaks, because nobody is asked to approve an image.
3. Click **Cheapest <5%** and watch the Pareto ring jump. Run the attack again to show it blocked.
4. Finish on **Poisoned Tool** with everything on and run it until it leaks.
5. Export 30–45 s at 1080×1350 (4:5).

## 4. LinkedIn drafts

### Post 1: Agent Heist (lead post, with video)

> I hid one sentence in an email.
> My AI assistant read it, then mailed my API key to a stranger.
>
> Prompt injection is the #1 risk for AI agents, and most people have only read about it. So I built a small simulator where you can watch it happen.
>
> 🔴 8 attacks: hidden text, fake tool output, invisible Unicode, a markdown "pixel", a poisoned MCP tool…
> 🛡️ 7 defenses: classifiers, spotlighting, plan-then-execute, least privilege, human approval…
> 📈 All 128 combinations plotted: breach rate vs. friction
>
> What surprised me:
> 1️⃣ "Human in the loop" alone still leaked 36%. Nobody is asked to approve an *image*.
> 2️⃣ Three cheap layers (pin tool hashes + least-privilege tools + egress allowlist) beat every expensive single defense: 60% → 5%.
> 3️⃣ With everything on, the poisoned MCP tool still got through. When the attacker *is* the tool, your guardrails trust them.
>
> Caveat: it's a toy model with assumed numbers, not a benchmark. Next I'm running the same attacks against real models.
>
> Try to get the breach rate under 2% for less than 20 friction 👇
> [link] · Code on GitHub: [repo]
>
> #AIAgents #MCP #LLMSecurity #PromptInjection #MachineLearning

### Post 2: Judge Flip (teaser, publish once you have the data)

> Same two answers. I only swapped their order.
> My AI judge changed its verdict __% of the time.
>
> If your evals use LLM-as-judge, your leaderboard may be measuring *position*, not quality. Here's the 10-line fix I used (swap-and-average) and how much it helped: __ → __.
> [chart]
>
> #LLM #Evals #MLOps

### Post 3: Agent Receipt (visual)

> My AI agent fixed one bug. Here's the receipt 🧾
> __ tool calls · __ retries · __ k tokens · $__
> The most expensive line item wasn't the fix. It was __.
>
> #AIEngineering #CodingAgents

**Tips for reach:** put the link in the first comment, not the post. Lead with a video or a carousel, not text alone. End with a challenge people can answer. Reply to every comment in the first hour. Never publish a number you haven't measured, because "toy model" honesty is what earns trust.

## Sources (context, Nov 2026)
- [Stack Overflow: Agents on a leash (2026)](https://stackoverflow.blog/2026/05/27/agents-on-a-leash-agentic-ai-remains-mostly-monitored-at-work/)
- [CSA: MCP tool poisoning and rug pull attacks](https://labs.cloudsecurityalliance.org/research-rb/csa-whitepaper-mcp-security-tool-poisoning-20260506-csa-styl/)
- [CSA research note: MCP security crisis](https://labs.cloudsecurityalliance.org/research/csa-research-note-mcp-security-crisis/)
- [The state of MCP security in 2026](https://dev.to/luckypipewrench/the-state-of-mcp-security-2026-incidents-attack-patterns-and-defense-coverage-2h45)
- [The New Stack: 5 key trends shaping agentic development in 2026](https://thenewstack.io/5-key-trends-shaping-agentic-development-in-2026/)
- [LLM Daily, Nov 2026 digests](https://buttondown.com/agent-k/archive/llm-daily-november-14-2026/)
