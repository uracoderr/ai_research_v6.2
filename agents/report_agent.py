"""
Phase 4: turns ranked, scraped source material into a finished report.

Design goals (full rationale in README.md):
- SPEED: routes lighter sections (intro/conclusion) to the fast 8B model
  and caps each section's token budget instead of asking every section
  for an "exhaustive" 3500-token answer. Assignment mode's total token
  budget is ~75% lower than the old fixed 4x3500 approach.
- ASSIGNMENT FIT: the default "assignment" mode targets ~2000-2500
  words - what a college assignment actually needs - instead of a
  ~10,000-word thesis chapter nobody asked for.
- TRUSTWORTHY REFERENCES: the References section is built in Python
  from the actual scraped source metadata, never from the LLM, so every
  citation is a real, clickable, verifiable source instead of a
  plausible-looking hallucination.
- HONEST CONFIDENCE SCORE: computed from the real average credibility
  of the sources that were actually used, not a hardcoded 8.5 shown on
  every single report regardless of what happened during the run.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import config
from utils.llm_client import LLMError, call_nvidia_api
from utils.logger import get_logger

logger = get_logger(__name__)


def _language_instruction(language: str) -> str:
    lang_lower = (language or "english").lower()
    if "hinglish" in lang_lower:
        return (
            "Write this ENTIRE section in conversational Hinglish (Hindi mixed with English, "
            "written in Latin/Roman script, not Devanagari). This applies to every single "
            "sentence from the first word to the very last - including the closing or summary "
            "lines. Do not drift into pure English partway through."
        )
    if "hindi" in lang_lower:
        return (
            "Write this ENTIRE section purely in Hindi using Devanagari script (हिन्दी में). "
            "This applies to every single sentence from the first word to the very last - "
            "including the closing or summary lines. Do not switch to English partway through."
        )
    return "Write in clear, well-structured academic English suitable for a college assignment."


def _is_non_english(language: str) -> bool:
    return (language or "english").strip().lower() not in ("english", "eng", "")


def _generate_section(section: dict, topic: str, scraped_text: str, language: str) -> Tuple[str, str]:
    time.sleep(section["delay"])
    lang_instruction = _language_instruction(language)
    # The language requirement is repeated in the system prompt (not just
    # buried in the user prompt) because that noticeably improves adherence,
    # especially on the smaller/faster model used for the intro & conclusion
    # sections, which was the one observed drifting back into English.
    system_prompt = (
        "You are a precise, factual research assistant writing one section of a student's "
        f"report. {lang_instruction} Output exactly what is asked for, with no filler, no "
        "preamble, and no markdown code fences unless explicitly requested."
    )
    prompt = f"""Write the '{section['title']}' section of a research report on: '{topic}'.

Instructions: {section['instruction']}
{lang_instruction}
Target length: approximately {section['target_words']} words. Be concise, precise and well organised -
prioritise clarity and factual accuracy over length. Use short paragraphs and, where useful, a
sub-heading (###) to organise ideas. When you state a specific fact, figure or claim drawn from the
source data, briefly note where it came from, e.g. "(Source: Reuters)" - only use source names that
actually appear in the data below. Do not fabricate sources, statistics, or quotes.

Source data:
{scraped_text[:config.REPORT_CONTEXT_CHAR_LIMIT]}
"""
    try:
        content = call_nvidia_api(
            prompt,
            system=system_prompt,
            max_tokens=section["max_tokens"],
            temperature=section["temperature"],
            model=section["model"],
            retries=2,
        )
        return section["title"], content.strip()
    except LLMError as e:
        logger.error("Section '%s' failed: %s", section["title"], e)
        return section["title"], (
            "_This section could not be generated because the AI model was unavailable. "
            "Please try regenerating the report._"
        )


def _build_references(scraped_sources: List[Dict]) -> str:
    """
    Built entirely from real scraped-article metadata - never from the LLM -
    so every reference is a real, verifiable source instead of a
    plausible-looking hallucination. This also directly serves the
    "students need real citations for their assignment" use case.
    """
    if not scraped_sources:
        return "_No sources were successfully scraped for this report._"
    lines = []
    seen_urls = set()
    n = 1
    for article in sorted(scraped_sources, key=lambda a: a.get("credibility_score", 0), reverse=True):
        url = article.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = article.get("title") or article.get("source_name") or url
        source_name = article.get("source_name", "Web Source")
        score = article.get("credibility_score", "-")
        lines.append(f"{n}. **{title}** — *{source_name}* (credibility {score}/10)  \n   {url}")
        n += 1
    return "\n".join(lines)


def _compute_confidence_score(scraped_sources: List[Dict], stats: dict) -> Tuple[int, float]:
    """Returns (confidence_score_percent, avg_credibility). Replaces the old
    version's hardcoded `avg_credibility = 8.5` shown on every report."""
    real_scores = [
        a.get("credibility_score", 0) for a in scraped_sources
        if isinstance(a.get("credibility_score"), (int, float))
    ]
    avg_credibility = round(sum(real_scores) / len(real_scores), 1) if real_scores else 0.0

    base_score = (avg_credibility / 10) * 60
    volume_score = min(30, (stats.get("scraped_success", 0) / max(config.SCRAPE_TARGET_SUCCESS, 1)) * 30)
    llm_penalty = 0 if stats.get("llm_ranking_success") else 15
    confidence = max(40, min(98, int(base_score + volume_score - llm_penalty)))
    return confidence, avg_credibility


def generate_report(
    topic: str,
    scraped_text: str,
    scraped_sources: List[Dict],
    language: str,
    stats: dict,
    mode: str = None,
) -> Tuple[str, dict]:
    mode = mode if mode in config.REPORT_MODES else config.DEFAULT_REPORT_MODE
    mode_config = config.REPORT_MODES[mode]
    sections = mode_config["sections"]

    if _is_non_english(language) and mode != "flash":
        # The fast/8B model is noticeably less reliable at sustaining a
        # non-English or code-switched (Hinglish) style across a whole
        # section - it tends to drift back into plain English partway
        # through (this showed up as the intro/conclusion sections coming
        # out in English while the 70B-powered sections stayed correctly in
        # Hindi/Hinglish). The 70B model doesn't have this problem, so for
        # non-English requests we bump any "fast" section up to "quality".
        # Flash mode is intentionally excluded - it's meant to always use
        # only the 8B model for guaranteed speed, in every language.
        sections = [dict(s, model="quality") if s["model"] == "fast" else s for s in sections]

    logger.info("PHASE 4: generating report in '%s' mode (%s sections, language=%s)", mode, len(sections), language)

    confidence_score, avg_credibility = _compute_confidence_score(scraped_sources, stats)

    results = {}
    with ThreadPoolExecutor(max_workers=len(sections)) as executor:
        futures = {
            executor.submit(_generate_section, section, topic, scraped_text, language): section["title"]
            for section in sections
        }
        for future in as_completed(futures):
            title, content = future.result()
            results[title] = content

    references_md = _build_references(scraped_sources)

    body = "".join(f"\n## {s['title']}\n{results.get(s['title'], '_Section unavailable._')}\n" for s in sections)
    body += f"\n## 📚 References\n{references_md}\n"

    word_count = len(body.split())
    reading_minutes = max(1, round(word_count / 200))

    header = (
        f"# {topic.title()} — Research Report\n\n"
        "## 🎯 Report Summary\n"
        f"- **Mode:** {mode_config['label']}\n"
        f"- **Confidence Score:** {confidence_score}%\n"
        f"- **Sources Used:** {stats.get('scraped_success', 0)}\n"
        f"- **Avg. Source Credibility:** {avg_credibility}/10\n"
        f"- **Approx. Word Count:** {word_count}\n"
        f"- **Est. Reading Time:** {reading_minutes} min\n\n"
        "> 📌 **Academic integrity tip:** use this report as a research starting point. "
        "Paraphrase in your own words and cite the original sources listed below in your "
        "actual submission — don't submit AI-generated text as your own work.\n"
    )

    report_text = header + body
    meta = {
        "confidence_score": confidence_score,
        "avg_credibility": avg_credibility,
        "word_count": word_count,
        "reading_minutes": reading_minutes,
        "mode": mode,
        "mode_label": mode_config["label"],
        "model_used": f"NVIDIA Llama-3.1 ({config.MODEL_QUALITY.split('/')[-1]} + {config.MODEL_FAST.split('/')[-1]})",
    }
    return report_text, meta


# ---------------------------------------------------------------------------
# Interactive tools: podcast script, mind-map diagram, RAG chat, debate mode
# ---------------------------------------------------------------------------

def generate_podcast_script(report_text: str) -> List[Dict]:
    prompt = (
        "Convert this report into a fun, natural 2-person podcast script (Host A and Host B). "
        "RETURN STRICTLY A JSON ARRAY, nothing else. "
        'Format: [{"speaker": "Host A", "text": "..."}]\n\n'
        f"Report:\n{report_text[:6000]}"
    )
    try:
        raw = call_nvidia_api(prompt, max_tokens=1500, temperature=0.3, model="fast", retries=2)
        match = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
        if not match:
            raise LLMError("No JSON array found in podcast response.")
        script = json.loads(match.group(0))
        clean = []
        for line in script[:40]:
            speaker = str(line.get("speaker", "Host"))[:20]
            text = str(line.get("text", ""))[:600]
            if text:
                clean.append({"speaker": speaker, "text": text})
        return clean or [{"speaker": "System", "text": "Sorry, the script came back empty. Please try again."}]
    except (LLMError, json.JSONDecodeError) as e:
        logger.error("Podcast generation failed: %s", e)
        return [{"speaker": "System", "text": "Podcast generation failed. Please try again."}]


def generate_diagram(report_text: str) -> str:
    prompt = f"""You are a strict Mermaid.js compiler. Create a flowchart summarising the core pillars of this report.
RULES:
1. Start exactly with 'graph TD'.
2. Use simple letters for node IDs (A, B, C...).
3. Node text must not contain parentheses, brackets, or quotes - plain words only.
4. Example: A[Data Analysis] --> B[Market Trends]
5. Return ONLY raw Mermaid code, no markdown fences, no commentary.

Report:
{report_text[:5000]}
"""
    try:
        raw = call_nvidia_api(prompt, max_tokens=700, temperature=0.1, model="fast", retries=2)
    except LLMError as e:
        logger.error("Diagram generation failed: %s", e)
        return "graph TD\n Z[Diagram unavailable] --> Y[Please try again]"

    cleaned = raw.replace("```mermaid", "").replace("```", "").strip()
    if "graph " in cleaned:
        cleaned = "graph " + cleaned.split("graph ", 1)[1]
    valid_lines = [
        line.strip() for line in cleaned.split("\n")
        if line.strip() and not line.strip().lower().startswith(("here", "sure", "note:", "this diagram"))
    ]
    return "\n".join(valid_lines) if valid_lines else "graph TD\n Z[Diagram unavailable] --> Y[Please try again]"


def rag_query(context: str, query: str) -> str:
    prompt = (
        f"Answer the question using ONLY the context below. If the answer isn't in the context, say so.\n\n"
        f"Context:\n{context[:config.REPORT_CONTEXT_CHAR_LIMIT]}\n\nQuestion: {query}"
    )
    try:
        return call_nvidia_api(prompt, max_tokens=700, temperature=0.2, model="fast", retries=2)
    except LLMError as e:
        logger.error("RAG query failed: %s", e)
        return "Sorry, I couldn't process that question right now. Please try again."


def challenge_query(context: str, query: str) -> str:
    prompt = (
        f"Act as a critical, evidence-based debater. Respond to this challenge using ONLY the context below.\n\n"
        f"Context:\n{context[:config.REPORT_CONTEXT_CHAR_LIMIT]}\n\nChallenge: {query}"
    )
    try:
        return call_nvidia_api(prompt, max_tokens=900, temperature=0.3, model="fast", retries=2)
    except LLMError as e:
        logger.error("Challenge query failed: %s", e)
        return "Sorry, I couldn't process that right now. Please try again."


# ---------------------------------------------------------------------------
# Viva Prep / Auto-Quiz - self-test tool for pre-submission / pre-viva practice
# ---------------------------------------------------------------------------

def generate_quiz(report_text: str, num_questions: int = 5) -> List[Dict]:
    """
    Generates `num_questions` quiz questions from a finished report - a mix
    of multiple choice (instantly self-gradable in the browser) and
    short-answer (graded by grade_short_answer() below). Uses the fast 8B
    model.
    """
    num_questions = max(1, min(20, int(num_questions)))
    mcq_count = max(1, round(num_questions * 0.6))
    short_count = max(0, num_questions - mcq_count)
    # Generation naturally needs more room the more questions are asked for;
    # scale the token budget accordingly instead of a one-size-fits-all cap.
    max_tokens = min(4500, 350 + num_questions * 220)

    prompt = f"""Based on this report, create exactly {num_questions} quiz questions to test whether a
student understood the material - like practice for a viva / oral exam. Make about {mcq_count}
multiple-choice ("mcq") and {short_count} short-answer ("short") questions, covering the report's
most important ideas.

RETURN STRICTLY A JSON ARRAY, nothing else. Format:
[
  {{"type": "mcq", "question": "...", "options": ["...", "...", "...", "..."], "correct_index": 0}},
  {{"type": "short", "question": "...", "key_points": "1-2 sentence summary of what a good answer should cover"}}
]

Report:
{report_text[:6000]}
"""
    try:
        raw = call_nvidia_api(
            prompt,
            system="Output strictly valid JSON and nothing else - no commentary, no markdown fences.",
            max_tokens=max_tokens, temperature=0.4, model="fast", retries=2,
        )
        match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
        if not match:
            raise LLMError("No JSON array found in quiz response.")
        questions = json.loads(match.group(0))

        clean = []
        for i, q in enumerate(questions[:num_questions]):
            qtype = q.get("type") if q.get("type") in ("mcq", "short") else "short"
            question_text = str(q.get("question", "")).strip()[:400]
            if not question_text:
                continue
            entry = {"id": f"q{i}", "type": qtype, "question": question_text}
            if qtype == "mcq":
                options = [str(o).strip()[:150] for o in (q.get("options") or []) if str(o).strip()][:4]
                if len(options) < 2:
                    continue
                idx = q.get("correct_index", 0)
                entry["options"] = options
                entry["correct_index"] = idx if isinstance(idx, int) and 0 <= idx < len(options) else 0
            else:
                entry["key_points"] = str(q.get("key_points", "")).strip()[:300]
            clean.append(entry)
        return clean
    except (LLMError, json.JSONDecodeError) as e:
        logger.error("Quiz generation failed: %s", e)
        return []


def grade_short_answer(question: str, key_points: str, student_answer: str) -> Dict:
    """Grades a free-text answer against what a good answer should cover. Uses the fast 8B model."""
    prompt = f"""A student answered this practice viva/exam question. Grade it fairly and briefly.

Question: {question}
What a good answer should cover: {key_points or "(use your own judgement based on the question)"}
Student's answer: {student_answer}

Respond with STRICT JSON only, nothing else:
{{"score": <integer 0-10>, "feedback": "<2-3 sentences, encouraging but honest - mention what was right and what was missing>"}}
"""
    try:
        raw = call_nvidia_api(
            prompt,
            system="Output strictly valid JSON and nothing else - no commentary, no markdown fences.",
            max_tokens=300, temperature=0.3, model="fast", retries=2,
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise LLMError("No JSON object found in grading response.")
        result = json.loads(match.group(0))
        raw_score = result.get("score", 0)
        score = max(0, min(10, int(raw_score))) if isinstance(raw_score, (int, float)) else 0
        feedback = str(result.get("feedback", "")).strip()[:500] or "Answer received."
        return {"score": score, "feedback": feedback}
    except (LLMError, json.JSONDecodeError, ValueError, TypeError) as e:
        logger.error("Answer grading failed: %s", e)
        return {"score": 0, "feedback": "Sorry, couldn't grade this answer right now. Please try again."}


# ---------------------------------------------------------------------------
# Auto-Slides - condenses the report into a presentation outline
# ---------------------------------------------------------------------------

def generate_slides(report_text: str) -> List[Dict]:
    """
    Summarises a report into 5-6 presentation slides (short title + a few
    bullet points each), for a "generate my class presentation" shortcut.
    Uses the fast 8B model - this is a compression/summarisation task, not
    one that needs the heavier model's deeper synthesis.
    """
    prompt = f"""Summarise this report into exactly 5-6 presentation slides for a short class
presentation. Each slide needs a short title (under 8 words) and 3-5 concise bullet points (each
under 15 words). The first slide should be a title/overview slide, the last should be a
conclusion/takeaways slide.

RETURN STRICTLY A JSON ARRAY, nothing else. Format:
[{{"title": "...", "bullets": ["...", "...", "..."]}}]

Report:
{report_text[:8000]}
"""
    try:
        raw = call_nvidia_api(
            prompt,
            system="Output strictly valid JSON and nothing else - no commentary, no markdown fences.",
            max_tokens=900, temperature=0.4, model="fast", retries=2,
        )
        match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
        if not match:
            raise LLMError("No JSON array found in slides response.")
        slides = json.loads(match.group(0))

        clean = []
        for s in slides[:8]:
            title = str(s.get("title", "")).strip()[:80]
            bullets = [str(b).strip()[:160] for b in (s.get("bullets") or []) if str(b).strip()][:6]
            if title and bullets:
                clean.append({"title": title, "bullets": bullets})
        return clean or [{"title": "Slides unavailable", "bullets": ["Please try again."]}]
    except (LLMError, json.JSONDecodeError) as e:
        logger.error("Slide generation failed: %s", e)
        return [{"title": "Slides unavailable", "bullets": ["Please try again."]}]
