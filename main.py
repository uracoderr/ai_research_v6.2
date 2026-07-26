"""
CLI entry point - run `python main.py` for the interactive terminal
version of ThesisPilot.

For the web UI (with Podcast, Mindmap, RAG chat and Debate tools), run
`python app.py` instead - see README.md.
"""
import time

from rich.prompt import Prompt

import config
from agents.filter_agent import filter_and_rank_articles
from agents.report_agent import generate_report
from agents.scraper_agent import scrape_top_articles
from agents.search_agent import fetch_articles, optimize_query
from utils import report_store
from utils.logger import get_logger, setup_logging

setup_logging(pretty=True)
logger = get_logger(__name__)

console = config.console
CLI_SESSION = "cli"  # single shared "session" folder for local CLI usage - no multi-user isolation needed


def main():
    try:
        config.validate_config()
    except RuntimeError as e:
        console.print(f"[error]❌ {e}[/error]")
        return

    console.print("\n[success]======================================================[/success]")
    console.print("[success]   🚀 THESISPILOT AI RESEARCH AGENT (V6.0 SAAS)   [/success]")
    console.print("[success]======================================================[/success]\n")

    raw_query = Prompt.ask("[step]Kis topic par deep research karni hai?[/step]")
    language_input = Prompt.ask(
        "[step]Report kis language me banani hai?[/step]",
        choices=["english", "hindi", "hinglish"], default="english",
    )
    mode_input = Prompt.ask(
        "[step]Report mode?[/step] (assignment = fast, ~2000 words | deep = exhaustive, ~4500 words)",
        choices=list(config.REPORT_MODES.keys()), default=config.DEFAULT_REPORT_MODE,
    )
    language = language_input.capitalize()

    start_time = time.time()
    llm_calls = 0

    console.print("\n[info]🚀 Initiating Autonomous Research Pipeline...[/info]")

    topic = optimize_query(raw_query)
    llm_calls += 1
    console.print(f"[info]✨ Query refined to:[/info] '{topic}'")

    console.print("[info]▶ Searching the web...[/info]")
    raw_articles = fetch_articles(topic)
    if not raw_articles:
        console.print("[error]❌ No articles found. Try a different topic.[/error]")
        return
    console.print(f"[success]✅ Found {len(raw_articles)} candidate sources.[/success]")

    console.print("[info]▶ Ranking source credibility...[/info]")
    ranked_articles, duplicates_removed, filter_calls, llm_success = filter_and_rank_articles(
        raw_articles, top_n=config.SEARCH_MAX_RESULTS
    )
    llm_calls += filter_calls
    console.print(f"[success]✅ {len(ranked_articles)} high-quality sources selected.[/success]")

    console.print("[info]▶ Reading full articles...[/info]")
    scraped_data, scraped_count, scraped_sources = scrape_top_articles(ranked_articles)
    if scraped_count == 0:
        console.print("[error]❌ Could not read any sources. Please try again in a bit.[/error]")
        return
    console.print(f"[success]✅ Extracted content from {scraped_count} sources.[/success]")

    console.print("[info]▶ Writing your report...[/info]")
    stats_dict = {
        "scraped_success": scraped_count,
        "duplicates_removed": duplicates_removed,
        "llm_ranking_success": llm_success,
    }
    final_report, meta = generate_report(topic, scraped_data, scraped_sources, language, stats_dict, mode_input)
    llm_calls += len(config.REPORT_MODES[mode_input]["sections"])

    end_time = time.time()
    cli_metrics = {
        "time_seconds": round(end_time - start_time, 1),
        "llm_calls": llm_calls,
        "sources_found": len(raw_articles),
        "sources_used": scraped_count,
        "word_count": meta["word_count"],
        "reading_minutes": meta["reading_minutes"],
        "confidence_score": meta["confidence_score"],
        "mode_label": meta["mode_label"],
        "model_used": meta["model_used"],
    }
    paths = report_store.save_report(CLI_SESSION, topic, final_report, scraped_data, cli_metrics)
    console.print("\n[step]============================================[/step]")
    console.print("[step]         📈 PIPELINE METRICS DASHBOARD      [/step]")
    console.print("[step]============================================[/step]")
    console.print(f"[info]  Total Time Taken  :[/info] {round(end_time - start_time, 1)}s")
    console.print(f"[info]  LLM API Calls     :[/info] {llm_calls}")
    console.print(f"[info]  Raw Fetched       :[/info] {len(raw_articles)}")
    console.print(f"[info]  Spam/Dupes Dropped:[/info] {duplicates_removed}")
    console.print(f"[info]  Successful Scrapes:[/info] {scraped_count}")
    console.print(f"[info]  Word Count        :[/info] {meta['word_count']} (~{meta['reading_minutes']} min read)")
    console.print(f"[info]  Confidence Score  :[/info] {meta['confidence_score']}% (avg source credibility {meta['avg_credibility']}/10)")
    console.print(f"[info]  Synthesis Models  :[/info] {meta['model_used']}")
    console.print("[step]============================================[/step]")

    console.print("\n[success]🎉 Research Complete![/success]")
    console.print(f"[info]📄 Markdown File : {paths['md']}[/info]")
    console.print(f"[info]🌐 HTML Export   : {paths['html']}[/info]")
    console.print(f"[interactive]🧠 Raw Context   : {paths['context']} (ready for RAG/debate)[/interactive]")
    console.print("\n[interactive]💡 Pro Tip: run `python app.py` for the web UI with Podcast, Mindmap, RAG and Debate tools![/interactive]\n")


if __name__ == "__main__":
    main()
