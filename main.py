"""
CLI entry point - run `python main.py` for the interactive terminal
version of ThesisPilot.

For the web UI (with Podcast, Mindmap, RAG chat and Debate tools), run
`uvicorn app:app --reload` instead - see README.md.
"""
import time

import config
from agents.filter_agent import filter_and_rank_articles
from agents.report_agent import generate_report
from agents.scraper_agent import scrape_top_articles
from agents.search_agent import fetch_articles, optimize_query
from utils import report_store
from utils.logger import get_logger, setup_logging

setup_logging(pretty=False)
logger = get_logger(__name__)

CLI_SESSION = "cli"  # single shared "session" folder for local CLI usage - no multi-user isolation needed


def _prompt(msg: str, choices: list = None, default: str = None) -> str:
    """Simple prompt wrapper (no rich required)."""
    hint = ""
    if choices:
        hint += f" [{'/'.join(choices)}]"
    if default:
        hint += f" (default: {default})"
    while True:
        val = input(f"{msg}{hint}: ").strip() or default or ""
        if choices and val not in choices:
            print(f"  Please enter one of: {', '.join(choices)}")
            continue
        return val


def main():
    try:
        config.validate_config()
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    print("\n======================================================")
    print("   🚀 THESISPILOT AI RESEARCH AGENT")
    print("======================================================\n")

    raw_query = _prompt("Research topic")
    language_input = _prompt(
        "Report language",
        choices=["english", "hindi", "hinglish"],
        default="english",
    )
    mode_input = _prompt(
        "Report mode (assignment=fast ~2000w | deep=exhaustive ~4500w)",
        choices=list(config.REPORT_MODES.keys()),
        default=config.DEFAULT_REPORT_MODE,
    )
    language = language_input.capitalize()

    start_time = time.time()
    llm_calls = 0

    print("\n🚀 Initiating Autonomous Research Pipeline...")

    topic = optimize_query(raw_query)
    llm_calls += 1
    print(f"✨ Query refined to: '{topic}'")

    print("▶ Searching the web...")
    raw_articles = fetch_articles(topic)
    if not raw_articles:
        print("❌ No articles found. Try a different topic.")
        return
    print(f"✅ Found {len(raw_articles)} candidate sources.")

    print("▶ Ranking source credibility...")
    ranked_articles, duplicates_removed, filter_calls, llm_success = filter_and_rank_articles(
        raw_articles, top_n=config.SEARCH_MAX_RESULTS
    )
    llm_calls += filter_calls
    print(f"✅ {len(ranked_articles)} high-quality sources selected.")

    print("▶ Reading full articles...")
    scraped_data, scraped_count, scraped_sources = scrape_top_articles(ranked_articles)
    if scraped_count == 0:
        print("❌ Could not read any sources. Please try again in a bit.")
        return
    print(f"✅ Extracted content from {scraped_count} sources.")

    print("▶ Writing your report...")
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

    print("\n============================================")
    print("         📈 PIPELINE METRICS DASHBOARD")
    print("============================================")
    print(f"  Total Time Taken  : {round(end_time - start_time, 1)}s")
    print(f"  LLM API Calls     : {llm_calls}")
    print(f"  Raw Fetched       : {len(raw_articles)}")
    print(f"  Spam/Dupes Dropped: {duplicates_removed}")
    print(f"  Successful Scrapes: {scraped_count}")
    print(f"  Word Count        : {meta['word_count']} (~{meta['reading_minutes']} min read)")
    print(f"  Confidence Score  : {meta['confidence_score']}% (avg source credibility {meta['avg_credibility']}/10)")
    print(f"  Synthesis Models  : {meta['model_used']}")
    print("============================================")

    print("\n🎉 Research Complete!")
    print(f"📄 Markdown File : {paths['md']}")
    print(f"🌐 HTML Export   : {paths['html']}")
    print(f"🧠 Raw Context   : {paths['context']} (ready for RAG/debate)")
    print("\n💡 Pro Tip: run `uvicorn app:app --reload` for the web UI with Podcast, Mindmap, RAG and Debate tools!\n")


if __name__ == "__main__":
    main()
