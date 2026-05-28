"""Example usage for the SRA LangGraph report-generation workflow."""

import os

from sra_agent import generate_research_report
from sra_rag import SraRagOptions, create_sra_rag


def main() -> None:
    options = SraRagOptions(
        working_dir="./sra_rag_data_test",
        llm_base_url=os.environ["SRA_LLM_BASE_URL"],
        llm_api_key=os.environ["SRA_LLM_API_KEY"],
        llm_model=os.environ["SRA_LLM_MODEL"],
        embedding_model=os.environ["SRA_EMBEDDING_MODEL"],
        embedding_dim=int(os.environ["SRA_EMBEDDING_DIM"]),
        max_token_size=int(os.environ.get("SRA_MAX_TOKEN_SIZE", "8192")),
    )
    rag = create_sra_rag(options)

    result = generate_research_report(
        {
            "research_goal": "研究特应性皮炎儿童人群的炎症通路调控",
            "research_direction": "肠皮轴、Th2 免疫调控",
            "outcome_indicators": "EASI、瘙痒评分、IL-4、IgE",
            "innovation_points": ["理论创新", "方法创新", "应用创新"],
            "research_type": "药理机制",
            "output_purpose": "基金申报",
            "focus_mechanism": "IL-4/IL-13 相关炎症通路",
            "include_safety_economics": True,
        },
        retriever=rag.retriever,
        llm_options=options,
    )
    print(result.final_report)


if __name__ == "__main__":
    main()
