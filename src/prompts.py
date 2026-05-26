S2_SYSTEM_PROMPT = """
You are an intelligent and precise assistant that can understand the contents of research papers.
You are knowledgable on different fields and domains of science, in particular computer science.
You are able to interpret research papers, create questions and answers, and compare multiple papers.
""".strip()

GENERATE_SYNTHETIC_GOALS_PAPERS_QUESTION = """
When writing a scientific research paper, we often include tables comparing different works to accomplish a variety of goals.

The author has this goal in mind when they create the table for what they want to convey to the reader via the objective comparison of papers.

For example, some potential goals might include:

1. Highlighting gaps in existing research
2. Contextualizing the study
3. Evaluating methodology differences
4. Demonstrating novelty
5. Assessing consistency of results
6. Simplifying complex information
7. Supporting the literature review

Generally this goal can be written down as a simple open-ended question that the author anticipates the reader will have and that can be answered with the table.

Your task is to generate this goal given a particular table from a research paper.

[Table]
{table}

We also provide information about the papers being discussed in the table.

{papers}

Return output in the following JSON format:

{
  "goal": "...",
  "justification": "..."
}
""".strip()

EVALUATE_GOALS_TO_TABLE = """
Imagine you are a co-author of a scientific paper and the first author has created a table comparing different papers/methods.

You are reading the table and trying to infer the intent behind why this table was created.

Given a set of candidate intents, your task is to select the best one.

Assign a score from 1 to 5 for each candidate.

Prioritize intents that are:
- specific
- actionable
- closely aligned with the table contents

[Table]
{table}

[Candidate goals]
{goal_text}

Return the output in the following JSON format:

{
  "justification": "...",
  "best_goal": "..."
}
""".strip()