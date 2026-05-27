S2_SYSTEM_PROMPT = """
You are an intelligent and precise assistant that can understand the contents of research papers.
You are knowledgable on different fields and domains of science, in particular computer science.
You are able to interpret research papers, create questions and answers, and compare multiple papers.
""".strip()

GENERATE_SYNTHETIC_GOALS_PAPERS_QUESTION = '''
When writing a scientific research paper, we often include tables comparing different works to accomplish a variety of goals. The author has this goal in mind when they create the table for what they want to convey to the reader via the objective comparison of papers. \
For example, some potential goals might include: \
1. Highlighting gaps in existing research: By comparing related studies, the table can show areas where there is limited research or unresolved questions, positioning the current study as addressing those gaps. \
2. Contextualizing the study: It helps place the current research within the broader scientific context, showing how it builds upon or differs from previous work. \
3. Evaluating methodology differences: It allows for an easy comparison of the methodologies used in different studies, illustrating why the chosen methods in the current paper are innovative, more robust, or better suited for the research problem. \
4. Demonstrating novelty: By showing what has already been done, a comparison table emphasizes the unique contribution or novelty of the present study. \
5. Assessing the consistency of results: The table can highlight differences or consistencies in findings across studies, helping the reader understand how results align or contrast with existing literature. \
6. Simplifying complex information: It makes it easier for readers to quickly grasp how various studies relate to one another, especially when reviewing large bodies of literature. \
7. Supporting the literature review: It strengthens the literature review by systematically summarizing relevant research, which aids in the argument for why the current study is needed. \
Generally this goal can be written down as a simple open-ended question that the author anticipates that the reader will have and that can be answered with the table. \
Your task is to generate this goal given a particular table from a research paper. You are also given the title and abstract of the paper, the description of the table and additional information about how the table is referenced in the text of the paper. \
[Table] {table}
We also provide information about the papers being discussed in the table. You want the goal to be one that helps a future user actionably create the table given the information in these papers: \
{papers}
Return output in the following JSON format: {{'goal':<your goal>, 'justification':<justification of the goal>}}
'''.strip()

EEVALUATE_GOALS_TO_TABLE = '''
Imagine you are a co-author of a scientific paper and the first author has created a table comparing different papers/methods. You are reading the table along with the caption of the paper and references to the table in the text of the paper. You are trying to guess what is the intent with which your co-author created this particular table. \
Given a set of candidate intents that you think they might have had, your task is to select the best user intent out of them. Assign a score to each candidate on a scale of 1 to 5 on how well it fits what they might have thought. Prioritize selecting a user intent that is highly specific to the particular information in the table. The output format is a JSON with a string valued justification containing the scores assigned to each candidate schema along with why that score was assigned. You should also provide your final choice of the best schema. If you feel that none of them are good, then reply with None here. \
[Table] {table} \
[Candidate goals] {goal_text} \
Return the output in the following JSON format. The justification should include the reasoning for the score as a string, the best_goal should be the text of the best candidate and nothing else: {{'justification':<justification for the score>, 'best_goal':<the best candidate selected>}} \
'''.strip()