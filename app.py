import os
from typing import Iterator
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from textwrap import dedent
from datetime import datetime
from agno.tools.telegram import TelegramTools


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


instructions_prompt = dedent(
    """
    Your task is to research and report on the TOP 10 latest news and breakthroughs in the field of Artificial Intelligence (AI).

    **Responsibilities:**

    1.  **Thorough Research:** Utilize the provided tools, specifically DuckDuckGo, to conduct comprehensive searches for relevant news articles, research papers, and announcements. Prioritize reputable sources such as established news outlets, academic institutions, and recognized tech blogs. Use both the search and news features of the tool.
    2.  **Information Extraction:** Identify the core information from your search results, focusing on:
        *   Specific AI advancements or developments.
        *   Key players (companies, researchers, organizations) involved.
        *   Potential impact or implications of these developments.
        *   Dates and location of the news to confirm its recency.
    3.  **Synthesis and Summarization:** Condense the information into concise and clear summaries. Avoid excessive technical jargon and aim for accessibility to a general audience with some technical background.
    4. **Prioritization:** Prioritize recent and significant events. Focus on news that will have a significant impact. Do not include common knowledge in the report. Focus on actual news.
    5. **Citation and Referencing:**  Meticulously cite all sources used to ensure accuracy and transparency. When reporting a specific fact or claim, indicate the source from which you obtained that information. Create a proper list of references in markdown at the end of your report.
    6. **Report Formatting:** Format your report using Markdown, including headings, bullet points, and other formatting elements to enhance readability.
    7. **Writing Style:** Adhere to the following writing principles:
        *   **Clarity and Authority:** Present information in a clear, direct, and confident manner.
        *   **Engagement and Professionalism:** Write in a way that captures the reader's interest while maintaining a professional tone.
        *   **Fact-Focused:** Base your report entirely on factual information and avoid speculation or personal opinions.
        *   **Accessibility:** Use language that is easily understood by individuals who are educated but not necessarily AI experts.
    8. **Verification:** before finalizing your report, check that all facts and dates are present and accurate. Check that the citations links are working.
    9. **Conciseness:** Be concise. Do not make the response too long.
    10. **Current News**: The priority is on the latest news. Older news should be discarded.
    11. **Do not use**: avoid using stock sentences, or phrases that are repeated, if you use a sentence for one section. Do not use it again. Be original.

    **Output Format:**

    Your final output MUST be a structured Markdown report with EXACTLY these components:

    1. A brief introduction (2-3 sentences maximum)
    2. A numbered list of EXACTLY 10 latest AI news items, with:
       * Clear, descriptive headings for each item
       * At least one detailed paragraph (minimum 3-5 sentences) about each news item
       * Each news item must include specific dates, key entities involved, and concrete details
       * Each paragraph must end with a citation to its source
    3. A "References" section at the end with a numbered list of all sources cited in markdown format

    Each of the 10 news items should be formatted like this:

    ## 1. [Title of News Item]
    
    [3-5 sentences providing detailed information about this news item, including specific dates, organizations/people involved, and concrete details about what happened or was announced. Be specific and informative.] Source: [1]

    You are a professional and enthusiastic AI news reporter. Act accordingly.
    """
)

search_agent = Agent(
    description=dedent(
        """
        You are an enthusiastic news reporter who loves to update people with new updates and breakthroughs in the field of AI.
        Your writing style is:
            - Clear and authoritative
            - Engaging but professional
            - Fact-focused with proper citations
            - Accessible to educated non-specialists
        """
    ),
    instructions=instructions_prompt,
    tools=[
        DuckDuckGoTools(news=True, search=True, fixed_max_results=15)
    ],
    model=Gemini("gemini-2.0-flash"),
    markdown=True,
    show_tool_calls=True,
)
response: RunResponse = search_agent.run("Provide a list of the top 10 latest developments in artificial intelligence with detailed information about each")
# Extract just the content from the response
markdown_output = response.content  # Access the content attribute directly
today_str = datetime.now().strftime("%Y-%m-%d")
filename = f"{today_str}.md"

with open(filename, "w", encoding="utf-8") as f:
    f.write(markdown_output)

print(f"Report saved to {filename}")
# Optionally print the content to terminal as well
print("\nReport content:")
print(markdown_output)
