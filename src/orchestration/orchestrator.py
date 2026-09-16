from collections import Counter
import random

from openai import OpenAI

from model.model_utils import engine
from model.openai_compat import OpenAICompatChatWrapper


class OrchestratorAgent:
    def __init__(self, model_name, agent_pool, max_tokens=4096, api_key="none", base_url="http://localhost:8001/v1", debug=False):
        """Initialize the orchestrator agent with an OpenAI-compatible model wrapper."""
        self.model_name = model_name
        self.agent_pool = agent_pool
        self.max_tokens = max_tokens
        self.conversation_history = []

        # A long run is worth more than a fast failure: wait a while for a
        # loaded server, and retry rather than lose the run to one bad call.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=300.0, max_retries=4)

        self.debug = debug

    def select_team(self, tag_profile, tag_frequencies, batch_size=5, prior_md=None,
                    team_size=4, invalid_feedback=None):
        """
        Select a team of agents based on tag profile.

        Args:
            tag_profile: List of all tags from the sampled questions
            tag_frequencies: Dict of tag -> count in the sample
            batch_size: Number of questions in the batch

        Returns:
            Dict with selected agents and reasoning
        """

        name_slots = ", ".join(f'"AgentName{i + 1}"' for i in range(team_size))
        system_prompt = f"""You are an intelligent orchestrator agent responsible for assembling teams of specialized AI agents to solve batches of questions.

Your task is to:
1. Analyze the tag profile of a batch of questions
2. Review the available agent pool and their specialties
3. Recommend a team of {team_size} agents from the pool that can effectively solve the batch
4. Provide your reasoning

Guidelines:
- Prefer diversity: avoid teams with similar agents
- Favor agents whose strengths match the tag profile
- Use the performance summary when one is given: prefer agents with a record on these tags, and try an untested agent when the record is thin
- Copy agent names exactly as written in the pool, including underscores
- Only select agents from the provided agent pool

Output ONLY a JSON object with this exact structure (no markdown, no extra text):
{{
    "selected_agents": [{name_slots}],
    "reasoning": "Brief explanation of why this team was selected, and the reason for each individual agent choice."
}}"""

        user_message = self._create_team_selection_prompt(
            tag_profile,
            tag_frequencies,
            batch_size,
            prior_md=prior_md,
            team_size=team_size,
        )

        if invalid_feedback:
            user_message = (
                f"{user_message}\n\nYour previous answer named agents that are not in the pool: "
                f"{invalid_feedback}. Choose only from the pool listed above and copy the names exactly."
            )

        if self.debug:
            print(f"\n [DEBUG] --- Orchestrator Prompt ---\n{user_message}\n---------------------------\n")

        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.conversation_history,
            max_tokens=self.max_tokens,
            temperature=0.1,
            top_p=0.5,
        )

        return self._parse_team_response(response)

    def _create_team_selection_prompt(self, tag_profile, tag_frequencies, batch_size, prior_md=None,
                                      team_size=4):
        """Create the prompt for team selection."""

        tag_profile_summary = ", ".join(tag_profile) if tag_profile else "none"
        tag_info = "\n".join([
            f"  - {tag}: {count}/{batch_size} questions"
            for tag, count in sorted(tag_frequencies.items(), key=lambda x: x[1], reverse=True)
        ])

        agent_info = "\n".join([
            f"  • {agent['name']}: {agent['specialty']}"
            + (f" (strengths: {', '.join(agent['strengths'][:4])})" if agent.get("strengths") else "")
            for agent in self.agent_pool
        ])

        prompt = f"""You have a batch of {batch_size} questions with the following tag profile:

Tag profile summary: {tag_profile_summary}
Tags present in this batch:
{tag_info}

Available agent pool:
{agent_info}

Based on this tag profile, select the best team of {team_size} agents from the agent pool to solve these questions.
Consider the dominant tags, the complementary strengths of agents.

Output only valid JSON with no additional text."""
        # If prior markdown summary is available, include it to inform selection
        if prior_md:
            print(f"--- PRIOR MD FOUND ---\n")
            # Prepend a short instruction and then include the prior summary
            prompt = (
                "Previous agent performance summary (inform your selection):\n\n"
                + prior_md
                + "\n\n"
                + prompt
            )
        else:
            print(f"--- NO PRIOR MD FOUND ---\n")

        return prompt
    
    def _parse_team_response(self, response):
        """Parse the agent response to extract team selection"""
        import json

        if self.debug:
            print(f"\n [DEBUG] --- Orchestrator Response ---\n{response}\n---------------------------\n")
        
        try:
            # Try to extract JSON from the response
            response_text = response.choices[0].message.content.strip()
            
            # If the response is wrapped in markdown code blocks, remove them
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            response_text = response_text.strip()
            
            # Parse JSON
            data = json.loads(response_text)
            
            selected_agents = data.get("selected_agents", [])
            reasoning = data.get("reasoning", "")
            
            pool_names = [agent["name"] for agent in self.agent_pool]
            valid_agents, invalid_agents = [], []
            for name in selected_agents:
                if name in pool_names and name not in valid_agents:
                    valid_agents.append(name)
                elif name not in pool_names:
                    invalid_agents.append(name)

            if invalid_agents:
                print(f"⚠️  Warning: agents not in the pool: {invalid_agents}")

            return {
                "agents": valid_agents,
                "invalid_agents": invalid_agents,
                "reasoning": reasoning,
                "reasoning trace": response.choices[0].message.reasoning if hasattr(response.choices[0].message, 'reasoning') else "No reasoning available",
            }
        
        except json.JSONDecodeError:
            print(f"⚠️  Warning: Could not parse JSON response. Using default team.")
            print(f"Raw response: {response_text[:200]}")
            return {
                "agents": [],
                "invalid_agents": [],
                "reasoning": "Parsing failed - no team selected",
                "reasoning trace": response.choices[0].message.reasoning if hasattr(response.choices[0].message, 'reasoning') else "No reasoning available",
            }
        

def sample_tag_questions(dataset, num_samples=5):
    """Randomly choose a tag and sample matching questions."""
    all_tags = []
    for tags_list in dataset['tags']:
        all_tags.extend(tags_list)

    if not all_tags:
        print("Error: No tags found in dataset")
        return None

    chosen_tag = random.choice(all_tags)
    print(f"\nRandomly selected tag: '{chosen_tag}'")

    questions_with_tag = []
    for idx, tags_list in enumerate(dataset['tags']):
        if chosen_tag in tags_list:
            questions_with_tag.append(idx)

    print(f"Found {len(questions_with_tag)} questions with tag '{chosen_tag}'")

    if len(questions_with_tag) < num_samples:
        num_samples = len(questions_with_tag)
        print(f"Adjusting sample size to {num_samples} (fewer questions available)")

    sampled_indices = random.sample(questions_with_tag, num_samples)
    sampled_questions = dataset.select(sampled_indices)

    return {
        "sampled_questions": sampled_questions,
        "chosen_tag": chosen_tag,
    }


def team_selection(orchestrator: OrchestratorAgent, dataset, num_samples=5, prior_md=None,
                   team_size=4, max_attempts=2):
    """
    Select a team of agents for a batch of questions using the orchestrator.

    Args:
        orchestrator: An instance of OrchestratorAgent
        dataset: A list of question dicts, each with 'question', 'answer', and 'tags'
        num_samples: Number of questions to sample for team selection
    """

    sample_result = sample_tag_questions(dataset, num_samples)
    sampled_questions = sample_result["sampled_questions"]
    chosen_tag = sample_result["chosen_tag"]

    all_tags_in_sample = []
    for tags_list in sampled_questions['tags']:
        all_tags_in_sample.extend(tags_list)

    tag_frequencies = Counter(all_tags_in_sample)
    tag_profile = list(tag_frequencies.keys())

    print(f"\nSampled {num_samples} questions")
    print(f"Tag profile from sampled questions:")
    for tag, count in sorted(tag_frequencies.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {tag}: {count}/{num_samples}")

    # Retry once when the model names agents that are not in the pool, telling it
    # which names were wrong; an unretried miss silently shrinks the team.
    invalid_feedback = None
    for attempt in range(1, max_attempts + 1):
        team_result = orchestrator.select_team(
            tag_profile,
            tag_frequencies,
            batch_size=num_samples,
            prior_md=prior_md,
            team_size=team_size,
            invalid_feedback=invalid_feedback,
        )
        if len(team_result.get("agents", [])) >= team_size:
            break
        invalid = team_result.get("invalid_agents") or []
        if not invalid or attempt == max_attempts:
            break
        print(f"Retrying team selection (attempt {attempt + 1}/{max_attempts})")
        invalid_feedback = ", ".join(invalid)

    return {
        "chosen_tag": chosen_tag,
        "batch_size": num_samples,
        "sampled_questions": sampled_questions,
        "tag_profile": tag_profile,
        "tag_frequencies": tag_frequencies,
        "selected_team": team_result.get("agents", []),
        "invalid_agents": team_result.get("invalid_agents", []),
        "reasoning": team_result.get("reasoning", ""),
        "reasoning trace": team_result.get("reasoning trace", ""),
        "team_selection": team_result,
    }

    
