from openai import OpenAI
import copy
import re
import time
from llm_utils import LanguageModel
from prompts.decompose_fantom_prompts import *
from prompts.decompose_hitom_prompts import *
from prompts.decompose_generic_prompts import *

class TheoryOfMindSystem:
    def __init__(self, mode: str | None = None, delimiter: str | None = None, model: str = "gpt-4o", model_type: str = "openai"):
        """
        Initialize the TheoryOfMindSystem.

        :param mode: Operation mode - 'hitom','fantom' or None
        """
        self.memory = {}
        self.agent = None
        self.qn = None
        self.story = None
        self.choices = None
        self.disamb = None
        self.locations = None
        self.counter = 0
        self.delimiter= "."
        self.mode=""
        self.counter = None
        self.model = LanguageModel(model_name=model, model_type=model_type)
        if mode:
            self.mode=mode
        else:
            if not delimiter:
                print("Error: Delimiter not set. Using default delimiter (.)")
            else:
                self.delimiter=delimiter


    # ---------------------- Setup World ----------------------

    def setup_world(self, story: str) -> str:
        if self.mode == 'fantom':
            chars = self.get_response(f"Return a comma separated list of agents who are participating in the given conversation at the start (before anyone else enters the conversation) \nConversation: {story}")
            world = f"in_conversation: [{chars}], out_of_conversation: [insert agents]"
            return world
        else:
            state = self.get_response(PROMPT_SETUP_WORLD_STORY.format(story=story))
            self.locations = state
            states = state.strip().strip(".").split(",")
            world = ""
            for loc in states:
                world += f"{loc.strip()}: [insert agents], "
            world += "Unknown: [insert agents]"
            return world

    # ---------------------- Disambiguate Story (Only for 'hitom') ----------------------

    def disambiguate_story(self, story: str) -> list:
        """
        Analyzes the story to find ambiguous location references and adds sentences
        to disambiguate them at the beginning of the story.

        :param story: List of sentences representing the story.
        :return: Updated story with disambiguating sentences at the beginning.
        """
        location_map = {}
        parent_location = None
        updated_story = []

        for sentence in story:
            character_entry_match = re.search(r"\bentered the ([\w_]+)(?=\.|$)", sentence)
            if character_entry_match:
                parent_location = character_entry_match.group(1)

            location_mention_match = re.search(r"\bis in the ([\w_]+)(?=\.|$)\b", sentence)
            if location_mention_match:
                location = location_mention_match.group(1)
                if location not in location_map:
                    location_map[location] = parent_location

            object_move_match = re.search(r"\b(moved the [\w_]+) to the ([\w_]+)(?=\.|$)\b", sentence)
            if object_move_match and parent_location:
                location = object_move_match.group(2)
                if location not in location_map:
                    location_map[location] = parent_location

        disambiguation_sentences = [
            f"The {location} is in the {parent_location}."
            for location, parent_location in location_map.items()
        ]

        return disambiguation_sentences

    # ---------------------- Get Agent ----------------------

    def get_agent(self, question: str) -> str:
        prompt = PROMPT_GET_AGENT.format(question=question)
        char = self.get_response(prompt).strip()
        if len(char.split(" ")) > 1:
            char = self.get_response(PROMPT_HANDLE_MULTIWORD_AGENT.format(response=char))
        if char.lower() in ["you", "i", "we"]:
            char = "Narrator"
        return char.strip().strip(".").lower()

    # ---------------------- Simulate Question ----------------------

    def sim_question(self, question: str, agent_name: str) -> str:
        prompt = PROMPT_SIM_QUESTION.format(agent_name=agent_name, question=question)
        qn = self.get_response(prompt)
        return qn

    # ---------------------- Decide Knowledge ----------------------

    def decide(self, story: str, part: str, agent: str, glob_world_model: str, note: str):
        if self.mode == 'hitom':
            prompt = PROMPT_DECIDE_KNOWLEDGE_HITOM.format(
                disamb=self.disamb,
                story=story,
                part=part,
                agent=agent,
                glob_world_model=glob_world_model
            )
            decision = self.get_response(prompt)
        elif self.mode == 'fantom':
            prompt = PROMPT_DECIDE_KNOWLEDGE_FANTOM.format(
                story=story,
                part=part,
                agent=agent,
                glob_world_model=glob_world_model
            )
            decision = self.get_response(prompt)
        else:
            prompt = PROMPT_DECIDE_KNOWLEDGE_GENERIC.format(
                story=story,
                note=note,
                part=part,
                agent=agent,
                glob_world_model=glob_world_model
            )
        ans = "yes"  # Default answer
        match = re.search(r'Answer: (\w+)', decision)
        if match:
            ans = match.group(1).strip(".").lower()
        else:
            dec = self.get_response(PROMPT_YES_NO_DECISION.format(decision=decision))
            ans = dec.strip().strip(".").lower()

        if ans not in ["yes", "no"]:
            ans = self.get_response(PROMPT_AMBIGUOUS_DECISION.format(decision=decision))
            ans = ans.strip().strip(".").lower()
            if ans not in ["yes", "no"]:
                ans = "yes"

        ret = ans == "yes"

        # Update World Model if necessary
        if_update = False
        if self.mode == 'hitom':
            prompt_check = f"Does the given statement involve an agent (or multiple agents) entering or exiting a location?\n Statement: {part}. \nAnswer in only yes/no with no other text\n Answer: : "
            if_update_decision = self.get_response(prompt_check)
            if if_update_decision.strip().strip(".").lower() != "no":
                prompt_update = PROMPT_UPDATE_WORLD_HITOM.format(
                    glob_world_model=glob_world_model,
                    part=part
                )
                glob_world_model = self.get_response(prompt_update)
        elif self.mode == 'fantom':
            prompt_check = f"Does the given dialogue involve the speaker leaving the conversation?\n Dialogue: {part}. \nAnswer in only yes/no with no other text\n Answer: : "
            if_update_decision = self.get_response(prompt_check)
            if if_update_decision.strip().strip(".").lower() != "no":
                prompt_update = PROMPT_UPDATE_WORLD_FANTOM.format(
                    glob_world_model=glob_world_model,
                    part=part
                )
                glob_world_model = self.get_response(prompt_update)
        else:
            prompt_check = f"Does the given statement involve an agent (or multiple agents) entering or exiting a location?\n Statement: {part}. \nAnswer in only yes/no with no other text\n Answer: : "
            if_update_decision = self.get_response(prompt_check)
            if if_update_decision.strip().strip(".").lower() != "no":
                prompt_update = PROMPT_UPDATE_WORLD_GENERIC.format(
                    glob_world_model=glob_world_model,
                    part=part
                )
                glob_world_model = self.get_response(prompt_update)

        return ret, glob_world_model

    # ---------------------- Data Processing ----------------------

    def data(self, story: str, agent_name: str, note: str) -> str:
        if self.mode == 'hitom':
            story_parts = story.split(".")
            updated_story = []
            glob_world_model = self.setup_world(story)
            storyx = ""
            self.disamb = "\n ".join(self.disambiguate_story(story))
            for part in story_parts:
                part = part.strip()
                if not part:
                    continue
                decision, glob_world_model = self.decide(
                    storyx, part, agent_name, glob_world_model, note
                )
                storyx += ". " + part
                if decision:
                    updated_story.append(part)

            ans = ". ".join(updated_story) + "."
            return ans

        elif self.mode == 'fantom':
            story_parts = story.split("\n")
            updated_story = []
            glob_world_model = self.setup_world(story)
            storyx = ""

            for part in story_parts:
                part = part.strip()
                if not part:
                    continue
                decision, glob_world_model, = self.decide(
                    storyx, part, agent_name, glob_world_model, note
                )
                storyx += "\n" + part
                if decision:
                    updated_story.append(part)

            ans = "\n".join(updated_story) + "\n"
            return ans
        else:
            story_parts = story.split(self.delimiter)
            updated_story = []
            glob_world_model = self.setup_world(story)
            storyx = ""
            for part in story_parts:
                part = part.strip()
                if not part:
                    continue
                decision, glob_world_model = self.decide(
                    storyx, part, agent_name, glob_world_model, note
                )
                storyx += self.delimiter + part
                if decision:
                    updated_story.append(part)

            ans = self.delimiter.join(updated_story) + self.delimiter
            return ans

    # ---------------------- Answering Questions ----------------------

    def answer(self, story: str, agent: str, answer_context: str, question: str, choices: str, note: str) -> str:
        if self.mode == 'hitom':
            prompt = PROMPT_ANSWER_HITOM.format(
                disamb=self.disamb,
                story=story,
                agent=agent,
                note=note,
                question=question,
                choices=choices
            )
        elif self.mode == 'fantom':
            prompt = PROMPT_ANSWER_FANTOM.format(
                agent=agent,
                agents=answer_context,
                story=story,
                question=question,
                choices=choices
            )
        else:
            prompt = PROMPT_ANSWER_GENERIC.format(
                agent=agent,
                story=story,
                note=note,
                question=question,
                choices=choices
            )

        ans = self.get_response(prompt)

        if self.mode == 'hitom':
            choice_prompt = PROMPT_EXTRACT_HITOM_SELECTION.format(ans=ans)
        elif self.mode == 'fantom':
            choice_prompt = PROMPT_EXTRACT_FANTOM_SELECTION.format(ans=ans)
        else:
            choice_prompt = PROMPT_EXTRACT_GENERIC_SELECTION.format(ans=ans)

        answer = self.get_response(choice_prompt).strip().strip(".").lower()
        return answer

    # ---------------------- Start Task ----------------------

    def start_task(self, story: str, question: str, choices: str, note: str, max_recursion: int | None = None) -> str:
        """
        Main function to process the theory of mind task.

        :param story: The story or conversation.
        :param question: The question to answer.
        :param choices: The multiple-choice options.
        :param note: Additional rules or notes.
        :param max_recursion: optionally set a maximum recursion level for the algorithm.
        :return: The selected answer.
        """
        self.agent = self.get_agent(question)
        self.qn = question
        self.story = story
        self.choices = choices
        self.note = note
        if max_recursion:
            self.counter=max_recursion
        return self.task(story, self.qn, self.agent, "", self.choices, note)

    # ---------------------- Task Recursion ----------------------

    def task(self, story: str, question: str, last_agent: str, answer_context: str, choices: str, note: str) -> str:
        """
        Recursive task processing based on mode.

        :param story: The current story or conversation.
        :param question: The current question.
        :param last_agent: The last agent processed.
        :param answer_context: Provide relevant extra context for the answer stage.
        :param choices: Multiple-choice options.
        :param note: Additional rules or notes.
        :return: The selected answer.
        """
        agent = self.get_agent(question)
        if agent.lower() == "narrator":
            return self.answer(story, last_agent, answer_context, question, choices, note)

        qn = self.sim_question(question, agent)
        updated_story = self.data(story, agent, note)
        if self.counter:
            if self.counter == 0:
                return self.answer(story, last_agent, answer_context, question, choices, note)
            self.counter -= 1
        if self.mode == 'fantom':
            answer_context += f"{agent} believes: "
        return self.task(updated_story, qn, agent, answer_context, choices, note)

    # ---------------------- Get Response Method ----------------------

    def get_response(self, prompt: str) -> str:
        """
        Get response based on the current mode.

        :param prompt: The prompt to send.
        :return: The generated response.
        """
        return self.model.get_output(prompt)

