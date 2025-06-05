
from llm_utils import *
from new_decompose import TheoryOfMindSystem

PERSPECTIVE_PROMPT = """\
The following is a sequence of events about some characters, that takes place in multiple locations.
Your job is to output only the events that the specified character, {character}, knows about.
Here are a few rules/assumptions:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.
4. An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. 
5. Note that every agent tends to lie. What an agent A tells others doesn't affect A's actual belief. An agent tends to trust an agent that exited the room later than himself. The exit order is known to all agents. 
6. Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims."
Story:
{disamb}
{story}

What events does {character} know about? Only output the events according to the above rules, do not provide an explanation."""


SIM_PROMPT = """\
Return a single word answer to the below scenario and question:

{disamb}
{perspective}

You are {name}.
Based on the above information, and the following rules, answer the following question:

Here are a few rules/assumptions:
1. A character knows about all events that they do.
2. If a character is in a certain room/location, that character knows about all other events that happens in the room. This includes other characters leaving or exiting the location, the locations of objects in that location, and whether somebody moves an object to another place.
3. If a character leaves a location, and is NOT in that location, they no longer know about any events that happen within that location. However, they can re-enter the location.
4. An agent A can infer another agent B's mental state only if A and B have been in the same location, or have private or public interactions. 
5. You, or another agent may lie. What an agent A (or you) tells others doesn't affect A's (or your's) actual belief. You can trust an agent that exited the room later than yourself. The exit order is known to all agents. 
6. Agents in private communications know that others won't hear them, but they know that anyone can hear any public claims."

Question:
{question}

Answer with ONLY the correct choice. The answer should contain only a single word.

Format: <option_letter>: <answer>

Answer:
"""

questionPrompt = """\
{question}
Choose from the following:
{choices}
""" 



class Agent:
    """Agent class for simulation.
    """
    def __init__(self, llm, name=""):
        self.name = name
        self.perspective = ""
        self.llm = llm
        self.evalPrompt = SIM_PROMPT
        # DEBUGGING PURPOSES
        self.wasAsked = None
        self.replied = None
        self.debug = False
    def evalQuestion(self, question:str, disamb) -> str:
        """Answers HiToM question based on agent belief.

        Args:
            question (str): HiToM question + answer choices

        Returns:
            str: Answer choice
        """
        prompt = self.evalPrompt.format(perspective=self.perspective, disamb =disamb,name=self.name, question=question)
        # print(prompt)
        choice = self.llm.get_output(prompt)
        self.wasAsked, self.replied = prompt, choice
        return choice, self.perspective # Return perspective for debugging purposes

class World:
    """World class that keeps track of true world state.
    """
    def __init__(self, llm, context:dict, debug=False, simModel=None):
        self.llm = llm
        self.agentNames = []
        self.agents:Dict[str, Agent] = {}                 # Note this is a dictioary for maybe future multi-agent simulations
        self.context = context                            # Original ToMi Question (JSON format)
        self.story = context["story"]                 # ToMI story
        self.perspectives : Dict[str, str] = {}           # Perspectives (Dict for same reason as above)
        self.debug = debug                                # This is verbose
        self.simModel = simModel
        self.disamb = TheoryOfMindSystem(mode="hitom").disambiguate_story(self.story)

        if self.simModel == None:
            self.simModel = self.llm
    
    def parseCharacters(self) -> None:
        # We use ChatGPT here a little bit, just to parse the characters in this story.
        prompt = f"""\
        {self.story}
        What are the characters in this story?
        Output only the character names, separated by commas. Don't output anything else\n Character Names: """
        gpt = LanguageModel("gpt-4o-mini")
        self.agentNames = gpt.get_output(prompt).replace(" ", "").split(",")
        if self.debug:
            print("Agent names:", self.agentNames)

    def takePerspective(self, characterName=None) -> None:
        
        # Here is the perspective-taking.
        prompt = PERSPECTIVE_PROMPT
        # Take perspective for given character.
        self.perspectives[characterName] = self.llm.get_output(prompt.format(story=self.story, disamb =self.disamb, character=characterName))
        if self.debug:
            print(f"Perspective of {characterName}:", self.perspectives[characterName])
        

    def setupAgent(self) -> None:
        """Create agents.
        """
        for agentName, perspective in self.perspectives.items():
            agent = Agent(self.simModel, name=agentName)
            agent.perspective = perspective
            # Add to agent dictionary
            self.agents[agentName] = agent
        

    def evalQuestion(self, question, agentName=None) -> str:
        if agentName is None:
            # This is a question about the truth.
            # For truth questions, there's really no point of simulation/perspective taking, so we just ask the LLM the question (same as baseline).
            return self.llm.get_output(f"{self.story}\nBased on the above information, answer the following question:\n{question}"), "Truth Question"
        else:
            # Here we ask the agent to simulate.
            return self.agents[agentName].evalQuestion(question = question, disamb = self.disamb)
        

# Evaluation function
def evalQuestion(llm:LanguageModel, context:dict, question:str, debug=False, simModel=None) -> Tuple[str, str]:
    """End to end function for Tomi evaluation.
    """
    # Create world for perspective taking
    world = World(llm, context, debug=debug, simModel=simModel)
    
    # What's the subject of the question (who's perspective do we have to take?)
    questionSubject = question.split(" ")[3]
    # What are the characters in this story?
    world.parseCharacters()
    
    # Is this a reality/memory question?
    if questionSubject not in world.agentNames:
        # Question about zeroth-order belief (Reality/Memory question)
        answer, perspective = world.evalQuestion(question)
        return answer, perspective
    
    # Else, run the pipeline.
    
    # 1. Perspective-taking
    world.takePerspective(characterName=questionSubject)
    world.setupAgent()
    
    # 2. Simulation
    answer, perspective = world.evalQuestion(question, agentName=questionSubject)
    
    # ...and we have our answer!
    return answer, perspective