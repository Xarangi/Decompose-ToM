from llm_utils import *

PERSPECTIVE_PROMPT = """\
The following is a dialogue scenario some characters.
Your job is to output only the events that the specified character, {character}, knows about.
Here are a few rules/assumptions:
1. An agent knows a dialogue if they are in the same location or conversation.
2. An agent knows all dialogues they say themselves.
Story:

{story}

What events does {character} know about? Only output the events according to the above rules, do not provide an explanation."""


SIM_PROMPT = """\
Return an answer to the below scenario and question:

{perspective}

You are {name}.
Based on the above information, and the following rules, answer the following question:

Here are a few rules/assumptions:
1. You don't know dialogues said before you enter a conversation, or after you exit a conversation (but you may re-joinand become aware again)
2. You don't know the answer to the question if you don't see a reference to it in the story you know.
3. Choose one of the choices from the given options to return your answer. Return the associated letter label of your choice(from A,B) alongside your choice.

Question:
{question}

Answer in the given format:
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
    def __init__(self, llm, context:str, debug=False, simModel=None):
        self.llm = llm
        self.agentNames = []
        self.agents:Dict[str, Agent] = {}                 
        self.story = context                
        self.perspectives : Dict[str, str] = {}           
        self.debug = debug                                
        self.simModel = simModel
        self.disamb =""

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
            return self.llm.get_output(f"{self.story}\nBased on the above information, answer the following question:\n{question}. Answer in the given format: Format: <option_letter>: <answer>. Answer:"), "Truth Question"
        else:
            # Here we ask the agent to simulate.
            return self.agents[agentName].evalQuestion(question = question, disamb = self.disamb)
        

# Evaluation function
def evalQuestion(llm:LanguageModel, context:str, question:str, debug=False, simModel=None) -> Tuple[str, str]:
    """End to end function for Tomi evaluation.
    """
    # Create world for perspective taking
    world = World(llm, context, debug=debug, simModel=simModel)
    
    # What's the subject of the question (who's perspective do we have to take?)
    questionSubject = question.split(" ")[2]
    # What are the characters in this story?
    world.parseCharacters()
    
    # Is this a reality/memory question?
    if questionSubject not in world.agentNames:
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