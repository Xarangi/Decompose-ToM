# Prompts for Agent Identification
PROMPT_GET_AGENT = '''
Based on the given question, which agent's belief or perspective do we want to find first? Use the given rules to name the agent:
Rules:
If the question does not mention the name of any agents, the answer should be Narrator 
Otherwise, output the primary agent's name. (Pronouns such as you/I/we/they/us aren't agent names and should not be outputted)

Examples:

Question: Where does Alex think Raj looks for the jam?
Agent Name: Alex

Question: Where do I think Sam thinks the ladder is?
Agent Name: Sam

Question: Where does Ava think Sophie thinks Sam thinks Brad thinks the cookie is?
Agent Name: Ava

Question: Where is the ladder?
Agent Name: Narrator

Question: Where do they think the ladder is?
Agent Name: Narrator

Task:
Question: {question}
Agent Name: 
'''

# Prompts for Simplifying Questions
PROMPT_SIM_QUESTION = '''
Reframe the question's perspective as if it was being asked directly to {agent_name} by framing another agent as the subject of the question. Don't mention {agent_name}'s name or use pronouns referring to them, instead make the question direct by removing their perspective. If there are no agents that can be made the subject, make it a direct question (Example: Where is X?) Only use 'you' when it's necessary and there are no other agents that can be framed as the subject. Output just the question and nothing else.

Examples: 

Question: Where does {agent_name} think Alex will look for the chocolate?
New Question: Where will Alex look for the chocolate?

Question: Where does {agent_name} find the apple?
New Question: Where is the apple?

Question: Where does {agent_name} think Brandon thinks Cody thinks the banana is?
New Question: Where does Brandon think Cody thinks the banana is?

Task:                     
Question: {question}
New Question: 
'''

PROMPT_DECIDE_KNOWLEDGE_GENERIC = '''
This is a given story:
{disamb}
{story}

The story is sequential with each statement happening after the previous one (if the statement is an event). 
This is the next statement in the story: Statement: {part}. 
Your task is to indicate whether {agent} knows about the statement happening, using the following rules:

Use the provided world state to check the location of {agent} and other agents that may be involved to determine knowledge of the given statement. It is formatted as Location 1: [Agents in location], Location 2:[] ...: 
World State: {glob_world_model} 

Rules:
{note}

If a statement can be interpreted ambiguously, then say yes.

Reason briefly using the rules, and indicate your answer about whether {agent} knows the next statement that occurs in the story in the format: Answer: <decision> (where decision is yes/no)
Answer: 
'''

PROMPT_UPDATE_WORLD_GENERIC = '''
This is the current world state, that holds the current world location of all the agents:
World State: {glob_world_model}. Please update it relevantly (if needed) after the given statement: {part}.

Follow the rules in completing the task:
No updates are needed if an agent does not enter or exit a location in the given statement.
An agent exits a location only when mentioned in the given statement. In that case, add the agent to the location "Unknown" and remove them from their original location.
In case an update isn't needed return the given world state. Only update the state for agents and not objects.
Ensure that no agent is in 2 locations, and only in the correct location. 
The format looks like this: Location 1: [Agent 1, Agent 2] , Location 2: []  and so on ....
Use the square brackets appropriately to indicate the agents inside a location. Only return the world state in the given format and no other text.
Answer: World State:
'''

PROMPT_ANSWER_GENERIC = '''
Read the following story and answer the question. Think step-by-step and then provide the answer.

Story: 
{story}

You are {agent}. Based on the above information, and the following rules, answer the following question:

Rules:
{note}

Question: {question}
Choices: {choices}

Provide the relevant label alongside the answer when providing your answer (<option_label>: <answer>).
'''

PROMPT_EXTRACT_GENERIC_SELECTION = '''
Provide the answer selected in the above solution. Answer with ONLY the correct choice.
                                       
Format: <option_letter>: <answer> :
                                       
Examples: 
The Answer: The answer is M: Chocolate 
Choice: M: Chocolate
The Answer: lorem ipsum dolor sit amet ........ So, I choose xyz, option L, 
Choice: L: xyz

Task:
The Answer: {ans}. 
Choice: 
'''

# Prompt for Handling Agent Name with Multiple Words
PROMPT_HANDLE_MULTIWORD_AGENT = '''
Answer in one word, what is the agent name mentioned in this response (can be Narrator)? {response} Answer: 
'''

# Prompt for Yes/No Decision Extraction
PROMPT_YES_NO_DECISION = '''
Give a single word answer indicating the choice decided by this reasoning (yes/no):\n Reasoning: {decision}\n Answer: 
'''

# Prompt for Yes/No from Ambiguous Decision
PROMPT_AMBIGUOUS_DECISION = '''
Give a one word answer as to if this sentence indicates the answer is a yes or no. Answer in only yes/no: 
{decision}
Answer: 
'''

PROMPT_SETUP_WORLD_STORY = '''
        Here is a story: 
        {story}. 
        Take note of locations in the story where characters acting in the scenario enter and exit from. Note that locations may be abstract (and not physical locations) but should still be named relevantly (such that the name defines the key characteristic). 
        Output the answer in the format:\n Location1, Location 2, ....\n. Don't return any other text. 
        
        Answer: '''
