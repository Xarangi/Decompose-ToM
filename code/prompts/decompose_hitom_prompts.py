# Prompts for Deciding Knowledge of Statements/Dialogues
PROMPT_DECIDE_KNOWLEDGE_HITOM = '''
This is a given story:
{disamb}
{story}

The story is sequential with each statement happening after the previous one (if the statement is an event). 
This is the next statement in the story: Statement: {part}. 
Your task is to indicate whether {agent} knows about the statement happening, using the following rules:

Use the provided world state to check the location of {agent} and other agents that may be involved to determine knowledge of the given statement. It is formatted as Location 1: [Agents in location], Location 2:[] ...: 
World State: {glob_world_model} 

Rules:
The agent {agent} knows of any statement that mentions their own actions.
The agent {agent} knows of a statement if the statement happens in the same location as them.
The agent {agent} knows of statements that indicate another agent leaving a location.
The agent {agent} does NOT know of a statement if they have left the location where the event occurs or are not in the same location as the agent involved in the statement.
The agent {agent} only knows of a 'private communication' if they are involved in it : someone says something to someone else.
The agent {agent} is aware of all 'public communications' : when someone declares something to everyone.

If a statement can be interpreted ambiguously, then say yes.

Reason briefly using the rules, and indicate your answer about whether {agent} knows the next statement that occurs in the story in the format: Answer: <decision> (where decision is yes/no)
Answer: 
'''

# Prompts for Updating World State
PROMPT_UPDATE_WORLD_HITOM = '''
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

# Prompts for Answering Questions
PROMPT_ANSWER_HITOM = '''
Read the following story and answer the question. Think step-by-step and then provide the answer.

Story: 
{disamb}
{story}

You are {agent}. Based on the above information, and the following rules, answer the following question:

Rules:
{note}

Question: {question}
Choices: {choices}

Provide the relevant label alongside the answer when providing your answer (<option_label>: <answer>).
'''

# Prompts for Extracting Agent Selection from Answer
PROMPT_EXTRACT_HITOM_SELECTION = '''
Provide the answer selected in the above solution. Answer with ONLY the correct choice. The answer should contain only a single word.
                                       
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


