PROMPT_DECIDE_KNOWLEDGE_FANTOM = '''
This is a given conversation:
{story}
                                        
The story is sequential with each dialogue happening after the previous one. 
This is the next dialogue in the story: Dialogue: {part}. 
Your task is to indicate whether {agent} knows about the dialogue, using the following rules:

Use the provided world state to check the location of {agent} and other agents that may be involved to determine knowledge of the dialogue. It is formatted as Location 1: [Agents in location], Location 2:[] ...: 
World State: {glob_world_model} 

Rules:
The agent {agent} knows a dialogue if they are in the same location or conversation.
The agent {agent} knows all dialogues they say themselves.
If {agent}'s location is unclear or not provided, assume they know of the dialogue.

Give a single word yes/no answer
Answer: 
'''


PROMPT_UPDATE_WORLD_FANTOM = '''
This is the current world state, that holds the current world location of all the agents:
World State: {glob_world_model}. Please update it relevantly (if needed) after the given dialogue: {part}.

Follow the rules in completing the task:
No updates are needed if an agent does not enter or exit the conversation in the given statement.
An agent exits/enters a conversation only when they mention leaving/entering themselves in the given dialogue.
The agent does not exit a location themselves if they only indicate someone else may be leaving.
In case an update isn't needed return the given world state.
Ensure that no agent is in 2 locations, and only in the correct location. 
The format looks like this: in_conversation: [Agent 1, Agent 2], out_of_conversation: [Agent 3, Agent 4] ....
Use the square brackets appropriately to indicate the agents inside a location. Only return the world state in the given format and no other text.
Answer: World State:
'''




PROMPT_ANSWER_FANTOM = '''
You are {agent}. Here is a conversation between individuals who have just met from the perspective of the given agents: 
                                    
{agents}

{story}
                                    
Answer the following question about it shortly by using the given rules to guide your reasoning.
                                     
Question: {question}, Choices: {choices}, 

Rules: 
You don't know dialogues said before you enter a conversation, or after you exit a conversation (but you may re-join and become aware again)
You don't know the answer to the question if you don't see a reference to it in the story you know.
Choose one of the choices from the given options to return your answer. Return the associated letter label of your choice(from A,B) alongside your choice.

Answer: 
'''

PROMPT_EXTRACT_FANTOM_SELECTION = '''
What selection (in A,B) does the given answer make? Return a single letter with no other text:

Examples: 
The Answer: The answer is A. Choice: A
The Answer: lorem ipsum dolor sit amet ........ So, I choose B, Choice: B

Task:
The Answer: {ans}. Choice: 
'''



