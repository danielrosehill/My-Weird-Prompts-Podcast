# Show Notes 

"My Weird Prompts" is an AI generated podcast Which is distinguished by the unusual format of real human generated prompts provided by me (Daniel Rosehill) Being followed by a AI generators, two host "answer." The idea is for me to have a more enjoyable way to listen to long, detailed answers and sharing it publicly in case anyone else finds it interesting. 

The two hosts Will be two TTS voice clones. 

They are:

Herman Poppleberry: Herman Poppleberry It is a donkey who lives in Jerusalem. He is the more serious and scholarly of the two and will do more talking in each episode. Given that most of the prompts are going to involve technical questions, he will cover the harder parts.

Cornelius (nickname Corn): Corn is Herman's brother. He is a sloth. He has his own voice clone also. His role will be mostly limited to introducing the episodes and turning the question over to Herman. He might occasionally equip in with some observations. Or help to breakdown complex parts of the podcast for a more generalist audience. 

## Notes For TTS Generation

Episodes were generated programmatically from a workflow that works as follows:

- Daniel records a prompt 
- Prompt gets TTS-d or sent to a multimodal LLM which processes and summarises

Then (depending on implementation):

- SSML script is generated diarised for Herman and Corn 
- TTS generation with voice clones 

Once all elements are received:

- A script produces the episode by cominbing intro jingle, prompt, response, outro and also meatdata (show name, desc)

Significance for the SSML/episode transcript generation phase:

The episodes will include Daniel's prompt in their entirety. Therefore, when the hosts begin discussing the question that they're going to answer, they shouldn't spend much time on summarizing the prompt as it will have just been read out. In the system prompt that is used to generate the transcript to be read. The large language model can be instructed on the method of operation and that the host can begin by referencing the prompt "we just heard." (or similar).

## Later Workflow

Right now it'll be a simple prompt answer workflow, but in future iterations we might consider breaking down the prompt into constituent elements so that there is a almost simulated dialog whereby elements of the prompt are sent for discussion.