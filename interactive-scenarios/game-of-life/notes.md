# Game of Life Interactive Scenario Notes

**First prompt:** The LLM provided a functional implementation using shared memory optimization but without bit packing or lookup tables. This brings us close to completing Milestone 01. Moving forward, we will provide a hint to avoid shared memory, which should simplify the generated code.

**Second prompt**: The LLM successfully generated a solution without shared memory, completing Milestone 01.

**Third prompt**: The LLM produced a correct but naive row-based bitwise kernel. However, it did not include a conversion function, though this was not explicitly requested. Next, we will prompt for the use of the popc function, as other hints from Milestone 02 are unnecessary.

**Fourth prompt**: The LLM used the popc function, but in a nonsensical way. Instead of computing three neighbors at a time, it processed the neighborhood bit by bit, making popc ineffective in terms of performance improvement. Since the script does not specify an ideal outcome, we consider Milestone 02 complete and proceed to the next step.

**Fifth prompt**: The LLM produced a novel and creative approach we did not expect—it computed cells in groups of two. While not the most optimal solution, it is correct and aligns with the requirements in some form. As a final step, we will provide a hint for the full-adder technique.

**Sixth prompt**: The LLM understood the task but produced incorrect output. We attempted to give it a chance to fix the issue but were unable to provide specific guidance on where the mistake might be.

**Seventh prompt**: The code is now correct and successfully implements bitwise full-adder logic. Although it is not the most optimized version seen in "one-shot" prompts, it meets the requirements. We consider the final milestone complete.
