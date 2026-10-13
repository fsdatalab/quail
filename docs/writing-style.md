# Writing documentation

Write so readers can understand the explanation on their first reading. Choose information according to their needs, and provide the context necessary to understand it.

## Understand the reader before choosing content

Before drafting, state who the page is for and what they want to accomplish. Establish what they already know. Identify the background they need to understand the page.

- Answer the reader's question early. Choose that answer according to the page's purpose.
- Respect the reader's existing knowledge. E.g., an audience that understands classification does not need another definition of classification.
- Explain the behavior the reader needs to understand. Describe implementation details only when those details help the intended reader.
- Do not mention vLLM or SGLang backends in the API documentation.
- Choose the depth deliberately. E.g., an API user needs different information from an engineer changing execution code.

## Present an outline before writing

Present the proposed outline before drafting the document. State the intended reader and the page's purpose. Explain what each section contributes.

- Order the explanation according to the reader's needs. Do not follow the implementation history.
- Put necessary background before the explanation that depends on it.
- Check whether every section belongs on the page. Remove unnecessary sections before drafting their prose.
- Fit additions into the existing documentation. Give a new operator the same treatment as comparable operators.

## Choose enough context, then remove irrelevant detail

Assume readers have not followed the PR conversation. Give them the information needed to understand each statement. Avoid repeating background they already know.

- Introduce an object before describing what happens to it. Explain a condition before describing its consequence.
- Introduce examples with "E.g.,".
- Include a detail when readers need it to understand behavior or make a decision.
- Keep explanations complete. Shortening a sentence does not help if readers must guess what you mean.
- Remove implementation history and details unrelated to the reader's task.
- Put model-specific behavior on the model's page. Link to necessary background already explained elsewhere.

## Choose sentence subjects carefully

Choose a subject that identifies what the sentence is about. Name the actor when explaining an action. Use the same subject across related sentences when that helps readers follow the explanation.

- Begin with an object or idea readers can already identify. Introduce new information about it afterward.
- Use present tense to explain behavior. Address readers directly when giving instructions, and keep that form consistent within the passage.
- Replace vague subjects with the specific noun. E.g., use "the returned table" instead of "the result".
- Avoid using an abstract feature name as the subject when you mean a concrete operation or value.

## Use Simplified Technical English

Simplified Technical English, or STE, is a controlled language defined by ASD-STE100. The standard contains writing rules and a dictionary of approved words. Approved words have specified meanings and parts of speech. Technical names and verbs are permitted under defined rules. See the [official explanation of STE](https://www.asd-ste100.org/about.html).

Apply controlled vocabulary deliberately. Do not equate STE with advice to use short sentences.

- Use a word with a consistent meaning. Use the same term for the same concept.
- Choose direct verbs that describe the action. Avoid nouns that hide what happens.
- Use active voice. E.g., write "Quail removes duplicate rows" instead of "Duplicate rows are removed." See [STE rule 3.6](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf).
- Use established technical terms when they are necessary. Explain unfamiliar terms before relying on them.
- Avoid invented shorthand. Replacing an opaque term with another opaque term does not clarify the explanation.
- Verify the relevant vocabulary and rules before claiming STE compliance.

## Keep sentence structure clear

The following rules include preferences from this conversation. They are additional guidance rather than a description of the STE standard.

- Give each sentence one main idea. A second clause may explain a closely related condition or consequence.
- Do not write sentences with three or more clauses. Avoid rhetorical groups of three.
- Write complete sentences. Keep related explanations connected instead of shortening every sentence into a separate instruction.
- State conditions and consequences explicitly. Do not require readers to infer the relationship.
- Use literal descriptions. Remove metaphors and dramatic contrasts.
- Remove empty emphasis and filler. Keep words that explain the behavior.

## Review meaning before formatting

Review the draft from the intended reader's perspective. First check the scope and order. Then read each sentence without relying on knowledge from the PR conversation.

- Check whether readers can identify every subject and understand each necessary term.
- Check whether the previous sentences provide enough context.
- Check whether every clause adds useful information. Delete irrelevant details rather than polishing them.
- Check that the explanation describes the current behavior accurately.
- Check whether readers can accomplish the task stated in the outline.

## Choose formatting for the content

- Use tables to show parameter names and types when a table makes them easier to read.
- Use bullets to explain separate methods or steps when a paragraph would be harder to follow.
