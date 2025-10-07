GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# for legal domain dataset, it can be uncommented for getting more better entity extraction
# PROMPTS["DEFAULT_ENTITY_TYPES"] = [
#     "Organization", "Person", "Date", "Jurisdiction", "LegalDocument",
#     "Clause", "Obligation", "MonetaryAmount", "Regulation", "Role"
# ]

# PROMPTS["entity_extraction"] = """-Goal-
# Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
# Use {language} as output language.

# -Steps-
# 1. Identify all entities. For each identified entity, extract the following information:
# - entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
# - entity_type: One of the following types: [{entity_types}]
# - entity_description: Comprehensive description of the entity's attributes and activities
# Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

# 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
# For each pair of related entities, extract the following information:
# - source_entity: name of the source entity, as identified in step 1
# - target_entity: name of the target entity, as identified in step 1
# - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
# - relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
# - relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
# Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

# 3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
# Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

# 4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

# 5. When finished, output {completion_delimiter}

# ######################
# -Examples-
# ######################
# {examples}

# #############################
# -Real Data-
# ######################
# Entity_types: {entity_types}
# Text: {input_text}
# ######################
# Output:
# """

# PROMPTS["entity_extraction_examples"] = [
#     """Example 1:

# Entity_types: [Organization, Person, Date, LegalDocument, MonetaryAmount, Role]
# Text:
# "The Board of Directors of Tiffany & Co. and the Board of Directors of Merger Sub have unanimously approved the Merger. At the Effective Time, each issued and outstanding share of Tiffany common stock will be converted into the right to receive $131.50 in cash."

# Output:
# ("entity"<|>"Tiffany & Co."<|>"Organization"<|>"New York–based luxury goods company, the Target in the merger")##
# ("entity"<|>"LVMH Moët Hennessy–Louis Vuitton SE"<|>"Organization"<|>"French conglomerate, Parent Company and Acquirer in the merger")##
# ("entity"<|>"Merger"<|>"LegalDocument"<|>"The merger agreement between Tiffany and LVMH")##
# ("entity"<|>"Effective Time"<|>"Date"<|>"The closing time when the merger becomes effective")##
# ("entity"<|>"$131.50"<|>"MonetaryAmount"<|>"Cash amount per share of Tiffany stock upon merger")##
# ("relationship"<|>"Tiffany & Co."<|>"Merger"<|>"Tiffany is a party to the merger agreement"<|>"isPartyTo"<|>9)##
# ("relationship"<|>"LVMH Moët Hennessy–Louis Vuitton SE"<|>"Merger"<|>"LVMH is a party to the merger agreement"<|>"isPartyTo"<|>9)##
# ("relationship"<|>"LVMH Moët Hennessy–Louis Vuitton SE"<|>"Tiffany & Co."<|>"LVMH acquires Tiffany in the merger"<|>"acquires"<|>10)##
# ("relationship"<|>"Merger"<|>"Effective Time"<|>"Merger effective at the specified date"<|>"subjectTo"<|>8)##
# ("relationship"<|>"Tiffany & Co."<|>"$131.50"<|>"Tiffany stock exchanged for cash"<|>"conversionPrice"<|>7)##
# ("content_keywords"<|>"merger, acquisition, shareholder rights, monetary consideration")<|COMPLETE|>
# """,
#     """Example 2:

# Entity_types: [Organization, Role, Regulation, LegalDocument, Clause, Obligation]
# Text:
# "HubSpot acts as a controller of your Personal Data when you sign up... Customers are controllers of the data under GDPR."

# Output:
# ("entity"<|>"HubSpot"<|>"Organization"<|>"CRM and marketing platform company that processes personal data")##
# ("entity"<|>"Customers"<|>"Role"<|>"Users of HubSpot services, acting as data controllers")##
# ("entity"<|>"GDPR"<|>"Regulation"<|>"EU General Data Protection Regulation governing data privacy")##
# ("entity"<|>"Personal Data"<|>"Clause"<|>"Clause referring to data about identifiable individuals")##
# ("relationship"<|>"HubSpot"<|>"Personal Data"<|>"HubSpot controls personal data of users"<|>"controls"<|>8)##
# ("relationship"<|>"Customers"<|>"Personal Data"<|>"Customers act as controllers of personal data"<|>"controls"<|>7)##
# ("relationship"<|>"HubSpot"<|>"GDPR"<|>"HubSpot's data handling governed by GDPR"<|>"governedBy"<|>9)##
# ("content_keywords"<|>"privacy, data control, GDPR, personal data")<|COMPLETE|>
# """
# ]


# for hotpotqa or any multi-hop generic dataset, it can be uncommented to get better entity extraction.
PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "Person", "Organization", "Location", "Event", "Date",
    "WorkOfArt", "Award", "Concept", "Miscellaneous"
]

PROMPTS["entity_extraction"] = """-Goal-
Given a text passage from Wikipedia (as used in HotpotQA distractor dataset) and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalize proper nouns correctly.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Brief but clear description of the entity, based on the text.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation of how they are related
- relationship_strength: numeric score (1–10) of relationship strength
- relationship_keywords: key words summarizing the relationship
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main topics of the entire passage.
Format as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all entities and relationships. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [Person, Film, Date, WorkOfArt, Location, Organization]
Text:
"Ed Wood is a 1994 American biographical film directed by Tim Burton and starring Johnny Depp as cult filmmaker Ed Wood. The film also featured Martin Landau as Bela Lugosi."

Output:
("entity"<|>"Ed Wood"<|>"Film"<|>"1994 American biographical film directed by Tim Burton")##
("entity"<|>"Tim Burton"<|>"Person"<|>"American film director and producer of Ed Wood")##
("entity"<|>"Johnny Depp"<|>"Person"<|>"Actor portraying filmmaker Ed Wood")##
("entity"<|>"Ed Wood (person)"<|>"Person"<|>"Cult filmmaker, subject of the film Ed Wood")##
("entity"<|>"Martin Landau"<|>"Person"<|>"Actor portraying Bela Lugosi in Ed Wood")##
("entity"<|>"Bela Lugosi"<|>"Person"<|>"Hungarian-American actor represented in the film")##
("entity"<|>"1994"<|>"Date"<|>"Year the film Ed Wood was released")##
("relationship"<|>"Tim Burton"<|>"Ed Wood"<|>"Directed the film Ed Wood"<|>"directedBy"<|>9)##
("relationship"<|>"Johnny Depp"<|>"Ed Wood (person)"<|>"Portrays Ed Wood in the film"<|>"portrays"<|>10)##
("relationship"<|>"Martin Landau"<|>"Bela Lugosi"<|>"Portrays Bela Lugosi in the film"<|>"portrays"<|>9)##
("relationship"<|>"Ed Wood"<|>"1994"<|>"Film released in 1994"<|>"releaseDate"<|>8)##
("content_keywords"<|>"film, biography, Hollywood, actors, Tim Burton, Ed Wood")<|COMPLETE|>
""",
    """Example 2:

Entity_types: [Person, Organization, WorkOfArt, Location, Award, Date]
Text:
"Doctor Strange is a 2016 American superhero film produced by Marvel Studios and directed by Scott Derrickson. It stars Benedict Cumberbatch as Stephen Strange."

Output:
("entity"<|>"Doctor Strange"<|>"Film"<|>"2016 American superhero film in the Marvel Cinematic Universe")##
("entity"<|>"Marvel Studios"<|>"Organization"<|>"American production company producing Doctor Strange")##
("entity"<|>"Scott Derrickson"<|>"Person"<|>"Film director of Doctor Strange")##
("entity"<|>"Benedict Cumberbatch"<|>"Person"<|>"Actor portraying Stephen Strange")##
("entity"<|>"Stephen Strange"<|>"Person"<|>"Fictional Marvel Comics character Doctor Strange")##
("entity"<|>"2016"<|>"Date"<|>"Release year of Doctor Strange")##
("relationship"<|>"Scott Derrickson"<|>"Doctor Strange"<|>"Directed the film Doctor Strange"<|>"directedBy"<|>9)##
("relationship"<|>"Marvel Studios"<|>"Doctor Strange"<|>"Produced the film Doctor Strange"<|>"producedBy"<|>9)##
("relationship"<|>"Benedict Cumberbatch"<|>"Stephen Strange"<|>"Portrays Stephen Strange"<|>"portrays"<|>10)##
("relationship"<|>"Doctor Strange"<|>"2016"<|>"Film released in 2016"<|>"releaseDate"<|>8)##
("content_keywords"<|>"superhero, Marvel, film, magic, MCU, actors")<|COMPLETE|>
"""
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""
