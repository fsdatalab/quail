"""QUAIL-B prompt templates.

Filter templates take the document as `{0}`. Join templates take
one DOCUMENT marker per joined table as `{0}` and `{1}`.
"""

F1 = ("Judge strictly from the review above whether it mentions at "
      "least one positive aspect of the movie.\n\n{0}\n\nInstruction: "
      "answer TRUE if the review mentions at least one positive aspect "
      "of the movie, FALSE otherwise.")

F4 = ("Judge strictly from the review above whether it discusses the "
      "ending of the movie.\n\n{0}\n\nInstruction: answer TRUE if the "
      "review discusses the ending of the movie, FALSE otherwise.")

F5 = ("Judge strictly from the review above whether it mentions any "
      "specific actor or actress by name.\n\n{0}\n\nInstruction: "
      "answer TRUE if the review mentions a specific actor or actress "
      "by name, FALSE otherwise.")

DISCUSS_ASPECT = ("Does the review in DOCUMENT {0} discuss the movie "
                  "aspect in DOCUMENT {1}?")


# IMDB-8 only: a second question over the same aspects table, joined
# under a second alias (a2) - a 2-join star, both joins anchored on
# reviews so the second stage runs over whatever DISCUSS_ASPECT
# already kept.
ASPECT_SENTIMENT = ("Does the review in DOCUMENT {0} express positive "
                    "sentiment about the movie aspect in DOCUMENT {1}?")

SERIOUS_ADVERSE_EVENT = (
    "Judge strictly from the report above whether it describes a serious "
    "or life-threatening adverse event.\n\n{0}\n\nInstruction: answer TRUE "
    "if the report describes a serious or life-threatening adverse event, "
    "FALSE otherwise."
)

REACTION = ("Does the medical report in DOCUMENT {0} describe the "
            "reaction in DOCUMENT {1} as something the patient "
            "experienced?")

NEUROLOGICAL_REACTION = (
    "Is this reaction neurological, affecting the nervous system? {0}"
)

CARDIOVASCULAR_REACTION = (
    "Is this reaction cardiovascular, affecting the heart or blood vessels? {0}"
)

AGENT_RECOVERED = (
    "Judge strictly from the agent trace above whether the agent recovered "
    "after pursuing an approach that did not work. Recovery means the agent "
    "recognized or moved past the unsuccessful approach and then made useful "
    "progress with a different or corrected approach.\n\n{0}\n\n"
    "Instruction: answer TRUE if the trace shows the agent recovering after "
    "an unsuccessful approach, FALSE otherwise."
)

AGENT_IMPLEMENTED_FIX = (
    "Judge strictly from the agent trace above whether, by the end of the "
    "trace, the agent has implemented a plausible fix that directly addresses "
    "the reported issue. A fix must include a code or configuration change "
    "whose purpose is to correct the issue. Inspection, reproduction, tests "
    "without a fix, and unrelated edits do not count.\n\n{0}\n\nInstruction: "
    "answer TRUE if the agent has implemented a plausible fix that directly "
    "addresses the reported issue. Answer FALSE otherwise."
)

F11 = ("Judge strictly from the claim above whether it asserts "
       "something about a person, rather than an organization, place, "
       "or event.\n\n{0}\n\nInstruction: answer TRUE if the claim "
       "asserts something about a person, FALSE otherwise.")

F12 = ("Judge strictly from the claim above whether it contains a "
       "specific date or year.\n\n{0}\n\nInstruction: answer TRUE if "
       "the claim contains a specific date or year, FALSE otherwise.")

F14 = ("Judge strictly from the claim above whether it references a "
       "specific place (a city, country, or other named location).\n\n"
       "{0}\n\nInstruction: answer TRUE if the claim references a "
       "specific place, FALSE otherwise.")

SUPPORT = ("Does the Wikipedia passage in DOCUMENT {1} support the "
           "claim in DOCUMENT {0}?")

REFUTE = ("Does the Wikipedia passage in DOCUMENT {1} refute or "
          "contradict the claim in DOCUMENT {0}?")

F13 = ("Judge strictly from the Wikipedia passage above whether it "
       "primarily describes a specific person (their life, actions, "
       "or role), rather than an organization, place, or event.\n\n"
       "{0}\n\nInstruction: answer TRUE if the passage primarily "
       "describes a specific person, FALSE otherwise.")

# LePaRD predicates: "excerpt" for destination_context throughout,
# to avoid colliding with this dataset's own use of "passage" for
# the quoted/cited text.
LEP1 = ("Judge strictly from the excerpt above whether it argues that "
        "the cited case's reasoning does not apply here.\n\n{0}\n\n"
        "Instruction: answer TRUE if the excerpt argues the cited "
        "case's reasoning does not apply here, FALSE otherwise.")

LEP2 = ("Judge strictly from the excerpt above whether it discusses a "
        "procedural or jurisdictional issue.\n\n{0}\n\nInstruction: "
        "answer TRUE if the excerpt discusses a procedural or "
        "jurisdictional issue, FALSE otherwise.")

LEP3 = ("Judge strictly from the excerpt above whether it treats the "
        "cited passage as binding precedent.\n\n{0}\n\nInstruction: "
        "answer TRUE if the excerpt treats the cited passage as "
        "binding precedent, FALSE otherwise.")

LEP4 = ("Judge strictly from the excerpt above whether it cites the "
        "passage to support a conclusion about a party's liability or "
        "guilt.\n\n{0}\n\nInstruction: answer TRUE if the excerpt "
        "cites the passage to support a conclusion about a party's "
        "liability or guilt, FALSE otherwise.")

LEP5 = ("Judge strictly from the excerpt above whether it acknowledges "
        "disagreement between courts on the issue.\n\n{0}\n\n"
        "Instruction: answer TRUE if the excerpt acknowledges "
        "disagreement between courts on the issue, FALSE otherwise.")

# LEP-5 only: filters the passage side of the self-join, not just the
# excerpt (anchor) side.
LEPS1 = ("Judge strictly from the passage above whether it states a "
         "general legal rule.\n\n{0}\n\nInstruction: answer TRUE if "
         "the passage states a general legal rule, FALSE otherwise.")

# The LEP-2..LEP-5 join predicate. Ground truth comes from the
# dataset's own passage_id, not a judge pass.
LEPJOIN = ("Is the passage in DOCUMENT {1} cited by the legal excerpt "
           "in DOCUMENT {0}?")


# PrivacyPolicies predicates: user-language questions about user
# outcomes, not legal language about company practices. The vocabulary
# mismatch means keyword search, regex, and embeddings cannot solve
# these.
P_MSG = ("Judge strictly from the policy above whether, if a user sent "
         "a private message through this service, an employee of the "
         "company could read it.\n\n{0}\n\nInstruction: answer TRUE if "
         "an employee could read the user's private messages, FALSE "
         "otherwise.")

P_LOC = ("Judge strictly from the policy above whether this service "
         "would keep track of the user's physical location, even when "
         "the user is not actively using the service.\n\n{0}\n\n"
         "Instruction: answer TRUE if the service would track the "
         "user's location while they are not using it, FALSE otherwise.")

SCENARIO_MATCH = ("Based on the privacy policy in DOCUMENT {0}, could "
                  "the situation described in DOCUMENT {1} happen to a "
                  "user of this service?")


# CUAD predicates: the document is a rendered contract page, or every
# page of one contract, so the question names what the reader sees.
# Each question follows the definition of one CUAD clause category,
# whose lawyer annotation is the reference label.
CUAD_PAGE_CAPS_LIABILITY = (
    "Judge strictly from the contract page above whether it caps a "
    "party's liability for breaching its obligations, as a maximum "
    "recoverable amount or a time limit for bringing claims.\n\n{0}\n\n"
    "Instruction: answer TRUE if the page caps a party's liability, "
    "FALSE otherwise."
)

CUAD_PAGE_UNCAPPED_LIABILITY = (
    "Judge strictly from the contract page above whether it leaves a "
    "party's liability uncapped for some breach, including a carve-out "
    "that exempts a kind of breach such as IP infringement or a "
    "confidentiality breach from a cap.\n\n{0}\n\nInstruction: answer "
    "TRUE if the page leaves a party's liability uncapped for some "
    "breach, FALSE otherwise."
)

CUAD_CHANGE_OF_CONTROL = (
    "Judge strictly from the contract above whether a party may "
    "terminate, or must be notified or give consent, if the other party "
    "undergoes a change of control, such as a merger, a stock sale, or "
    "a sale of all or substantially all of its assets or business.\n\n"
    "{0}\n\nInstruction: answer TRUE if the contract gives a party "
    "rights upon the other party's change of control, FALSE otherwise."
)

CUAD_EXCLUSIVITY = (
    "Judge strictly from the contract above whether it contains an "
    "exclusive dealing commitment, such as buying all requirements from "
    "one party, or a prohibition on selling, licensing, or working with "
    "third parties, during or after the term.\n\n{0}\n\nInstruction: "
    "answer TRUE if the contract contains an exclusive dealing "
    "commitment, FALSE otherwise."
)

CUAD_NON_COMPETE = (
    "Judge strictly from the contract above whether it restricts a "
    "party's ability to compete with the other party, or to operate in "
    "a certain geography, business, or technology sector.\n\n{0}\n\n"
    "Instruction: answer TRUE if the contract restricts a party from "
    "competing, FALSE otherwise."
)

CUAD_LICENSE_GRANT = (
    "Judge strictly from the contract above whether one party grants "
    "the other a license, such as to intellectual property, software, "
    "or a trademark.\n\n{0}\n\nInstruction: answer TRUE if the contract "
    "grants a license, FALSE otherwise."
)

CUAD_NON_TRANSFERABLE_LICENSE = (
    "Judge strictly from the contract above whether it limits the "
    "licensee's ability to transfer or sublicense the license granted to "
    "a third party.\n\n{0}\n\nInstruction: answer TRUE if the license "
    "granted is non-transferable, FALSE otherwise."
)

CUAD_PERPETUAL_LICENSE = (
    "Judge strictly from the contract above whether it grants a license "
    "that is irrevocable or perpetual.\n\n{0}\n\nInstruction: answer TRUE "
    "if the contract grants an irrevocable or perpetual license, FALSE "
    "otherwise."
)

# FinanceBench predicates. The question filter reads the question's
# text; the page join reads a rendered filing page against a question,
# and the evidence pages the dataset marks label it.
FIN_NEEDS_CALCULATION = (
    "Judge strictly from the analyst question above whether answering it "
    "requires computing a value from two or more reported figures, such "
    "as a ratio, a margin, a growth rate, or a change between periods, "
    "rather than reading one reported figure or fact.\n\n{0}\n\n"
    "Instruction: answer TRUE if the question requires a calculation over "
    "reported figures, FALSE otherwise."
)

FIN_PAGE_EVIDENCE = (
    "Does the filing page in DOCUMENT {1} show the figures or statements "
    "an analyst needs to answer the question in DOCUMENT {0}?"
)

TREAS_COMBINES_FIGURES = (
    "Judge strictly from the question above, asked of a U.S. Treasury "
    "statement of receipts and expenditures, whether answering it requires "
    "combining two or more reported figures, such as a total over several "
    "months or years, a difference, a ratio, or a share, rather than "
    "reading one reported figure.\n\n{0}\n\n"
    "Instruction: answer TRUE if the question requires combining reported "
    "figures, FALSE otherwise."
)

TREAS_PAGE_EVIDENCE = (
    "Does the Treasury statement page in DOCUMENT {1} report the figures "
    "needed to answer the question in DOCUMENT {0}?"
)
