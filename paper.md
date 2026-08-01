\documentclass[11pt]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathpazo}
\usepackage{microtype}
\usepackage[margin=0.92in,headheight=22pt]{geometry}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{booktabs,longtable,tabularx,array,multirow}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}
\usepackage{xurl}
\usepackage[colorlinks=true,linkcolor=paperblue,citecolor=paperblue,urlcolor=paperblue]{hyperref}
\usepackage[nameinlink,noabbrev]{cleveref}

\definecolor{ink}{HTML}{172033}
\definecolor{paperblue}{HTML}{245C8A}
\definecolor{muted}{HTML}{5E6775}
\definecolor{rulegray}{HTML}{D6DCE3}
\definecolor{lightblue}{HTML}{EDF4FA}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\color{muted}\textsc{Scheduling $n$-Stage AI Filters}}
\fancyhead[R]{\footnotesize\color{muted}Working paper}
\fancyfoot[C]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{rulegray}\leaders\hrule height \headrulewidth\hfill}}

\titleformat{\section}{\Large\bfseries\color{paperblue}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries\color{ink}}{\thesubsection}{0.7em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{ink}}{\thesubsubsection}{0.7em}{}
\setlength{\parindent}{1.15em}
\setlength{\parskip}{2pt plus 1pt minus 1pt}
\setlist[itemize]{leftmargin=1.5em,itemsep=2pt,topsep=3pt}
\setlist[enumerate]{leftmargin=1.7em,itemsep=2pt,topsep=3pt}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\setlength{\emergencystretch}{3em}

\newtheorem{definition}{Definition}[section]
\newtheorem{assumption}[definition]{Assumption}
\newtheorem{proposition}[definition]{Proposition}
\newtheorem{lemma}[definition]{Lemma}
\newtheorem{corollary}[definition]{Corollary}
\newtheorem{remark}[definition]{Remark}

\newcommand{\E}{\mathbb{E}}
\newcommand{\Prb}{\mathbb{P}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\I}{\mathbb{I}}
\newcommand{\calD}{\mathcal{D}}
\newcommand{\calA}{\mathcal{A}}
\newcommand{\calB}{\mathcal{B}}
\newcommand{\calK}{\mathcal{K}}
\newcommand{\calS}{\mathcal{S}}
\newcommand{\calX}{\mathcal{X}}
\newcommand{\OPT}{\operatorname{OPT}}
\newcommand{\Tok}{\operatorname{Tok}}
\newcommand{\Anc}{\operatorname{Anc}}
\newcommand{\argmin}{\operatorname*{arg\,min}}
\newcommand{\one}{\mathbf 1}
\newcommand{\bytes}{\operatorname{bytes}}

\title{\color{ink}\textbf{Scheduling $n$-Stage AI Filters\\Under KV-Cache Constraints}}
\author{Working paper}
\date{July 31, 2026}

\begin{document}
\maketitle

\begin{abstract}
An AI SQL query can apply an ordered conjunction of language-model filters to every row of a table.  Each surviving row advances to the next filter, so filter selectivity determines which model calls are necessary.  At the same time, token order determines which prefixes can be reused, finite GPU memory determines which key-value (KV) blocks can remain resident, and batching determines how efficiently the GPU is used.  We formulate the resulting $n$-stage scheduling problem for a fixed filter order chosen by an upstream query optimizer.  The model keeps the realized, heterogeneous document lengths; permits variable-length sequences in the same batch; permits prefill to be divided into valid chunks; and records exactly which prefix tokens each new token may attend to.  We give an FP8 batch-latency model and a concrete dynamic program that enumerates feasible next batches, KV-retention choices, and their continuations.  A two-stage instance serves as a worked warm-up for task-first prefix caching, document-first pipelining, and fused speculative evaluation.  We then generalize to arbitrary $n$, including partial speculation over contiguous blocks of future filters.  The paper separates three quantities: a hardware resource lower bound, the exact optimum of the analytical model when it can be computed, and latency measured from a concrete inference engine.  It also specifies the data, schedule records, and plots required for reproducible latency and break-even experiments on Qwen3-4B-FP8 and Qwen3-32B-FP8 using H100 and L40S GPUs.
\end{abstract}

\noindent\textbf{Keywords:} AI SQL, LLM inference, query optimization, scheduling, KV cache, chunked prefill, speculative execution, dynamic programming

\section{Introduction}

Consider a query of the form
\begin{center}
\small\ttfamily
SELECT * FROM documents WHERE\\[-2pt]
AI\_FILTER($F_1$, document) AND $\cdots$ AND AI\_FILTER($F_n$, document).
\end{center}
The logical query optimizer has already chosen the order $F_1,\ldots,F_n$.  Execution remains nontrivial.  A row rejected by $F_j$ requires no later filter, but that fact is known only after $F_j$ completes.  Processing all future filters immediately removes these information barriers but performs speculative work.  Waiting preserves short-circuit semantics but may separate operations that could have shared the document's KV state.  Retaining that state saves recomputation but consumes capacity needed by other rows.  Finally, model throughput is a property of a GPU batch, not of a single logical call.

The workload considered here is deliberately prefill-heavy.  A query makes thousands of calls against distinct documents, each task prompt is short, and each filter returns a binary decision from the logits at the final prompt position.  There is no material autoregressive decode phase.  The objective is therefore not time to first token or inter-token latency.  It is makespan: the time until the SQL query has produced every required row-level result.

Two properties prevent a token-count-only analysis.  First, documents have heterogeneous lengths.  The optimizer receives the full realized vector $(d_1,\ldots,d_N)$ and may mix arbitrary lengths in a ragged batch.  Replacing this vector by a mean changes attention work, memory feasibility, and packing decisions.  Second, a document may be processed in several chunks.  Chunking does not reduce its mathematical transformer work, but it expands the set of feasible packings and can eliminate underfilled batches.

This paper develops one model that can be used in two ways.  With nominal device ceilings and zero software overhead, it defines a speed-of-light scheduling experiment.  With calibrated rates and measured reserve memory, it becomes an empirical predictor.  In both cases, the same schedule manifest lists the exact document IDs, filter stages, token chunks, prefix blocks, and post-batch evictions.

The main contributions are:
\begin{itemize}
  \item A formal $n$-stage AI-filter workload model with a fixed, externally optimized filter order, heterogeneous document lengths, conditional stage selectivities, and distinct offline and online information structures.
  \item A batch abstraction that supports variable-length sequences, chunked prefill, explicit within-batch dependencies, and physical KV-block sharing.  It counts a shared prompt or document block once when the assumed kernel loads it once.
  \item A transparent FP8 latency model based on dense transformer work, attention work, model-weight traffic, KV traffic, and peak HBM feasibility.  Weight precision and KV precision remain separate parameters.
  \item One exact dynamic programming algorithm for finite instances, a two-stage worked warm-up, and an $n$-stage generalization that includes task-first execution, document-first pipelining, full speculation, and bounded lookahead speculation.
  \item An experiment specification that makes every latency claim auditable: each reported point is labeled as a lower bound, an exact model optimum, a feasible schedule, or a measured engine result.
\end{itemize}

The paper does not assume that vLLM or any other existing engine implements the schedules.  Tree-aware attention, post-outcome retention, and cross-stage batching are architectural capabilities of the modeled engine.  Profiling an implementation is a separate experiment.

\section{Query semantics and realized workloads}

\subsection{Documents, filters, and fixed order}

Let $[N]=\{1,\ldots,N\}$ index documents and $[n]=\{1,\ldots,n\}$ index filters.  The filter order is fixed before inference begins.

\begin{assumption}[External filter ordering]
The sequence $F_1,\ldots,F_n$ is an input to the inference scheduler.  An upstream query optimizer has already selected it using semantic validity, selectivity estimates, and any operator-level costs.  The inference scheduler does not reorder filters.
\end{assumption}

Document $i$ has raw text $D_i$ and realized model-token length
\begin{equation}
d_i = \left|\Tok(D_i)\right|.
\label{eq:token-length}
\end{equation}
Filter $j$ has a task prompt of $p_j$ tokens.  The primary experiment uses $p_j\approx 50$, but every equation retains the individual $p_j$.

Let $X_{ij}\in\{0,1\}$ be the result of applying $F_j$ to document $i$.  Define the survival indicator
\begin{equation}
Y_{i1}=1,
\qquad
Y_{ij}=\prod_{k=1}^{j-1}X_{ik}quad (j\ge 2).
\label{eq:survival}
\end{equation}
The call $(i,j)$ is logically required exactly when $Y_{ij}=1$.  The realized logical job set is therefore
\begin{equation}
\mathcal J(X)=\{(i,j)\in[N]\times[n]:Y_{ij}=1\}.
\label{eq:logical-jobs}
\end{equation}

The conditional selectivity of stage $j$ is
\begin{equation}
s_j=\Prb(X_{ij}=1\mid Y_{ij}=1).
\label{eq:selectivity}
\end{equation}
Only $s_1,\ldots,s_{n-1}$ affect downstream inference work.  The value of $s_n$ affects the final SQL result but cannot create another model call.

\begin{assumption}[Length-independent selectivity]
For the baseline analysis,
\begin{equation}
\Prb(X_{ij}=1\mid Y_{ij}=1,d_i)=s_j.
\label{eq:length-independent}
\end{equation}
Filter outcomes may still have different selectivities at different stages.  The assumption says only that length is not predictive of survival.
\end{assumption}

The probability that a row reaches stage $j$ is
\begin{equation}
\pi_j=\Prb(Y_{ij}=1)=\prod_{k=1}^{j-1}s_k,
\qquad \pi_1=1,
\label{eq:reach-probability}
\end{equation}
under the corresponding conditional-selectivity model.  Thus the expected number of logically required filter evaluations per document is
\begin{equation}
\E[H_i]=\sum_{j=1}^{n}\pi_j.
\label{eq:expected-stages}
\end{equation}

\subsection{The document-length distribution is outside the scheduler}

Let $\mathcal D_{\mathrm{text}}$ denote the document population.  A workload draw produces raw documents and then tokenizes them:
\begin{equation}
(D_1,\ldots,D_N)\sim\mathcal D_{\mathrm{text}}^{(N)},
\qquad
\mathbf d=(d_1,\ldots,d_N).
\label{eq:length-draw}
\end{equation}
Here $\mathcal D_{\mathrm{text}}^{(N)}$ may mean independent sampling or uniform sampling without replacement from a finite dataset.  The experiment must state which.

The scheduler does not optimize against a histogram or an average.  Before scheduling begins, it knows the realized $\mathbf d$.  For a fixed outcome matrix $X$, its cost is $\OPT(\mathbf d,X)$.  For a fixed realized query but unknown filter outcomes, the relevant expectation is
\begin{equation}
L_{\mathrm{off}}(\mathbf d)
=
\E_{X\mid\mathbf d}\!\left[\OPT_{\mathrm{off}}(\mathbf d,X)\right].
\label{eq:conditional-estimand}
\end{equation}
If the scientific target is the population of possible $N$-document queries, an outer expectation is required:
\begin{equation}
\overline L_{\mathrm{off}}
=
\E_{\mathbf d}\!\left[L_{\mathrm{off}}(\mathbf d)\right].
\label{eq:population-estimand}
\end{equation}
The online quantities are defined analogously.  Equations~\eqref{eq:conditional-estimand} and \eqref{eq:population-estimand} are different estimands and require different Monte Carlo designs.

\subsection{Why the full vector matters}

Suppose a batch contains documents of lengths $100$, $600$, and $2{,}000$.  Ignoring task prompts, its new-token count is $2{,}700$, while its document self-attention pairs are
\begin{equation}
\frac{100\cdot101}{2}
+\frac{600\cdot601}{2}
+\frac{2{,}000\cdot2{,}001}{2}.
\label{eq:heterogeneous-example}
\end{equation}
This is neither three padded copies of the longest document nor one $2{,}700$-token sequence.  Every solver coefficient is computed from the actual item lengths in this way.

\subsection{Information structures}

\begin{definition}[Clairvoyant offline scheduler]
For a realized $(\mathbf d,X)$, an offline scheduler knows all outcomes before selecting the first batch.  Its optimum is $\OPT_{\mathrm{off}}(\mathbf d,X)$.  This oracle supplies a value-of-information lower bound; it is not implementable.
\end{definition}

\begin{definition}[Online scheduler]
An online scheduler knows $\mathbf d$, the selectivity model, and all outcomes revealed by completed batches.  It cannot use a filter result before the batch producing that result has finished.  Its optimal expected makespan is $\OPT_{\mathrm{on}}(\mathbf d,\mathbf s)$.
\end{definition}

All documents are available at time zero.  There is no arrival process in the primary model.

\section{Execution model}

\subsection{Token templates and reusable prefixes}

The same logical filter can be serialized in two different causal orders.

\begin{table}[ht]
\centering
\caption{Token templates and their reusable KV blocks.}
\label{tab:templates}
\begin{tabularx}{\textwidth}{@{}p{0.20\textwidth}p{0.28\textwidth}X@{}}
\toprule
Policy family & Causal template & Reuse opportunity \\
\midrule
Task first & $[F_j\text{ prompt}][D_i]$ & The $p_j$-token task prefix is shared by every document evaluated by $F_j$.  Document KV is specific to $(i,j)$. \\
Document first & $[D_i][F_j\text{ prompt}]$ & The $d_i$-token document prefix can be reused by several filter branches for the same document. \\
Fused & $[D_i]\rightarrow\{[F_j]:j\in G\}$ & All filter prompts in $G$ share one physical document prefix. \\
\bottomrule
\end{tabularx}
\end{table}

Changing token order can change model quality.  The paper treats both templates as semantically valid prompts and requires the eventual evaluation to verify that assumption.  Scheduling results alone cannot establish semantic equivalence.

\begin{figure}[ht]
\centering
\begin{tikzpicture}[
  node distance=7mm and 13mm,
  box/.style={draw=paperblue,rounded corners=2pt,fill=lightblue,minimum height=7mm,inner xsep=6pt,font=\small},
  gate/.style={circle,draw=muted,fill=white,inner sep=1.5pt,font=\scriptsize},
  arr/.style={-{Latex[length=2mm]},draw=muted}
]
\node[box] (doc) {document $D_i$};
\node[box,right=of doc] (f1) {$F_1$};
\node[gate,right=of f1] (g1) {$X_{i1}$};
\node[box,right=of g1] (f2) {$F_2$};
\node[gate,right=of f2] (g2) {$X_{i2}$};
\node[box,right=of g2] (fn) {$F_n$};
\draw[arr] (doc)--(f1);
\draw[arr] (f1)--(g1);
\draw[arr] (g1)--node[above,font=\scriptsize]{pass}(f2);
\draw[arr] (f2)--(g2);
\draw[arr,dashed] (g2)--node[above,font=\scriptsize]{$\cdots$}(fn);
\node[below=8mm of g1,font=\small,align=center,text=muted] {A failure terminates the row;\\speculation crosses one or more gates.};
\end{tikzpicture}
\caption{The fixed $n$-stage filter order.  A non-speculative schedule crosses an outcome gate only after the preceding batch completes.}
\label{fig:stage-chain}
\end{figure}

\subsection{Ragged batches}

A batch may contain sequences of different lengths.  Tokens are packed, together with sequence boundaries and a causal mask; no padding-to-maximum term appears in the ideal model.  Sequences from different documents never attend to one another.

For cost accounting, the batch records which earlier tokens each new token may attend to.  For batch $B$, let $Q(B)$ be its newly evaluated tokens.  For every $v\in Q(B)$, let $\Anc(v)$ contain the cached or same-batch tokens visible to $v$, including $v$ itself.  This notation covers ordinary independent requests as well as several filter prompts that share one document prefix.  It is only a description of the work inside a batch; it is not a scheduling algorithm.  The two basic batch statistics are
\begin{align}
U(B) &= |Q(B)|, \\
A(B) &= \sum_{v\in Q(B)}|\Anc(v)|.
\label{eq:batch-stats}
\end{align}
For an ordinary linear segment with $c$ cached tokens and $q$ new tokens,
\begin{equation}
a(c,q)=cq+\frac{q(q+1)}{2}.
\label{eq:linear-attention}
\end{equation}

\subsection{Chunked prefill}

Document $i$ may be divided into integral chunks $q_{i1},\ldots,q_{im_i}$ with
\begin{equation}
q_{i\ell}\ge 1,
\qquad
\sum_{\ell=1}^{m_i}q_{i\ell}=d_i.
\label{eq:chunk-partition}
\end{equation}
If $c_{i\ell}=\sum_{h<\ell}q_{ih}$, then
\begin{equation}
\sum_{\ell=1}^{m_i}a(c_{i\ell},q_{i\ell})
=
\frac{d_i(d_i+1)}{2}.
\label{eq:chunk-attention-invariance}
\end{equation}
Thus chunking preserves ideal dense-token and causal-attention work.  It can still improve makespan by filling otherwise unused batch capacity.  It can hurt measured performance through additional kernel launches, scheduler work, or repeated KV reads.  These overheads are parameters of a calibrated cost model, not negative token counts.

Let $r_i$ denote the currently resident prefix length of a partially processed document.  A batch may select an increment
\begin{equation}
0\le \Delta_i\le d_i-r_i,
\qquad
r_i'=r_i+\Delta_i.
\label{eq:chunk-progress}
\end{equation}
Choosing $\Delta_i=d_i-r_i$ recovers atomic prefill.

\subsection{Outcome gates and intra-batch dependencies}

Filter outcomes become visible only at batch boundaries.  A non-speculative batch cannot include $F_{j+1}(i)$ when $F_j(i)$ is unresolved at the start of that batch.  A speculative batch may include both operations because it deliberately performs the second without waiting for the first result.  In such a batch, both filter prompts may use document tokens produced earlier in that same batch.

\subsection{The modeled policy families}

\begin{definition}[Task-first execution]
For every required $(i,j)$, the engine evaluates document $D_i$ after the shared prefix for task $F_j$.  Calls remain separated by outcome gates.  The task-prefix KV blocks may be pinned for the query.
\end{definition}

\begin{definition}[Document-first pipeline]
The engine evaluates a document prefix and then executes required task prompts as separate branches from that prefix.  A document KV block may remain resident across outcome gates; if evicted, it must be recomputed before another branch can use it.
\end{definition}

\begin{definition}[Contiguous speculative block]
When the next unresolved stage is $j$, a speculative action with lookahead $k$ evaluates the branches $F_j,\ldots,F_{j+k-1}$ from the same document prefix without waiting for intermediate outcomes.  The row advances beyond the block only if all $k$ results pass; work after the first failure inside the block is wasted.
\end{definition}

Pipeline execution has $k=1$.  Full speculation at the first stage has $k=n$.  Allowing $1\le k\le n-j+1$ makes partial speculation a scheduling decision.

\section{Hardware and analytical batch cost}

\subsection{Parameters}

\begin{table}[ht]
\centering
\caption{Model, device, and runtime parameters.}
\label{tab:parameters}
\begin{tabularx}{\textwidth}{@{}p{0.12\textwidth}Xp{0.18\textwidth}@{}}
\toprule
Symbol & Meaning & Units \\
\midrule
$P$ & Dense non-embedding parameters repeatedly used at each token position & parameters \\
$W_{\mathrm{mem}}$ & Total resident model-weight footprint used in the HBM constraint & bytes \\
$W_{\mathrm{run}}$ & Compulsory transformer-weight traffic assigned to one nonempty batch & bytes/batch \\
$L$ & Transformer layers & layers \\
$h$ & Hidden width & elements \\
$n_{KV}$ & KV heads per layer & heads \\
$d_h$ & Attention head dimension & elements \\
$q_{KV}$ & Bytes per stored KV element & bytes/element \\
$\kappa$ & KV bytes per cached token & bytes/token \\
$M$ & Physical accelerator memory & bytes \\
$S$ & Non-weight, non-KV reserve: runtime, allocator slack, and mandatory scratch & bytes \\
$R_D$ & Dense FP8 tensor throughput ceiling used by the model & FLOP/s \\
$R_A$ & Attention-compute throughput ceiling used by the model & FLOP/s \\
$BW$ & Device-memory bandwidth ceiling & bytes/s \\
$L_{ctx}$ & Maximum tokens on any root-to-leaf causal path & tokens \\
\bottomrule
\end{tabularx}
\end{table}

Weight precision and KV precision are independent.  ``FP8 model'' refers to the weight checkpoint.  The primary experiment must separately choose $q_{KV}=1$ for FP8 KV or $q_{KV}=2$ for FP16/BF16 KV.

\subsection{KV footprint}

Grouped-query attention stores one key and one value for every KV head and layer.  Therefore
\begin{equation}
\kappa=2L n_{KV}d_hq_{KV}.
\label{eq:kv-token-bytes}
\end{equation}
The factor two denotes K and V.  It is unrelated to the multiply-add convention used for FLOPs.

\begin{table}[ht]
\centering
\caption{Qwen3 reference architecture inputs.  The configured maximum position value is recorded separately from any recommended native-context regime.}
\label{tab:qwen}
\small
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Model & $P$ & $L$ & $h$ & Q/KV heads & $d_h$ & $\kappa$ at $q_{KV}=1$ & $L_{ctx}$ \\
\midrule
Qwen3-4B-FP8 & 3.6B & 36 & 2,560 & 32/8 & 128 & 72 KiB/token & 40,960 \\
Qwen3-32B-FP8 & 31.2B & 64 & 5,120 & 64/8 & 128 & 128 KiB/token & 40,960 \\
\bottomrule
\end{tabular}
\end{table}

The parameter $P$ is used in the repeated dense-block approximation.  The experiment separately records $W_{\mathrm{mem}}$ after loading and $W_{\mathrm{run}}$ for the repeated transformer blocks.  This avoids charging the full embedding or output-head allocation as compulsory traffic in every batch while still reserving its HBM capacity.  Neither value is inferred by substituting $P$ bytes.

\begin{table}[ht]
\centering
\caption{Nominal device ceilings used as reference inputs.  NVIDIA marks the larger FP8 figures as sparsity-qualified; the dense column uses one half of those values.}
\label{tab:devices}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Device & Memory & Bandwidth & Advertised sparse FP8 & Dense FP8 input $R_D$ \\
\midrule
H100 SXM & 80 GB HBM3 & 3.35 TB/s & 3.958 PFLOP/s & 1.979 PFLOP/s \\
L40S & 48 GB GDDR6 & 864 GB/s & 1.466 PFLOP/s & 0.733 PFLOP/s \\
\bottomrule
\end{tabular}
\end{table}

These are ceilings, not achieved rates.  Final calculations use the exact GPU variant and report both decimal GB/TB and binary GiB/TiB conversions where relevant.

\subsection{Dense transformer work and the factor \texorpdfstring{$2P$}{2P}}

For a linear map $y=Wx$, each weight participates in one multiplication and one addition for each token position.  Counting a fused multiply-add as two FLOPs gives approximately $2P$ FLOPs per new token for the repeated dense projections in a dense transformer.  Thus
\begin{equation}
F_D(B)=2P\,U(B).
\label{eq:dense-flops}
\end{equation}
This approximation excludes vocabulary projection, embeddings, normalization, and elementwise terms unless the experiment adds them explicitly.  It also excludes the quadratic query-key and attention-value products, which are counted next.

\subsection{Attention work}

Each allowed query-key pair requires a dot product and an attention-value accumulation.  With hidden width $h$ across $L$ layers, the first-order attention FLOPs are
\begin{equation}
F_A(B)=4Lh\,A(B).
\label{eq:attention-flops}
\end{equation}
The effective attention ceiling $R_A$ need not equal the dense tensor-core ceiling $R_D$.  Setting them equal is an optimistic analytical choice that must be labeled; a calibrated model estimates $R_A$ from attention microbenchmarks.

\subsection{Physical KV blocks and ideal traffic credit}

Let $\mathcal R(B)$ be the union of physical resident KV token blocks referenced by batch $B$, after deduplicating shared prefixes.  Let $K_R(B)=|\mathcal R(B)|$.  If two speculative branches for the same document use one fused operation, the document block appears once in this union.  If they are separate operations that reload the block, it appears in the traffic ledger twice, even if the logical token IDs match.

Let $K_W(B)$ be the number of new token positions whose KV must be written to device memory for use by a later chunk, branch, or batch.  It need not equal $U(B)$.  In particular, an ideal fused zero-decode operation may avoid persistent KV writes for final prompt positions that have no future consumer, while an early document chunk must materialize its KV before a later chunk can resume.  Temporary on-chip or scratch storage remains part of $K_{tmp}$ or the calibrated kernel model.  An ideal compulsory KV traffic term is
\begin{equation}
B_{KV}(B)=\kappa\bigl(K_R(B)+K_W(B)\bigr).
\label{eq:kv-traffic}
\end{equation}
This is a lower-bound traffic model.  A real kernel can transfer the same block more than once because of tiling, cache capacity, or unfused execution.

\subsection{Peak-memory feasibility}

Let $\mathcal K_t$ be the physical KV token blocks resident at the start of batch $t$, and let $K_{tmp}(B_t)$ be the maximum additional token-equivalent KV storage needed while the batch runs.  The peak-memory condition is
\begin{equation}
W_{\mathrm{mem}}+S+\kappa\left(|\mathcal K_t|+K_{tmp}(B_t)\right)
\le M.
\label{eq:hbm-capacity}
\end{equation}
Shared physical blocks are counted once.  Temporary prompt-branch KV is included even when discarded immediately after logits are read.  A separate scratch term may be added to $S$ or modeled as a batch-dependent quantity.

Every causal path must also satisfy
\begin{equation}
d_i+p_j\le L_{ctx}.
\label{eq:context-limit}
\end{equation}
For a speculative tree, total nodes may exceed $L_{ctx}$; the limit applies to each document-to-branch path.

\subsection{The analytical latency assigned to one batch}

The dense-block time is
\begin{equation}
D(B)=\max\left\{
\frac{2P\,U(B)}{R_D},
\frac{W_{\mathrm{run}}\,\one\{U(B)>0\}}{BW}
\right\}.
\label{eq:dense-time}
\end{equation}
The attention time is
\begin{equation}
H(B)=\max\left\{
\frac{4Lh\,A(B)}{R_A},
\frac{B_{KV}(B)}{BW}
\right\}.
\label{eq:attention-time}
\end{equation}
The primary analytical cost is
\begin{equation}
\tau_0(B)=D(B)+H(B).
\label{eq:ideal-batch-cost}
\end{equation}
This definition treats the dense and attention kernel groups as serialized and allows perfect overlap within each group between its compute and bandwidth demands.  It omits host scheduling, launch gaps, tensor construction, output handling, and shape-dependent efficiency.

A calibrated version can use
\begin{equation}
\tau_{\theta}(B)
=
\beta_0+\beta_DD(B)+\beta_HH(B)
+\beta_U U(B)+\beta_Q |Q(B)|_{\mathrm{segments}},
\label{eq:calibrated-cost}
\end{equation}
with coefficients fit separately for every model-device pair.  The analytical and calibrated results must be reported as different curves.

\section{Schedules, states, and exact optimization}

\subsection{Schedules}

\begin{definition}[Feasible batch]
A batch specifies the document and filter token chunks evaluated together, which physical KV blocks they read, which new blocks they produce, and which prefixes are shared.  It is feasible when: (i) every required earlier filter result is already known, unless the batch explicitly speculates; (ii) every input KV block is resident or produced earlier in the batch; (iii) the HBM and context limits in \cref{eq:hbm-capacity,eq:context-limit} hold; and (iv) any configured token or sequence-count limits hold.
\end{definition}

\begin{definition}[Schedule]
A realized schedule $\sigma=(B_1,E_1,\ldots,B_T,E_T)$ is an ordered sequence of feasible batches and boundary eviction decisions.  After $B_t$ completes and its outcomes are visible, $E_t$ identifies which resident blocks are discarded before the next batch.  Every logically or speculatively selected operation completes exactly once, except that a discarded incomplete prefix may later be recomputed.
\end{definition}

For one GPU, batches execute sequentially.  The modeled makespan is
\begin{equation}
T_{\theta}(\sigma)=T_{\mathrm{init}}+\sum_{t=1}^{T}\tau_{\theta}(B_t),
\label{eq:schedule-cost}
\end{equation}
where $T_{\mathrm{init}}$ includes any policy-specific prefix initialization that is not already represented by a batch in $\sigma$.  An implementation must choose one representation and never count the same initialization twice.

\subsection{State and action}

A complete state $S_t$ contains:
\begin{itemize}
  \item the next unresolved logical stage of every document;
  \item the resident progress of every incomplete causal prefix;
  \item the physical KV blocks in $\mathcal K_t$;
  \item all filter outcomes revealed before batch $t$;
  \item policy-specific branch or recomputation status.
\end{itemize}
An offline action chooses a feasible batch and the KV blocks to evict after it.  Because this oracle knows $X$, it may use future outcomes when making that choice.  An online scheduler first chooses a batch.  After that batch finishes and reveals its results, it chooses which KV blocks to evict.  The transition $T(S_t,a_t,\xi_t)$ updates completed work, observed results, resident KV, and any prefixes that must later be recomputed.

\subsection{Bellman equations}

For a fixed realized outcome matrix $X$, the offline value function is
\begin{equation}
V^{\mathrm{off}}_g(S;\mathbf d,X)
=
\min_{a\in\calA_g(S;\mathbf d,X)}
\left\{
\tau_{\theta}(B(a))
+V^{\mathrm{off}}_g(T_g(S,a;X);\mathbf d,X)
\right\},
\label{eq:offline-bellman}
\end{equation}
where $g$ identifies a policy class and $V_g(S_{\mathrm{terminal}})=0$.

For an online state whose history is $\mathcal H(S)$,
\begin{equation}
V^{\mathrm{on}}_g(S;\mathbf d,\mathbf s)
=
\min_{a\in\calA_g(S;\mathbf d)}
\left\{
\tau_{\theta}(B(a))
+\E\left[
V^{\mathrm{on}}_g(T_g(S,a;\xi);\mathbf d,\mathbf s)
\mid \mathcal H(S),a
\right]
\right\}.
\label{eq:online-bellman}
\end{equation}
The online action set cannot depend on outcomes absent from $\mathcal H(S)$.

\begin{proposition}[Bellman optimality]
For any finite token lengths, finite HBM capacity, and positive batch costs, \cref{eq:offline-bellman,eq:online-bellman} equal the minimum makespan within their stated policy and information classes.
\end{proposition}
\begin{proof}
Every feasible schedule has a first batch.  The algorithm considers every feasible choice for that batch and every permitted KV-retention decision.  After those choices, the resulting state contains all information that can affect the remaining cost.  Applying the same argument repeatedly considers every feasible complete schedule.  Taking the least costly continuation at every state therefore gives the global optimum.  In the online case, the first batch is selected before its results are known, while eviction and later batches may depend on results that have just been observed.
\end{proof}

\subsection{Allowing chunking cannot worsen the optimum}

\begin{proposition}[Chunking dominance]
Let $\Sigma_{\mathrm{atomic}}$ be the schedules in which every document prefill is indivisible, and let $\Sigma_{\mathrm{chunk}}$ permit arbitrary integral partitions.  Under the same cost and memory rules,
\begin{equation}
\min_{\sigma\in\Sigma_{\mathrm{chunk}}}T_{\theta}(\sigma)
\le
\min_{\sigma\in\Sigma_{\mathrm{atomic}}}T_{\theta}(\sigma).
\label{eq:chunk-dominance}
\end{equation}
\end{proposition}
\begin{proof}
Every atomic prefill is the one-chunk partition $q_{i1}=d_i$.  Hence $\Sigma_{\mathrm{atomic}}\subseteq\Sigma_{\mathrm{chunk}}$.
\end{proof}

Equality is possible and likely when a large backlog already yields full batches.  Strict improvement requires that splitting at least one document enables a lower-cost packing after accounting for any modeled chunk overhead.

\subsection{Three different uses of the word ``optimal''}

The paper reports three layers separately:
\begin{enumerate}
  \item A \emph{resource lower bound} that no schedule can beat under stated hardware ceilings.
  \item A \emph{model optimum} that minimizes \cref{eq:schedule-cost} under the exact encoded constraints.
  \item A \emph{measured latency} from an implementation, which includes software and kernel effects absent from $\tau_0$.
\end{enumerate}
An optimizer certificate proves only the second claim.  Equality between a feasible schedule and a valid resource lower bound proves both first and second for that instance.  Neither proves that an existing serving engine attains the schedule.

\section{Two-stage warm-up}

The two-stage case isolates the essential tradeoff before introducing $n$-stage lookahead.  Write $X_i=X_{i1}$ and $X=\{i:X_i=1\}$.  Every document requires $F_1$; only documents in $X$ require $F_2$.

\subsection{An ideal token-work ledger}

First ignore eviction, recomputation, batch boundaries, and traffic.  The resulting token totals are not latency estimates; they are a consistency check.

For task-first execution, each filter prompt is computed once and each eligible document is processed after it:
\begin{equation}
U_T
=p_1+\one\{|X|>0\}p_2
+\sum_{i=1}^{N}d_i
+\sum_{i\in X}d_i.
\label{eq:two-task-tokens}
\end{equation}
For a document-first pipeline with every document KV retained until its last required branch,
\begin{equation}
U_P
=\sum_{i=1}^{N}d_i+Np_1+|X|p_2.
\label{eq:two-pipeline-tokens}
\end{equation}
For full speculation,
\begin{equation}
U_S
=\sum_{i=1}^{N}d_i+N(p_1+p_2).
\label{eq:two-spec-tokens}
\end{equation}
Pipeline and speculation compute each document once.  Speculation replaces the random $|X|p_2$ term with $Np_2$, but it may remove an outcome barrier and load the document KV once for both prompt branches in a fused operation.

The corresponding attention ledger is
\begin{align}
A_T
&=a(0,p_1)+\one\{|X|>0\}a(0,p_2)
  +\sum_i a(p_1,d_i)+\sum_{i\in X}a(p_2,d_i),
\label{eq:two-task-attn}\\
A_P
&=\sum_i\left[a(0,d_i)+a(d_i,p_1)\right]
  +\sum_{i\in X}a(d_i,p_2),
\label{eq:two-pipe-attn}\\
A_S
&=\sum_i\left[a(0,d_i)+a(d_i,p_1)+a(d_i,p_2)\right].
\label{eq:two-spec-attn}
\end{align}
These expressions use every $d_i$ individually.

\subsection{Task-first state and available batches}

For each document, let $z_i\in\{1,2,3\}$ be its logical state: $1$ means $F_1$ is unfinished; $2$ means $F_1$ passed and $F_2$ is unfinished; $3$ means the row is finished.  Let $r_i\in\{0,\ldots,d_i\}$ be the resident document progress under the current task prefix.  If an incomplete request is evicted, $r_i$ resets to zero.  The task-prefix blocks for $F_1$ and $F_2$ are shared physical blocks and may be pinned.

A task-first action chooses increments $\Delta_i$ for documents at their current $z_i\in\{1,2\}$.  When $r_i+\Delta_i=d_i$, the corresponding filter completes at the end of the batch.  An $F_1$ pass changes $z_i$ from $1$ to $2$ and starts a new request with $r_i=0$ under the $F_2$ prefix; an $F_1$ failure changes $z_i$ to $3$.  Completion of $F_2$ also changes $z_i$ to $3$.

Let
\begin{equation}
S_T=(\mathbf z,\mathbf r,\mathcal K,\mathcal H).
\label{eq:task-state}
\end{equation}
The exact task-first recurrences are \cref{eq:offline-bellman,eq:online-bellman} with this state and the action rules above.  A batch may mix $F_1$ and already eligible $F_2$ chunks from different documents.  It may not place $F_2(i)$ in the same online batch that first reveals $X_i$.

\subsection{Pipeline state and available batches}

For the pipeline, retain the same logical state $z_i$.  Let $r_i\in\{0,\ldots,d_i\}$ now denote the resident document-prefix length, independent of the current filter.  A prompt branch may execute only when $r_i=d_i$.  If a full document prefix is evicted while $z_i\in\{1,2\}$, then $r_i$ becomes zero and the document must be recomputed before its next branch.

The pipeline state is
\begin{equation}
S_P=(\mathbf z,\mathbf r,\mathcal K,\mathcal H).
\label{eq:pipeline-state}
\end{equation}
A pipeline action may contain:
\begin{itemize}
  \item arbitrary document-prefix increments $\Delta_i$;
  \item an $F_1$ prompt branch for any $i$ with $z_i=1$ and a complete prefix;
  \item an $F_2$ prompt branch for any $i$ with $z_i=2$ and a complete prefix;
  \item post-batch evictions of any document blocks not required by an intra-batch descendant.
\end{itemize}
An online action cannot include $F_2(i)$ when $z_i=1$ at batch start.  After an $F_1$ batch, retention may depend on the revealed $X_i$ because that choice is made in the next observed state.

\subsection{Fused speculation state and available batches}

For full speculation, each unfinished document has a document-prefix progress $r_i$.  One batch may complete the remaining document tokens and then evaluate both filter prompts from that shared prefix.  This is feasible only if the document blocks are already resident or are produced earlier in the same batch.  After both logits are available, the document is complete and no persistent KV is required.

Let
\begin{equation}
S_S=(\mathbf z,\mathbf r,\mathcal K),
\qquad z_i\in\{1,3\}.
\label{eq:spec-state}
\end{equation}
The action and transition do not depend on $X$.  Consequently, for every realized outcome vector,
\begin{equation}
\OPT_{S}^{\mathrm{off}}(\mathbf d,X)
=
\OPT_{S}^{\mathrm{on}}(\mathbf d,\mathbf s)
=
\OPT_{S}(\mathbf d).
\label{eq:spec-outcome-independent}
\end{equation}

\subsection{Optimality and value of information}

\begin{proposition}[Two-stage exactness]
When instantiated with the states and action sets in this section, the Bellman equations return the minimum analytical makespan for task-first execution, document-first pipelining, and full fused speculation, respectively, including heterogeneous lengths, chunking, KV eviction, and recomputation.
\end{proposition}
\begin{proof}
Each state contains every variable that can affect future feasibility or cost.  From each state, the algorithm considers every batch and eviction choice permitted by the selected policy.  The claim follows from Bellman optimality.
\end{proof}

\begin{proposition}[Value of clairvoyance]
For task-first and pipeline execution,
\begin{equation}
\E_X\left[\OPT_g^{\mathrm{off}}(\mathbf d,X)\right]
\le
\OPT_g^{\mathrm{on}}(\mathbf d,\mathbf s),
\qquad g\in\{T,P\}.
\label{eq:value-information}
\end{equation}
\end{proposition}
\begin{proof}
Every online policy induces a feasible schedule for every realized $X$.  The offline optimizer minimizes over a larger set because it may condition its first action on $X$.  Apply the pointwise inequality and take expectations.
\end{proof}

The inequality may be strict because an offline scheduler can retain precisely the document KVs that will be needed by $F_2$.  An online scheduler can condition its immediate post-batch eviction decision on outcomes just revealed, but it cannot recover prefixes evicted earlier or anticipate outcomes of unfinished $F_1$ calls.

\subsection{A concrete chunking example}

Suppose a simplified batch accepts at most ten new tokens and has a fixed positive weight-read cost.  Documents have lengths $8,8,4$.  Atomic prefill requires batches $\{8\},\{8\},\{4\}$.  Chunking the four-token document into two pieces permits $\{8,2\},\{8,2\}$.  The transformer token and attention work are unchanged, but one underfilled batch and one weight-read event disappear.  This example explains why chunking can improve makespan without invoking decode latency.

\section{The general \texorpdfstring{$n$}{n}-stage problem}

\subsection{Logical frontier}

Let $z_i\in\{1,\ldots,n+1\}$ be the next unresolved stage, with $z_i=n+1$ denoting completion.  For non-speculative execution, a passing result increments $z_i$; a failure sets $z_i=n+1$.  The online scheduler observes only outcomes for stages below the current frontier.

\subsection{Task-first execution}

Task-first execution maintains one shared prompt block per filter and one current document sequence per active row.  At stage $z_i=j$, the resident progress $r_i$ is the number of document tokens computed after the $F_j$ task prefix.  When stage $j$ passes, the next request begins from the distinct $F_{j+1}$ prefix with $r_i=0$.  Therefore document work is repeated at every reached stage.

Ignoring batching and evictions, expected dense new-token work is
\begin{equation}
\E[U_T\mid\mathbf d]
=
\sum_{j=1}^{n}\Prb\!\left(\sum_iY_{ij}>0\right)p_j
+\sum_{i=1}^{N}d_i\sum_{j=1}^{n}\pi_j.
\label{eq:n-task-work}
\end{equation}
The prompt term is query-level prefix initialization; the document term is row-level work.  Under independent row outcomes, $\Prb(\sum_iY_{ij}>0)=1-(1-\pi_j)^N$.

\subsection{Document-first pipeline}

The pipeline maintains a document prefix independent of stage.  If it remains resident, all reached prompt branches extend that same prefix.  Ignoring eviction and recomputation,
\begin{equation}
\E[U_P\mid\mathbf d]
=
\sum_{i=1}^{N}d_i
+N\sum_{j=1}^{n}\pi_jp_j.
\label{eq:n-pipeline-work}
\end{equation}
Finite memory couples documents: retaining a long prefix saves future recomputation if the row survives, but occupies capacity while other rows are scheduled.

\subsection{Full and partial speculation}

Full speculation computes every branch for every document:
\begin{equation}
U_S
=
\sum_{i=1}^{N}d_i+N\sum_{j=1}^{n}p_j.
\label{eq:n-full-spec-work}
\end{equation}
It has no outcome-dependent schedule.  The expected excess prompt work relative to an ideal no-eviction pipeline is
\begin{equation}
N\sum_{j=1}^{n}(1-\pi_j)p_j.
\label{eq:spec-excess}
\end{equation}
This token excess can be offset only by batch-level effects: fewer outcome barriers, fuller batches, fewer model-weight reads, and deduplicated physical document-KV loads.

Partial speculation chooses a lookahead $k$ at frontier $j$.  Let
\begin{equation}
G(j,k)=\{j,j+1,\ldots,j+k-1\}.
\label{eq:spec-block}
\end{equation}
All prompts in $G(j,k)$ are computed as branches from the same document.  If the first failure inside the block is at stage $h$, prompts $h+1,\ldots,j+k-1$ were unnecessary but already computed.  If every result passes, the new frontier is $j+k$.

\subsection{General state and Bellman recurrence}

For task-first execution, use
\begin{equation}
S_T^{(n)}=(\mathbf z,\mathbf r,\mathcal K,\mathcal H),
\label{eq:n-task-state}
\end{equation}
where each $r_i$ is progress under the current task prefix.  For document-first execution with adaptive speculation, use
\begin{equation}
S_D^{(n)}=(\mathbf z,\mathbf r,\mathcal K,\mathcal H),
\label{eq:n-document-state}
\end{equation}
where $r_i$ is progress on the reusable document prefix and an action may select a lookahead $k_i$ for each branch-ready document.  The symbols coincide but their physical blocks differ; a solver manifest records the policy family explicitly.

Substituting these states and their policy-specific feasible batches into \cref{eq:offline-bellman,eq:online-bellman} gives the exact $n$-stage recurrences.  The two-stage models are the $n=2$ specialization.

\begin{proposition}[Full speculation ignores selectivity]
For full $n$-stage speculation, the feasible action set, transition sequence, and analytical cost are independent of $X$ and $\mathbf s$.
\end{proposition}
\begin{proof}
Every document receives all $n$ prompt branches, regardless of outcomes.  Outcome values affect only the returned SQL result, not future speculative work or feasibility.
\end{proof}

\subsection{Complexity}
\label{sec:complexity}

Even with atomic jobs and no outcome uncertainty, selecting feasible batches under capacity contains bin packing.  KV retention adds inventory state; chunking adds progress state; and partial speculation adds lookahead decisions.  The Bellman equations are exact but exponentially large.

\begin{proposition}[Strong NP-hardness]
Computing the offline optimum is strongly NP-hard, even for the full-speculation policy with zero attention cost and atomic documents.
\end{proposition}
\begin{proof}
Reduce from 3-PARTITION.  Given $3m$ item sizes $a_i$ satisfying $B/4<a_i<B/2$ and $\sum_i a_i=mB$, create one speculative document per item with token size $a_i$, batch capacity $B$, and a positive constant cost for every nonempty batch.  A schedule of cost $m$ exists exactly when the items can be partitioned into $m$ triples of total size $B$.  Strong NP-hardness follows.
\end{proof}

\section{The exact scheduling algorithm}

The Bellman equations define the optimum compactly.  This section states the corresponding algorithm in operational terms.  There is one state representation and one candidate-batch generator.  From the current state, enumerate every feasible next batch, calculate its cost, and record the resulting state.  For fixed outcomes, a least-cost search over those states returns the optimal schedule.  For unknown outcomes, the same transitions are combined using the expected-cost recurrence below.  Task-first, pipeline, and speculation differ only in which next batches they permit.

\subsection{Inputs and output}

Fix a policy family $g$, realized document lengths $\mathbf d$, prompt lengths, model and device parameters, and a batch cost $\tau_\theta$.  For the offline problem, also fix the realized filter outcomes $X$.  For the online problem, supply the stage selectivities $\mathbf s$ instead; outcomes are revealed only when their batches finish.

Tokens are discrete, so the exact action set is finite.  To reduce it further, an experiment may choose a chunk quantum $\delta\ge1$: document chunks may end only at multiples of $\delta$, except for a final shorter remainder.  Let $\OPT_\delta$ be the best schedule under this restriction.  Then
\begin{equation}
\OPT_1\le\OPT_\delta.
\label{eq:quantum-bound}
\end{equation}
Thus $\delta=1$ is token-exact.  A larger $\delta$ may make computation easier, but it produces only an upper bound on the unrestricted model optimum and must be reported.

The algorithm returns both a latency and a concrete schedule
\begin{equation}
\sigma=(B_1,E_1,\ldots,B_T,E_T),
\end{equation}
where $B_t$ lists the work in batch $t$ and $E_t$ lists the KV blocks evicted after its results are observed.

\subsection{Step 1: represent the current state}

For every document $i$, the state stores:
\begin{enumerate}
  \item $z_i$, the next filter whose result is still needed;
  \item $r_i$, the number of document tokens currently represented by usable KV;
  \item the identities of its resident KV blocks;
  \item any unfinished chunk and any filter results already observed.
\end{enumerate}
The state also stores shared task-prompt blocks and the total resident KV set.  An evicted incomplete prefix loses its progress.  An evicted complete document prefix may be used again only after the document has been recomputed.  Consequently, two states with different resident KV are not merged even when their documents have reached the same filters.

The initial state has every document waiting for $F_1$.  The terminal state has a final pass or failure for every document and no remaining required work.

\subsection{Step 2: enumerate every feasible next batch}

Given state $S$, the batch generator first lists the work currently available.  It then considers every combination of those items and every permitted chunk length.  A combination is retained only if its required KV exists, its dependencies are satisfied, and its peak memory and context length fit the hardware limits.  For every retained batch $B$, the algorithm calculates $U(B)$, $A(B)$, KV reads and writes, peak memory, and $c(B)=\tau_\theta(B)$.

The policy family changes only the available work:
\begin{table}[ht]
\centering
\small
\begin{tabularx}{\textwidth}{@{}p{0.18\textwidth}X@{}}
\toprule
Policy & Work that may appear in the next batch \\
\midrule
Task-first & Chunks of $[F_{z_i},D_i]$ for documents whose current stage is already known.  Passing a filter starts a new request under the next task prefix. \\
Pipeline & Document chunks, plus $F_{z_i}$ prompt tokens for documents whose complete document KV is resident.  Work from different stages and different documents may share a batch. \\
Speculation & The pipeline work above, plus a contiguous block $F_{z_i},\ldots,F_{z_i+k-1}$ evaluated from one shared document prefix without waiting for intermediate results. \\
\bottomrule
\end{tabularx}
\caption{The exact algorithm is unchanged across policies; only the candidate batches differ.}
\label{tab:batch-generation}
\end{table}

For a non-speculative policy, the batch that first computes $F_j(i)$ cannot also compute $F_{j+1}(i)$, because the pass or failure is unknown when the batch is formed.  This restriction applies only within one document.  The same batch may contain $F_j$ work for one document and already-eligible $F_{j+1}$ work for another.

\subsection{Step 3: find the least-cost complete schedule}

For the offline problem, treat every distinct state as a point in the search.  Every feasible pair consisting of a batch $B$ and a post-batch eviction set $E$ leads from $S$ to
\begin{equation}
S'=T_g(S,B,X,E)
\label{eq:offline-transition}
\end{equation}
at cost $c(B)$.  States and transitions are generated only when reached; they need not be listed in advance.  The exact search is:
\begin{enumerate}
  \item Give the initial state distance zero and every other state distance infinity.
  \item Select the reached, unprocessed state $S$ with the smallest current distance.
  \item Generate every feasible $(B,E)$ from $S$.  For each resulting $S'$, replace its distance if
  \begin{equation}
  \operatorname{dist}(S)+c(B)<\operatorname{dist}(S').
  \label{eq:distance-update}
  \end{equation}
  When replacing it, store $S$ and $(B,E)$ as the predecessor choice.
  \item Mark $S$ processed and repeat until the terminal state is selected.
\end{enumerate}
This is Dijkstra's algorithm with scheduler states generated on demand.  It is valid because every nonempty batch has positive cost.  Following the stored predecessor choices backward from the terminal state reconstructs the optimal schedule.  Unlike naive recursive memoization, this search remains correct when eviction and recomputation create loops between cache states; every positive-cost loop is automatically dominated.

The online recurrence differs in one place.  The next batch must be selected before its new results $x$ are known, but the scheduler may choose what to evict after observing $x$.  Therefore
\begin{equation}
V_g^{\mathrm{on}}(S)
=
\min_{B\in\mathcal B_g(S)}
\left[
c(B)
+\sum_{x\in\mathcal X(S,B)}
\Prb(x\mid S,B)
\min_{E\in\mathcal E(S,B,x)}
V_g^{\mathrm{on}}\!\left(T_g(S,B,x,E)\right)
\right].
\label{eq:explicit-online-dp}
\end{equation}
Here $\mathcal X(S,B)$ is the set of result vectors that batch $B$ can reveal, and $\mathcal E(S,B,x)$ is the set of allowed post-result eviction choices.  Equation~\eqref{eq:explicit-online-dp} makes the information order explicit: choose $B$, observe $x$, choose $E$, and continue.  For the small instances considered here, the online table is obtained by enumerating the same states and evaluating this recurrence.  The paper does not introduce a second large-scale formulation for it.

\subsection{Why the algorithm is optimal}

\begin{proposition}[Exactness of the dynamic program]
For a finite instance and fixed policy family $g$, the state search and online recurrence above return the minimum makespan under the stated batch cost, memory rules, and information constraints.
\end{proposition}
\begin{proof}
Every feasible schedule has a first batch and a first post-batch eviction decision.  The algorithm enumerates every such permitted first choice.  Once that choice is made and its results are incorporated, the stored state contains everything that can affect future feasibility or cost.  Hence the remainder of an optimal schedule must itself be an optimal schedule from the resulting state.  Considering every transition therefore covers every feasible complete schedule.  In the offline problem, positive transition costs make Dijkstra's least-cost path the global optimum.  In the online problem, the outer minimization in \cref{eq:explicit-online-dp} chooses the batch before its results are known, while the inner minimization chooses eviction after those results are observed, exactly matching the scheduler's information.
\end{proof}

\subsection{Computational scope}

The exact algorithm is exponential.  There can be exponentially many resident-KV sets, progress vectors, and feasible batches, even before uncertain filter outcomes are included.  This is consistent with the NP-hardness result in \cref{sec:complexity}; recording each reached state avoids duplicated work but does not remove the worst-case combinatorial growth.

The paper therefore makes two different computational claims:
\begin{enumerate}
  \item On small instances, run the exact dynamic program and report the returned schedule and latency as the model optimum.
  \item On the 10,000-document workload, report a concrete feasible schedule and the hardware lower bound in \cref{sec:resource-lower-bounds}.  Call the schedule optimal only if its latency equals that bound.  Otherwise report the interval between them.
\end{enumerate}

Other implementations may later accelerate the same search.  They are engineering choices rather than part of the mathematical result, so this paper neither introduces nor assumes them.

\section{Resource lower bounds}
\label{sec:resource-lower-bounds}

For a fixed policy instance, let $U_{tot}$ be total dense new tokens, $A_{tot}$ total allowed attention pairs, $B_{KV,tot}$ compulsory KV bytes under a stated sharing model, and $B_{\min}$ any valid lower bound on the number of nonempty batches.  Then
\begin{equation}
LB_{\mathrm{res}}
=
\max\left\{
\frac{2PU_{tot}}{R_D},
\frac{B_{\min}W_{\mathrm{run}}}{BW}
\right\}
+
\max\left\{
\frac{4LhA_{tot}}{R_A},
\frac{B_{KV,tot}}{BW}
\right\}.
\label{eq:resource-lb}
\end{equation}
This lower bound may be loose because it globally pools resources that cannot necessarily be balanced within each batch.

\begin{proposition}[Equality certificate]
If a feasible schedule $\sigma$ satisfies $T_0(\sigma)=LB_{\mathrm{res}}$, then $\sigma$ is optimal under the analytical cost $\tau_0$.
\end{proposition}
\begin{proof}
Every feasible schedule costs at least $LB_{\mathrm{res}}$.  The exhibited schedule reaches it.
\end{proof}

A positive gap to \cref{eq:resource-lb} does not prove that the schedule is suboptimal; it only means that the available lower bound does not certify optimality.

\section{Experimental program}

\subsection{Research questions}

The first numerical study should answer:
\begin{enumerate}
  \item For $n=2$, what are the certified analytical latencies of task-first, pipeline, and full speculation as selectivity and document length change?
  \item How much does allowing chunked prefill improve the optimum relative to atomic requests under a makespan objective?
  \item When does finite KV capacity force pipeline eviction and recomputation?
  \item Which break-even conclusions change between Qwen3-4B and Qwen3-32B, and between H100 and L40S?
  \item How large is the value-of-information gap between the offline oracle and the best solved online policy?
  \item For $n>2$, when does bounded speculative lookahead beat both strict pipelining and full speculation?
\end{enumerate}

\subsection{Constructing the 10,000-document workload}

The motivating IMDb review histogram is measured in whitespace-separated words.  It must not be used as though its x-axis were Qwen tokens.  The preferred construction is:
\begin{enumerate}
  \item Sample 10,000 unique review IDs uniformly without replacement from the source dataset using a recorded seed.
  \item Store the raw text hash and tokenize each review with the exact tokenizer revision used by the model.
  \item Record every $d_i$ and reject or truncate only under a written context-window rule.
  \item Use the identical realized $\mathbf d$ for every policy, model, device, and selectivity comparison.
\end{enumerate}
If only the histogram is available, the fallback must specify a within-bin distribution, a model for the open-ended tail, and a measured words-to-Qwen-tokens conversion.  That synthetic workload is labeled as such.

The canonical workload file contains
\begin{equation}
(\texttt{doc\_id},\texttt{text\_hash},d_i,\texttt{sample\_seed},\texttt{tokenizer\_revision}).
\label{eq:workload-schema}
\end{equation}

\subsection{Outcome generation}

For $n=2$ and selectivity $s$, generate one Bernoulli vector $X^{(r)}$ per Monte Carlo replication using a recorded seed, independently of $\mathbf d$.  Reuse that vector across task-first and pipeline.  Full speculation is solved once per realized $\mathbf d$ because its schedule is outcome-independent.

For $n>2$, generate conditional stage outcomes only for rows that reach each stage.  Store the full latent matrix anyway so that every policy uses a coupled scenario.  The file records $(s_1,\ldots,s_{n-1})$, the random seed, and all $X_{ij}$.

\subsection{Hardware and model configuration}

Each run records:
\begin{itemize}
  \item model repository and immutable revision;
  \item tokenizer revision;
  \item $P,L,h,n_{KV},d_h,L_{ctx}$;
  \item total loaded weight bytes $W_{\mathrm{mem}}$ and per-batch compulsory transformer-weight traffic $W_{\mathrm{run}}$;
  \item KV dtype and $q_{KV}$;
  \item device SKU, physical memory $M$, bandwidth input $BW$, and dense/attention rate inputs;
  \item reserve $S$, scratch convention, chunk quantum $\delta$, and any sequence-count or new-token cap.
\end{itemize}
The primary speed-of-light curves use FP8 weights.  FP8 and BF16 KV are separate sensitivity cases unless one is explicitly selected as primary.

\subsection{Computation sequence}

The computational work proceeds in increasing scale:
\begin{enumerate}
  \item Hand-enumerate very small cases and confirm that the dynamic program returns the same schedule and latency.
  \item Run the exact dynamic program on small heterogeneous instances for all three policies, with and without chunking.
  \item Increase $N$ until exact enumeration becomes impractical; record that boundary instead of hiding it.
  \item For $N=10{,}000$, construct one explicit schedule for each policy, recompute its latency from the batch records, and compare it with the resource lower bound.
  \item Label a result ``exact'' only when it comes from completed enumeration or when a feasible schedule matches the lower bound.  Otherwise report the lower and upper bounds.
\end{enumerate}

\subsection{Schedule manifest}

For each batch $t$, the solver emits:
\begin{equation}
\begin{split}
(&t,\ \texttt{policy},\ \{(i,j,\texttt{token\_start},\texttt{token\_end})\},\\
&\texttt{input\_blocks},\ \texttt{generated\_blocks},\ \texttt{evicted\_blocks},\\
&U_t,A_t,K_{R,t},K_{W,t},M_t^{peak},D_t,H_t,\tau_t).
\end{split}
\label{eq:manifest-schema}
\end{equation}
The validator reconstructs which prefix tokens each operation used, checks that no unrevealed outcome was used, recomputes memory and cost, and confirms completion of every required operation.

\subsection{Plots}

The initial paper should contain:
\begin{itemize}
  \item latency versus first-stage selectivity for the three two-stage policies;
  \item latency versus document-length scale, including the unscaled empirical sample and context-feasible larger scales;
  \item a selectivity-by-length break-even map;
  \item atomic versus optimally chunked latency;
  \item peak resident KV and recomputed document tokens;
  \item number of batches, new-token fill, and active sequences per batch;
  \item hardware lower bound, feasible schedule latency, and the gap between them;
  \item H100 versus L40S and Qwen3-4B versus Qwen3-32B.
\end{itemize}
Latency plots use a logarithmic y-axis when device-model combinations span orders of magnitude.  Captions state whether curves are resource bounds, exact model optima, bounded solutions, calibrated estimates, or measured engine results.

\subsection{Break-even reporting}

For policies $g$ and $h$, define
\begin{equation}
\Delta_{g,h}(\mathbf s,\mathbf d)
=L_g(\mathbf s,\mathbf d)-L_h(\mathbf s,\mathbf d).
\label{eq:break-even}
\end{equation}
Integer batch boundaries make $\Delta$ nonsmooth.  Report an interval or grid cell in which the sign changes, together with Monte Carlo uncertainty and optimization gaps.  Do not report a high-precision root unsupported by the solver resolution.

\subsection{Monte Carlo and optimization uncertainty}

For $R$ independent outcome scenarios with exact offline optima,
\begin{equation}
\widehat L_{\mathrm{off}}(\mathbf d)
=\frac{1}{R}\sum_{r=1}^{R}\OPT_{\mathrm{off}}(\mathbf d,X^{(r)}).
\label{eq:mc-estimator}
\end{equation}
If scenario $r$ has only bounds $LB_r\le\OPT_r\le UB_r$, report
\begin{equation}
\frac1R\sum_rLB_r
\le
\frac1R\sum_r\OPT_r
\le
\frac1R\sum_rUB_r.
\label{eq:average-bounds}
\end{equation}
Sampling uncertainty across $X$ and solver uncertainty inside each scenario are different quantities.  Population-level estimates additionally resample $\mathbf d$ and must report that outer variance separately.

\section{Validation and claim discipline}

Before accepting a numerical result, verify:
\begin{itemize}
  \item Every document has its actual token length; no solver input substitutes a mean.
  \item Attention work is summed separately for each sequence, with no padding to the longest sequence and no attention across packed documents.
  \item Attention work across all chunks equals the work for the same unchunked left-to-right sequence.
  \item Every required input KV block is resident or produced earlier in the same batch.
  \item Shared blocks are counted once only when the batch manifest uses one physical block and the kernel assumption permits one load.
  \item Peak memory includes weights, reserve, resident blocks, temporary new blocks, and prompt branches.
  \item A non-speculative operation never crosses an unresolved outcome gate.
  \item An online decision depends only on its observed history.
  \item Evicted incomplete prefixes lose progress; evicted complete document prefixes must be recomputed before reuse.
  \item The independent validator reproduces $U,A,B_{KV},M^{peak}$ and every $\tau_t$.
  \item Exactness is claimed only with a matching bound or a zero solver gap.
\end{itemize}

\section{Limitations}

The analytical model is intentionally explicit about what it omits.
\begin{itemize}
  \item It assumes one GPU.  Tensor and pipeline parallelism introduce communication and additional placement decisions.
  \item It assumes dense Qwen3 checkpoints.  MoE models require active-expert work, routing, and expert-placement terms.
  \item It assumes binary decisions can be read from final-position logits.  Generated output tokens add a decode state and persistent branch KV.
  \item It treats the selected filter order as fixed and optimal.  Jointly choosing order and inference schedule is a larger problem.
  \item It assumes both task-first and document-first prompt templates retain acceptable accuracy.  This requires empirical validation.
  \item It permits a custom tree-aware kernel to load one document KV block for several same-batch branches.  Separate requests without such fusion receive no traffic credit.
  \item Nominal roofline rates are not measured efficiency.  Kernel shape, launch overhead, host scheduling, allocator behavior, and cache effects belong in the calibrated or measured layer.
  \item Chunking preserves mathematical work but can increase physical traffic.  The manifest and calibrated model must account for repeated reads.
  \item The exact online dynamic program is exponentially large.  Large-scale results will generally require bounds or restricted policy classes.
\end{itemize}

\section{Conclusion}

An $n$-stage AI filter query is simultaneously a short-circuit evaluation problem, a ragged GPU-packing problem, and a KV inventory problem.  The two-stage case is a useful warm-up, but the central object is the general stage frontier together with document-prefix progress, observed outcomes, and physical KV residency.  A valid optimal schedule may mix document lengths, divide long prefills into chunks, retain selected prefixes across gates, and speculate over a chosen block of future filters.

The next step is computational rather than rhetorical.  First implement and validate the exact dynamic program on small heterogeneous instances.  Then instantiate the workload with 10,000 actual Qwen-tokenized documents and report explicit feasible schedules beside their hardware lower bounds.  Only after those steps can the paper draw quantitative conclusions about when pipeline retention or fused speculation wins on H100 and L40S.

\appendix

\section{Consolidated notation}

\begin{longtable}{@{}p{0.20\textwidth}p{0.73\textwidth}@{}}
\caption{Core notation.}\label{tab:notation}\\
\toprule
Symbol & Definition \\
\midrule
\endfirsthead
\toprule
Symbol & Definition \\
\midrule
\endhead
$N,n$ & Number of documents and ordered filters. \\
$D_i,d_i$ & Raw document and its realized tokenizer-specific length. \\
$F_j,p_j$ & Filter $j$ and its prompt length. \\
$X_{ij}$ & Binary result of filter $j$ on document $i$. \\
$Y_{ij}$ & Indicator that document $i$ reaches stage $j$. \\
$s_j,\pi_j$ & Conditional pass probability at stage $j$ and probability of reaching it. \\
$z_i$ & Next unresolved stage for document $i$; $n+1$ means complete. \\
$r_i$ & Policy-specific resident prefix progress for document $i$. \\
$\mathcal K_t$ & Physical KV token blocks resident before batch $t$. \\
$Q(B)$ & New tokens evaluated in batch $B$. \\
$U(B)$ & Number of new token nodes in batch $B$. \\
$A(B)$ & Number of allowed query-key pairs in batch $B$. \\
$K_R(B),K_W(B)$ & Deduplicated resident KV token blocks read and new KV token positions materialized. \\
$P,W_{\mathrm{mem}},W_{\mathrm{run}}$ & Repeated dense parameter count, resident weight bytes, and per-batch compulsory transformer-weight traffic. \\
$\kappa$ & KV bytes per cached token. \\
$M,S$ & Physical GPU memory and non-weight/non-KV reserve. \\
$R_D,R_A,BW$ & Dense compute, attention compute, and memory-bandwidth ceilings. \\
$D(B),H(B),\tau(B)$ & Dense time, attention time, and assigned batch latency. \\
$\sigma,T(\sigma)$ & Concrete schedule and its makespan. \\
$\OPT_{\mathrm{off}},\OPT_{\mathrm{on}}$ & Clairvoyant and online optima under the stated model. \\
\bottomrule
\end{longtable}

\section{Independent schedule-checker contract}

The solver and validator must not share the objective implementation.  The validator consumes the immutable workload, outcome scenario, hardware-model configuration, and schedule manifest.  It performs the following deterministic pass:
\begin{enumerate}
  \item Initialize task-prefix blocks, logical frontiers, document progress, and resident KV.
  \item For each batch in order, reconstruct the prefix tokens visible to every operation and verify all input blocks.
  \item Verify information availability and the selected policy's speculation rules.
  \item Recompute $U_t$, $A_t$, the physical block union, KV writes, and peak memory.
  \item Apply completions, reveal outcomes, and then apply the listed evictions.
  \item Recompute $D_t,H_t,\tau_t$ from the configuration rather than trusting solver fields.
  \item At termination, verify that every logically required result exists and no required row remains unfinished.
\end{enumerate}
The checker returns a batch-indexed error, not a Boolean alone, so a faulty formulation can be diagnosed.

\section{Recommended implementation artifacts}

The computational repository should contain:
\begin{itemize}
  \item \texttt{workloads/documents.parquet}: immutable document IDs, hashes, and token lengths;
  \item \texttt{workloads/outcomes/}: coupled outcome matrices and selectivity metadata;
  \item \texttt{configs/models/}: Qwen architecture and measured weight footprints;
  \item \texttt{configs/devices/}: H100 and L40S capacities, rates, reserve, and KV dtype;
  \item \texttt{solver/exact\_dp/}: exact small-instance dynamic program and memoized state table;
  \item \texttt{validator/}: independent manifest checker;
  \item \texttt{results/manifests/}: one concrete schedule per reported point;
  \item \texttt{results/bounds/}: solver lower bounds, upper bounds, gaps, and logs;
  \item \texttt{plots/}: scripts that read result tables rather than solver internals.
\end{itemize}

\section*{References}

\begin{enumerate}[label={[\arabic*]},leftmargin=2.2em]
\item S. Williams, A. Waterman, and D. Patterson. ``Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures.'' \emph{Communications of the ACM}, 52(4), 2009. \href{https://dl.acm.org/doi/10.1145/1498765.1498785}{ACM DOI}.
\item A. Yang et al. ``Qwen3 Technical Report.'' arXiv:2505.09388, 2025. \href{https://arxiv.org/abs/2505.09388}{arXiv}.
\item Qwen Team. ``Qwen3-4B-FP8 Model Card and Configuration.'' \href{https://huggingface.co/Qwen/Qwen3-4B-FP8}{Model card}.
\item Qwen Team. ``Qwen3-32B-FP8 Model Card and Configuration.'' \href{https://huggingface.co/Qwen/Qwen3-32B-FP8}{Model card}.
\item NVIDIA. ``H100 Tensor Core GPU Specifications.'' Accessed July 31, 2026. \href{https://www.nvidia.com/en-us/data-center/h100/}{Product page}.
\item NVIDIA. ``L40S GPU Specifications.'' Accessed July 31, 2026. \href{https://www.nvidia.com/en-us/data-center/l40s/}{Product page}.
\item A. Agrawal et al. ``SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills.'' arXiv:2308.16369, 2023. \href{https://arxiv.org/abs/2308.16369}{arXiv}.
\item A. Agrawal et al. ``Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.'' arXiv:2403.02310, 2024. \href{https://arxiv.org/abs/2403.02310}{arXiv}.
\item W. Kwon et al. ``Efficient Memory Management for Large Language Model Serving with PagedAttention.'' \emph{SOSP}, 2023. \href{https://arxiv.org/abs/2309.06180}{arXiv}.
\item T. Dao et al. ``FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.'' \emph{NeurIPS}, 2022. \href{https://arxiv.org/abs/2205.14135}{arXiv}.
\item M. R. Garey and D. S. Johnson. ``Strong NP-Completeness Results: Motivation, Examples, and Implications.'' \emph{Journal of the ACM}, 25(3), 1978. \href{https://dl.acm.org/doi/10.1145/322077.322090}{ACM DOI}.
\item M. L. Puterman. \emph{Markov Decision Processes: Discrete Stochastic Dynamic Programming}. Wiley, 1994.
\end{enumerate}

\end{document}