% ARCHIVED 2026-08-04.
%
% What this is: the theory program's working paper, "Scheduling
% n-Stage AI Filters Under KV-Cache Constraints" — the finished
% first generation of this repo. It defines the batch-level cost
% model and the exact schedule solvers that the code in attic/
% implements.
%
% What supersedes it: paper/PAPER.md ("Plan-Governed KV State for
% Semantic Scans") is the submission now. This file is kept because
% banked claims still cite its machinery.
%
% What survives where:
% - The retention theory lives on as the new paper's Section 6
%   (the plan-aware KV retention problem, its caching reduction,
%   and its complexity results).
% - The solver semantics live on as the attic's validation layer:
%   the exact solvers, the expected-flow linear program, and the
%   replay checker, kept runnable beside this file. Their evidence
%   fills the first half of notes/RESULTS.md.
% - The review (notes/REVIEW.md) found four real errors, all fixed
%   in the solvers and to be fixed in any future edit of this text:
%   (1) the attention-compute formula used the hidden width where
%   query heads times head width belongs, a 1.6-times undercount on
%   both Qwen3 models; (2) the NP-hardness reduction from
%   3-PARTITION collapsed once documents may split, and holds only
%   for indivisible documents unless capacity is routed through the
%   memory limit; (3) the online recurrence has loops, so it is a
%   stochastic shortest-path problem solved by value iteration, not
%   a recursion to evaluate; (4) the lower bound was built from
%   schedule-dependent totals, and must take each total's minimum
%   over feasible schedules, with the batch count derived from the
%   memory limit.
%
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
\newcommand{\residentBytes}{m_{\mathrm{res}}}
\newcommand{\xferBytes}{m_{\mathrm{xfer}}}

\title{\color{ink}\textbf{Scheduling $n$-Stage AI Filters\\Under KV-Cache Constraints}}
\author{Working paper}
\date{July 31, 2026}

\begin{document}
\maketitle

\begin{abstract}
An AI SQL query can apply an ordered conjunction of language-model filters to every row of a table.  Each surviving row advances to the next filter, so filter selectivity determines which model calls are necessary.  At the same time, token order determines which prefixes can be reused, finite GPU memory determines which key-value (KV) blocks can remain resident, and batching determines how efficiently the GPU is used.  We formulate the resulting $n$-stage scheduling problem for a fixed filter order chosen by an upstream query optimizer.  The model keeps the realized, heterogeneous document lengths; permits variable-length sequences in the same batch; permits prefill to be divided into valid chunks; and records exactly which prefix tokens each new token may attend to.  We give an FP8 batch-latency model and an expected-flow linear-program relaxation over observable cache states and feasible next batches.  One outer formulation covers task-first prefix caching, document-first pipelining, full speculation, and bounded lookahead speculation by changing the method-specific state and action sets.  Every stable stochastic scheduler with the stated workload mix and action set induces a feasible LP occupation measure, so the LP supplies an upper bound on sustainable throughput and an asymptotic fluid latency target.  Because the main state omits finite host-queue inventories, LP feasibility alone does not prove that its rates are executable without queue starvation; a concrete replay instead establishes one finite 10,000-document execution with explicit fill, repair, and drain costs.  Finite-workload Bellman equations remain as an exact small-instance reference.  The paper separates a hardware resource bound, the asymptotic fluid target, a constructed finite schedule, and latency measured from a concrete inference engine.  It also specifies the data, schedule records, and plots required for reproducible latency and break-even experiments on Qwen3-4B-FP8 and Qwen3-32B-FP8 using H100 and L40S GPUs.
\end{abstract}

\noindent\textbf{Keywords:} AI SQL, LLM inference, query optimization, continuous batching, KV cache, chunked prefill, speculative execution, linear programming

\section{Introduction}

Consider a query of the form
\begin{center}
\small\ttfamily
SELECT * FROM documents WHERE\\[-2pt]
AI\_FILTER($F_1$, document) AND $\cdots$ AND AI\_FILTER($F_n$, document).
\end{center}
The logical query optimizer has already chosen the order $F_1,\ldots,F_n$.  Execution remains nontrivial.  A row rejected by $F_j$ requires no later filter, but that fact is known only after $F_j$ completes.  Processing all future filters immediately removes these information barriers but performs speculative work.  Waiting preserves short-circuit semantics but may separate operations that could have shared the document's KV state.  Retaining that state saves recomputation but consumes capacity needed by other rows.  Finally, model throughput is a property of a GPU batch, not of a single logical call.

The workload considered here is deliberately prefill-heavy.  A query makes thousands of calls against distinct documents, each task prompt is short, and each filter returns a binary decision from the logits at the final sequence position used for classification.  There is no material autoregressive decode phase.  The objective is therefore not time to first token or inter-token latency.  It is makespan: the time until the SQL query has produced every required row-level result.

Two properties prevent a token-count-only analysis.  First, documents have heterogeneous lengths.  The optimizer receives the full realized vector $(d_1,\ldots,d_N)$ and may mix arbitrary lengths in a ragged batch.  Replacing this vector by a mean changes attention work, memory feasibility, and packing decisions.  Second, a document may be processed in several chunks.  Chunking does not reduce its mathematical transformer work, but it expands the set of feasible packings and can eliminate underfilled batches.

This paper develops one model that can be used in two ways.  With nominal device ceilings and zero software overhead, it defines a speed-of-light scheduling experiment.  With calibrated rates and measured reserve memory, it becomes an empirical predictor.  In both cases, the same schedule manifest lists exact document IDs, filter stages, token chunks, versioned blocks, physical transfer events, pre-evictions, and final resident sets.

The main contributions are:
\begin{itemize}
  \item A formal $n$-stage AI-filter workload model with a fixed, externally optimized filter order, heterogeneous document lengths, conditional stage selectivities, and distinct offline and online information structures.
  \item A batch abstraction that supports variable-length sequences, chunked prefill, explicit within-batch dependencies, and physical KV-block sharing.  It counts a shared prompt or document block once when the assumed kernel loads it once.
  \item A transparent FP8 latency model based on dense transformer work, attention work, model-weight traffic, KV traffic, and peak HBM feasibility.  Weight precision and KV precision remain separate parameters.
  \item A single expected-flow linear-program template for state-conditioned continuous batching.  The same equations cover task-first execution, document-first pipelining, full speculation, bounded lookahead speculation, and optional hybrids by changing the observable cache states and allowed actions.
  \item A proof that every stable scheduler maps to a feasible LP occupation measure, making the LP optimum an upper bound on sustainable throughput.  We state precisely why the converse need not hold when finite host-queue inventories are omitted, and we convert selected LP rates into a finite schedule whose fill, core, repair, and drain costs are measured separately.
  \item Exact finite-workload Bellman equations for small-instance validation and an experiment specification that labels every reported point as a resource lower bound, an asymptotic fluid LP target, a constructed finite schedule, or a measured engine result.
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
Y_{ij}=\prod_{k=1}^{j-1}X_{ik}
\qquad (j\ge 2).
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
$n_Q$ & Query heads per layer & heads \\
$n_{KV}$ & KV heads per layer & heads \\
$d_h$ & Attention head dimension & elements \\
$w_Q=n_Qd_h$ & Total query-vector width processed by attention & elements \\
$q_{KV}$ & Bytes per stored KV element & bytes/element \\
$\kappa_{\mathrm{meta}}$ & Per-token quantization scales and storage metadata & bytes/token \\
$\kappa$ & KV bytes per cached token & bytes/token \\
$\residentBytes(b)$ & HBM allocation size of physical block $b$ & bytes \\
$\xferBytes(e)$ & Bytes moved by physical load/store event $e$ & bytes \\
$M$ & Physical accelerator memory & bytes \\
$S$ & Non-weight, non-KV reserve: runtime, allocator slack, and mandatory scratch & bytes \\
$R_D$ & Dense FP8 tensor throughput ceiling used by the model & FLOP/s \\
$R_A$ & Attention-compute throughput ceiling used by the model & FLOP/s \\
$BW$ & Device-memory bandwidth ceiling & bytes/s \\
$L_{ctx}$ & Maximum tokens on any root-to-leaf causal path & tokens \\
\bottomrule
\end{tabularx}
\end{table}

Weight precision and KV precision are logically independent.  This paper's primary experiment fixes both the checkpoint weights and KV elements to FP8, so $q_{KV}=1$ in every headline result.  Any later FP16/BF16 KV sensitivity study is reported separately and must record its scale and layout overhead in $\kappa_{\mathrm{meta}}$.

\subsection{KV footprint}

Grouped-query attention stores one key and one value for every KV head and layer.  Therefore
\begin{equation}
\kappa=2L n_{KV}d_hq_{KV}+\kappa_{\mathrm{meta}}.
\label{eq:kv-token-bytes}
\end{equation}
The factor two denotes K and V.  It is unrelated to the multiply-add convention used for FLOPs.  The idealized tables below set $\kappa_{\mathrm{meta}}=0$; an implementation uses the measured allocated bytes per cached token, including scales and alignment.

\begin{table}[ht]
\centering
\caption{Qwen3 reference architecture inputs.  The configured maximum position value is recorded separately from any recommended native-context regime.}
\label{tab:qwen}
\small
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Model & $P$ & $L$ & $h$ & $n_Q/n_{KV}$ & $d_h$ & $\kappa$ at $q_{KV}=1$ & $L_{ctx}$ \\
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

Each query head performs one query-key dot product and one attention-value accumulation for every allowed query-key pair.  Define the total query-vector width
\begin{equation}
w_Q=n_Qd_h.
\label{eq:query-width}
\end{equation}
The first-order attention FLOPs across $L$ layers are
\begin{equation}
F_A(B)=4Lw_Q\,A(B)=4Ln_Qd_h\,A(B).
\label{eq:attention-flops}
\end{equation}
Grouped-query attention reduces the number of stored K and V heads, so $n_{KV}$ controls the KV footprint in \cref{eq:kv-token-bytes}.  It does not reduce the number of query heads that perform attention compute.  For Qwen3-4B, $w_Q=32\cdot128=4{,}096$ while $h=2{,}560$; for Qwen3-32B, $w_Q=64\cdot128=8{,}192$ while $h=5{,}120$.  Substituting $h$ would undercount attention work by a factor of $1.6$ for both models.
The effective attention ceiling $R_A$ need not equal the dense tensor-core ceiling $R_D$.  Setting them equal is an optimistic analytical choice that must be labeled; a calibrated model estimates $R_A$ from attention microbenchmarks.

\subsection{Physical KV blocks and transfer accounting}

The model uses a block-event convention.  A physical block has an immutable, versioned identifier, semantic token span, producer operation, HBM resident size $\residentBytes(b)$, and causal consumers.  Recomputing the same logical prefix creates a new block version.  Every HBM load or store event names the block, transferred byte range, kernel group, and event ordinal.  A load additionally states whether the block was resident at launch or was produced earlier in the same batch.

Let $\mathcal E_L(B)$ and $\mathcal E_S(B)$ be all HBM load and store events.  Let $K_L(B)$ be the sum of semantic token spans over \emph{all} load events, including same-batch reloads, and let $K_W(B)$ be the corresponding sum over stores of newly materialized KV positions.  Consumers in one supported fused producer-consumer group can stream a produced block without a load event.  A consumer in another kernel group creates a load event even when its block was produced earlier in $B$.  Distinct load groups create distinct events even when their logical tokens match.

The constant-$\kappa$ token-equivalent ledger is
\begin{equation}
B_{KV}^{\mathrm{tok}}(B)
=\kappa\bigl(K_L(B)+K_W(B)\bigr).
\label{eq:kv-traffic}
\end{equation}
The exact transferred-byte ledger is
\begin{equation}
B_{KV}^{\mathrm{xfer}}(B)
=\sum_{e\in\mathcal E_L(B)}\xferBytes(e)
+\sum_{e\in\mathcal E_S(B)}\xferBytes(e).
\label{eq:xfer-kv-traffic}
\end{equation}
Transfer bytes include the payload, scales, and transaction overhead actually moved for the recorded byte range.  They are not the allocation size of a partially occupied page.  A run selects either \cref{eq:kv-traffic} or \cref{eq:xfer-kv-traffic} and denotes the selected quantity by $B_{KV}(B)$ everywhere below.

The primary analytical action set uses an explicit \emph{write-through fused ledger}: every new KV position with a later causal consumer has one store event; only a terminal leaf position with no descendant may be ephemeral.  Same-group fusion can remove its later load, but it does not remove the assigned store.  This is a physical-method convention, not a universal hardware lower bound.  A fully streamed kernel that also removes the store is a different action family and is admitted only if its finite on-chip capacity and lifetime rules are specified.  Arbitrary ``on-chip'' declarations are not allowed in the primary model.

\subsection{Peak-memory feasibility}

Each physical action contains a topological event order.  Every HBM allocation $b$ records a birth ordinal $\operatorname{birth}_B(b)$ and a free ordinal $\operatorname{free}_B(b)$ strictly after its last use.  Blocks resident after the action's pre-evictions are live at ordinal zero; new KV and batch-dependent scratch allocations become live at their recorded birth.  Define
\begin{equation}
\mathcal L_B(r)
=\left\{b:\operatorname{birth}_B(b)\le r
<\operatorname{free}_B(b)\right\}.
\label{eq:live-hbm-set}
\end{equation}
The exact peak-memory condition is
\begin{equation}
M^{\mathrm{peak}}(B)
=W_{\mathrm{mem}}+S
+\max_r\sum_{b\in\mathcal L_B(r)}\residentBytes(b)
\le M.
\label{eq:hbm-capacity}
\end{equation}
The constant-$\kappa$ idealization replaces $\residentBytes(b)$ by $\kappa$ times the block's token span.  Launch-resident, newly materialized, and temporary HBM blocks are all included during their actual lifetimes, even when discarded at the batch boundary.  Shared physical blocks are counted once.  Constant mandatory scratch is in $S$; shape-dependent scratch is an explicit lifetime-tracked allocation.  A store must follow the block's birth, every load must precede its free event, and no identifier can be used after eviction.

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
\frac{4Lw_Q\,A(B)}{R_A},
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

\section{Finite-workload reference problem}

\subsection{Schedules}

\begin{definition}[Feasible batch]
A batch specifies the document and filter token chunks evaluated together, its operation order, physical fusion/load/store events, allocation lifetimes, and shared prefixes.  It is feasible when: (i) every required earlier filter result is already known, unless the batch explicitly speculates; (ii) every causal input is supplied by a launch-resident load, a validated same-group stream, or a store followed by a same-batch reload; (iii) the HBM and context limits in \cref{eq:hbm-capacity,eq:context-limit} hold; and (iv) any configured token or sequence-count limits hold.
\end{definition}

\begin{definition}[Schedule]
A realized schedule is
\begin{equation}
\sigma=(E^{\mathrm{pre}}_1,B_1,R_1,\ldots,
E^{\mathrm{pre}}_T,B_T,R_T).
\end{equation}
At step $t$, $E_t^{\mathrm{pre}}$ is applied to the observed cache state before launch; it is not a standalone zero-time action.  Batch $B_t$ then executes under its topological kernel plan.  After its outcomes $\xi_t$ are visible, the contingent rule $R_t(\xi_t)$ names the exact final resident set and hence the next cache state.  Every logically or speculatively selected operation completes exactly once, except that an evicted incomplete prefix may later be recomputed under a new block version.  Every evicted unfinished row remains represented by exactly one uncached host item at the same logical frontier unless it is consumed or recomputed in that same action.
\end{definition}

For one GPU, batches execute sequentially.  The modeled makespan is
\begin{equation}
T_{\theta}(\sigma)=T_{\mathrm{init}}+\sum_{t=1}^{T}\tau_{\theta}(B_t),
\label{eq:schedule-cost}
\end{equation}
where $T_{\mathrm{init}}$ permits an explicitly priced external initialization.  The primary experiment sets $T_{\mathrm{init}}=0$: task-prefix construction and every other KV-producing operation appear as ordinary batches in $\sigma$, count against peak HBM, and contribute to pipeline fill.  This convention is also used by the lower bound and finite-schedule validator, so initialization is neither omitted nor counted twice.

\subsection{State and action}

A complete state $S_t$ contains:
\begin{itemize}
  \item the next unresolved logical stage of every document;
  \item the resident progress of every incomplete causal prefix;
  \item the physical KV blocks in $\mathcal K_t$;
  \item all filter outcomes revealed before batch $t$;
  \item policy-specific branch or recomputation status.
\end{itemize}
An offline action chooses pre-evictions, a feasible batch, and the final resident set.  Because this oracle knows $X$, it may use future outcomes when making that choice.  An online scheduler first chooses the pre-evictions and batch.  After that batch finishes and reveals its results, it chooses the final resident set.  The transition updates completed work, observed results, resident KV, and uncached host items for prefixes that must later be recomputed.

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

For an online state whose history is $\mathcal H(S)$, let $\calB_g(S;\mathbf d)$ be the feasible launch actions---pre-evictions together with a batch---that can be selected before the next outcomes are known, and let $\mathcal F_g(S,B,\xi)$ be the feasible final resident sets after batch $B$ reveals outcome vector $\xi$.  Then
\begin{equation}
V^{\mathrm{on}}_g(S;\mathbf d,\mathbf s)
=
\min_{B\in\calB_g(S;\mathbf d)}
\left\{
\tau_{\theta}(B)
+\E\left[
\min_{F\in\mathcal F_g(S,B,\xi)}
V^{\mathrm{on}}_g(T_g(S,B,\xi,F);\mathbf d,\mathbf s)
\mathrel{}\middle|\mathrel{} \mathcal H(S),B
\right]
\right\}.
\label{eq:online-bellman}
\end{equation}
The outer minimization chooses pre-evictions and the batch without unseen outcomes.  The inner minimization chooses the final resident set only after $\xi$ is visible.  Equivalently, the scheduler may choose in advance a contingent retention rule with one branch for each possible $\xi$; neither representation is clairvoyant.

\begin{assumption}[Proper finite reference problem]
The discretized finite-workload state and action sets are finite, every nonterminal decision has cost at least $\epsilon>0$, and there exists one policy that, from every reachable state, reaches the terminal state with probability one and finite expected cost.
\end{assumption}

\begin{proposition}[Bellman optimality and computation]
Under the proper-reference assumption, \cref{eq:offline-bellman,eq:online-bellman} have the optimal finite-workload makespan and the optimal finite-workload \emph{expected} makespan, respectively, as their unique minimal nonnegative solutions within their stated policy and information classes.  The offline problem is a deterministic shortest-path problem.  The online problem is a stochastic shortest-path problem, and value iteration initialized at zero converges monotonically to its value.
\end{proposition}
\begin{proof}
Let $\mathcal T$ denote the appropriate Bellman operator and set $V_k=\mathcal T^k0$.  Equivalently, give the state reached after $k$ nonterminal decisions zero continuation cost.  Backward induction gives the $k$-step truncated accumulated cost, including the online order ``launch action, outcome, final resident set.''  The sequence $V_k$ is monotone and bounded above by the cost of the proper policy, so it converges to a finite limit $\bar V$.  Finiteness of the state and action sets permits passage through the minimum and expectation, hence $\bar V=\mathcal T\bar V$.

Choose a stationary action attaining the minimum in every state.  If the resulting policy $\mu$ were improper, iterating the fixed-point equality would accumulate at least $\epsilon$ for every nonterminal decision and yield infinite expected cost, contradicting finiteness of $\bar V$.  Thus $\mu$ is proper, and iteration until absorption gives $\bar V=J_\mu\ge V^\star$.  Conversely, every truncated value satisfies $V_k\le V^\star$, so $\bar V\le V^\star$.  Therefore $\bar V=V^\star$.  Every nonnegative fixed point dominates $\mathcal T^k0$ for every $k$, so $V^\star$ is the unique minimal nonnegative fixed point.  Deterministic transitions give the offline shortest-path specialization; expectation over $\xi$ gives the online stochastic-shortest-path specialization.
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

\subsection{Four different latency claims}

The paper reports four layers separately:
\begin{enumerate}
  \item A \emph{resource lower bound} that no schedule can beat under stated hardware ceilings.
  \item An \emph{asymptotic fluid LP target}: the reciprocal of an expected-flow throughput upper bound over stated cache states and feasible actions.
  \item A \emph{constructed finite latency} obtained by assigning the 10,000 realized documents to an explicit sequence of batches and evaluating \cref{eq:schedule-cost}.
  \item A \emph{measured latency} from an implementation, which includes software and kernel effects absent from $\tau_0$.
\end{enumerate}
The LP optimum is exact only for the expected-flow relaxation and enumerated state-action set.  Without host-queue inventories in the state, its reciprocal is an asymptotic target, not a certified finite-$N$ bound or an automatically attainable steady-state latency.  A finite schedule may differ because of queue starvation, integer batches, pipeline fill and drain, and random survivor counts.  Equality between a constructed schedule and a valid resource lower bound certifies the finite analytical optimum for that instance.  None of these claims proves that an existing serving engine attains the schedule.

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
The exact task-first Bellman equations are \cref{eq:offline-bellman,eq:online-bellman} with this state and the action rules above.  A batch may mix $F_1$ and already eligible $F_2$ chunks from different documents.  It may not place $F_2(i)$ in the same online batch that first reveals $X_i$.

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
  \item a contingent final resident set after the outcomes are revealed.
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

The inequality may be strict because an offline scheduler can retain precisely the document KVs that will be needed by $F_2$.  An online scheduler can condition its final resident set on outcomes just revealed, but it cannot recover prefixes evicted earlier or anticipate outcomes of unfinished $F_1$ calls.

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

\subsection{General state and Bellman equations}

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

Substituting these states and their policy-specific feasible batches into \cref{eq:offline-bellman,eq:online-bellman} gives the exact $n$-stage Bellman equations.  The two-stage models are the $n=2$ specialization.

\begin{proposition}[Full speculation ignores selectivity]
For full $n$-stage speculation, the feasible action set, transition sequence, and analytical cost are independent of $X$ and $\mathbf s$.
\end{proposition}
\begin{proof}
Every document receives all $n$ prompt branches, regardless of outcomes.  Outcome values affect only the returned SQL result, not future speculative work or feasibility.
\end{proof}

\subsection{Complexity}
\label{sec:complexity}

Even with atomic jobs and no outcome uncertainty, selecting feasible batches under capacity contains bin packing.  KV retention adds inventory state; chunking adds progress state; and partial speculation adds lookahead decisions.  The Bellman equations are exact but exponentially large.

\begin{proposition}[Indivisible single-fusion-group speculation]
The offline scheduling problem restricted to indivisible full-speculation jobs that each execute in one supported fusion group is strongly NP-hard.  Here ``indivisible'' means that a document and all of its speculative prompt branches must appear as one job; neither the document prefill nor its branches may be split across batches.
\end{proposition}
\begin{proof}
Reduce from 3-PARTITION.  Given $3m$ integer sizes $a_i$ satisfying $B/4<a_i<B/2$ and $\sum_i a_i=mB$, use $n=2$ one-token prompts and create one indivisible full-speculation job with document length $d_i=a_i$.  Its new-token size is $u_i=a_i+2$.  Set the batch new-token cap to $C=B+6$.  Then
\begin{equation}
\sum_i u_i=mC,
\qquad
\frac{C}{4}<u_i<\frac{C}{2}.
\end{equation}
Make context and HBM capacity nonbinding.  Let $c_W=W_{\mathrm{run}}/BW>0$ and choose a finite $R_D$ satisfying $2PC/R_D\le c_W$.  Therefore $D(B)=c_W$ for every nonempty feasible batch.

Independent documents never attend to one another, so there is a positive integer $A_i$ for each job such that $A(B)=\sum_{i\in B}A_i$.  Every constructed job executes in one supported fused producer-consumer group.  Thus $\mathcal E_L(B)=\varnothing$, while each new token creates at most one store event, so
\begin{equation}
K_W(B)\le U(B)\le C,
\qquad
B_{KV}(B)\le\kappa C.
\end{equation}
Choose the finite rational rate
\begin{equation}
R_A=\frac{4Lw_QBW}{\kappa C}.
\end{equation}
Because every nonempty batch has $A(B)\ge1$, this choice gives $4Lw_QA(B)/R_A\ge B_{KV}(B)/BW$ for every feasible batch.  Hence
\begin{equation}
H(B)=\frac{4Lw_Q}{R_A}\sum_{i\in B}A_i,
\qquad
\sum_tH(B_t)=C_A
\end{equation}
for a constant $C_A$ independent of the packing.  A complete schedule with $q$ nonempty batches therefore has
\begin{equation}
T_0=c_Wq+C_A.
\end{equation}
A schedule of latency at most $mc_W+C_A$ exists exactly when the jobs fit in $m$ batches.  Total volume forces every one of those batches to be full, and the strict size bounds force exactly three jobs per batch.  Removing the two prompt tokens from each job gives three original sizes summing to $B$ in every batch, which is a 3-partition.  The transformation is polynomial, so the restriction is strongly NP-hard.
\end{proof}

This proposition concerns the explicitly indivisible restriction.  The hardware rates and capacity are inputs to the abstract complexity instance, and the proof uses the ideal constant-$\kappa$ traffic model; it is not a hardness statement for fixed H100 parameters.  Nor does it claim that the same reduction remains valid when documents or speculative branches may be split across batches; chunking destroys this particular packing reduction.

\section{The steady-state expected-flow LP}
\label{sec:steady-state-lp}

The finite Bellman equations define a reference optimum but are not the primary computational object for a 10,000-document backlog.  The speed-of-light question is instead: what sustained rate is consistent with GPU time, logical work conservation, and observable KV-cache evolution while batches from different filter stages are continuously interleaved?  We answer that question first with an expected-flow linear program over observable cache states and feasible next batches.  The cache state must be explicit: a batch-rate model without it could use a retained document prefix in a later batch without proving that the prefix survived in HBM.

The main LP deliberately does \emph{not} put finite host-queue inventories in the Markov state.  It balances every host queue only in expectation.  This makes the model small enough to use as a planning relaxation, but it also creates a precise limitation: mean flow balance does not rule out sample-path queue starvation or correlations between a cache state and which logical work is ready.  Consequently, the LP optimum is a throughput upper bound, not an achievability theorem.  An exact finite-state average-throughput model is obtained by augmenting the state with bounded host-queue inventories; \cref{sec:lp-claim} states both facts formally.

\subsection{Empirical length types and host queues}

Let $\mathcal L$ be the set of document-length types.  For the exact empirical distribution, one type is created for every distinct realized token length.  Let $N_\ell$ be the number of sampled documents of type $\ell$, let $d_\ell$ be that type's token length, and define
\begin{equation}
p_\ell=\frac{N_\ell}{N},
\qquad
\sum_{\ell\in\mathcal L}p_\ell=1.
\label{eq:length-mass}
\end{equation}
No mean document length appears in the optimization.

Let $\mathcal R_g$ be the finite set of host-queue \emph{types} for method $g$.  A type $r$ records at least a length type $\ell$, the next unresolved filter $j$, and whether document KV must be created or recomputed.  The LP records rates for these types, not their instantaneous queue lengths.  The initial type for a new length-$\ell$ document is written $r^0_{\ell1}$.  Define its external input mass by
\begin{equation}
b_r=
\begin{cases}
p_\ell,&r=r^0_{\ell1},\\
0,&\text{otherwise}.
\end{cases}
\label{eq:external-input-mix}
\end{equation}
A row represented by a resident KV block is recorded in the cache state and is \emph{not} simultaneously counted in a host queue.  This separation prevents the LP from consuming one logical row twice or pairing queued work with a cache block that does not exist.

\subsection{Observable cache states and feasible actions}

Fix a physical method $g$.  Let $\mathcal S_g$ be a finite set of observable states immediately after the preceding batch has finished and its outcomes are known.  A state records the resident physical KV blocks, tagged by the corresponding row's length type, next unresolved stage, prefix progress, and resident byte size.  It also records any pinned shared task prefixes.  Counts may replace document identities only under an additional within-type exchangeability assumption: conditional on the recorded length, frontier, progress, and physical-status counts, the joint outcome law and every action cost are invariant to row identities and prior history not already in the state.  Length-independent marginal selectivity alone does not imply this property.  If exchangeability fails, the necessary identities or history variables belong in $s$.  Finiteness is obtained by fixing the length, token-chunk, cache-page, and cache-state quanta and by retaining only states that fit HBM; an implementation may additionally bound the active scheduling window.

Let $\mathcal A_g(s)$ be the feasible next actions in state $s$.  One action specifies:
\begin{itemize}
  \item any pre-batch evictions;
  \item the concrete batch, including every document, stage, and token chunk;
  \item which physical KV blocks remain resident at launch, which load group reads each block, and which new blocks are materialized;
  \item a contingent post-batch retention rule that maps the outcomes just revealed to blocks retained for the next state.
\end{itemize}
Specifying a contingent retention rule is not clairvoyance: the rule is chosen before the batch, but its prescribed branch is applied only after the corresponding outcome becomes visible.

For each state-action pair $(s,a)$, precompute
\begin{equation}
U_{sa},\ A_{sa},\ K_{L,sa},\ K_{W,sa},\
B_{KV,sa}^{\mathrm{xfer}},\ M_{sa}^{\mathrm{peak}},
\qquad
\tau_{sa}=\tau_\theta(B_{sa}).
\label{eq:state-action-time}
\end{equation}
The action first applies its listed pre-batch evictions.  Thus its block snapshots satisfy
\begin{equation}
\mathcal K_{sa}^{\mathrm{launch}}
=\mathcal K_s\setminus\mathcal E_{sa}^{\mathrm{pre}}.
\label{eq:pre-eviction-state}
\end{equation}
Every block in $\mathcal K_{sa}^{\mathrm{launch}}$ counts toward peak HBM even when the batch does not read it; a pre-evicted block does not count and cannot be an input.  A resident prefix that a branch attends to still contributes a load event; residency avoids recomputing or transferring that prefix from the host, not the attention read itself.  New HBM blocks count at peak whether they are retained or discarded after the batch.  An action is included only if its dependencies, outcome gates, context paths, sequence limits, block events, and peak HBM satisfy the stated model.

Logical conservation is checked at the same boundary.  Every unfinished resident row removed by a pre-eviction or omitted from the contingent final resident set produces exactly one uncached host item at the same logical frontier, unless that row is consumed, recomputed, or completed in the same action.  It may not disappear from both the cache state and the host queues.  Two actions with the same logical token nodes but different fusion groups, load/store events, transfer ranges, or allocation lifetimes are distinct physical actions with separately computed costs.

The cost $\tau_{sa}$ is deterministic before the outcomes are known.  The state-action table stores the complete mapping $R_{sa}(\omega)$ from every possible outcome to its final resident set.  Any generated block appearing in $\bigcup_\omega R_{sa}(\omega)$ is materialized and charged in $K_{W,sa}$ for every outcome and remains live through the outcome boundary; its free ordinal cannot precede that boundary.  The contingent branch decides only whether that already-produced block remains in $s'$.  If an implementation instead has outcome-dependent traffic or runtime, the GPU-time coefficient must be $\E[\tau_{sa}(\omega)]$ and peak HBM must be checked for every outcome.  The primary model uses deterministic coefficients.

Let
\begin{equation}
P_g(s'\mid s,a)
\label{eq:cache-transition-law}
\end{equation}
be the probability that the observed next cache state is $s'$.  This law is derived from the filter-outcome model and the action's retention rule.  Selectivities alone determine multirow transition probabilities only when outcomes are conditionally independent across rows; otherwise the joint outcome law must be supplied explicitly.

Finally, let $u_{sa,r}$ be the number of host-queue items of type $r$ consumed by the action.  For outcome vector $\omega$, let $v_{sa,r}(\omega)$ be the host-queue items produced and let $h_{sa}(\omega)$ be the number of documents completed.  Write
\begin{equation}
\bar v_{sa,r}=\E[v_{sa,r}(\omega)\mid s,a],
\qquad
\bar h_{sa}=\E[h_{sa}(\omega)\mid s,a].
\label{eq:expected-action-output}
\end{equation}
The same outcome vector determines $P_g(s'\mid s,a)$, $\bar v_{sa,r}$, and $\bar h_{sa}$; they must be generated together so their one-step marginals are consistent.  This does not restore the cache--queue correlation discarded by the expected-flow relaxation.

For example, if a strict stage-$j$ action consumes an uncached row and does not retain its document prefix, it produces a host item for stage $j+1$ with expectation $s_j$ when $j<n$.  If it retains a passing prefix, the survivor appears in the next cache state instead and no host item is produced.  A speculative block $F_j,\ldots,F_k$ advances only with probability
\begin{equation}
\rho_{j,k}=\prod_{h=j}^{k}s_h.
\label{eq:speculative-survival}
\end{equation}
If $k=n$ or any evaluated filter fails, the row completes and contributes to $h_{sa}(\omega)$.  Every prompt in the block still contributes to the batch cost, including prompts that turn out to have been unnecessary after an early failure.

\subsection{The same outer LP, different physical actions}

The equations below are common to every method.  The state set, feasible actions, transition law, and batch costs are method-specific.
\begin{table}[ht]
\centering
\small
\begin{tabularx}{\textwidth}{@{}p{0.18\textwidth}X@{}}
\toprule
Method & Permitted state and action behavior \\
\midrule
Task-first & A request has the form $[F_j,D_i]$.  The shared $F_j$ prompt block may be reused across documents, but document KV produced under $F_j$ cannot serve $F_{j+1}$.  A survivor creates a fresh stage-$(j+1)$ request.  Only a chunked, incomplete stage-specific request needs per-row KV state across batches. \\
Strict pipeline & A document prefix is independent of the current filter.  An action may mix new document prefill, $F_j$ for one ready row, and $F_{j+1}$ for a previously surviving row.  The state and $P_g$ explicitly prove which document prefixes were retained; actions may also evict a prefix and later recompute it. \\
Full fused speculation & At frontier one, an action evaluates all $n$ prompt branches from one document prefix.  If the document block was resident at launch, it contributes one fused load to $K_L$; if it is produced in the same supported fusion group, the write-through ledger contributes one materialization to $K_W$ and zero to $K_L$.  Every branch still contributes its own attention pairs.  A nonfused implementation is a distinct physical action with same-batch reload events and a different cost. \\
Partial speculation & At frontier $j$, an action chooses a contiguous block $F_j,\ldots,F_k$.  It shares the document prefix across those branches and produces frontier $(\ell,k+1)$ only if the entire block passes. \\
\bottomrule
\end{tabularx}
\caption{One LP framework covers the physical methods, but each method has a different feasible state-action set.}
\label{tab:physical-transition-sets}
\end{table}

Different document lengths may appear in one ragged batch.  Chunked prefill is represented by an action that advances only part of a prefix and a next state that retains the incomplete KV.  Shared prompt or document blocks are counted once only when the manifest contains one physical block and the modeled kernel reads that block once.

For a fair comparison, solve separately for task-first, strict pipeline, and full speculation.  A fourth solve may allow partial speculation.  If an engine may switch physical methods by row or stage, take the compatible union of their actions and solve again.  That result is a hybrid policy; it must not be labeled as any one of the baselines.

\subsection{The linear program}

Let $y_{sa}\ge0$ be a candidate fluid rate, in uses per second, for selecting action $a$ from cache state $s$, and let $\lambda\ge0$ be the candidate admitted and completed document rate.  For a fixed method $g$, solve
\begin{align}
\max_{\lambda,y}\quad &\lambda
\label{eq:fluid-objective}\\
\text{s.t.}\quad
&\lambda b_r
+\sum_{s\in\mathcal S_g}\sum_{a\in\mathcal A_g(s)}
y_{sa}\bar v_{sa,r}
=\sum_{s\in\mathcal S_g}\sum_{a\in\mathcal A_g(s)}
y_{sa}u_{sa,r}
&&\forall r\in\mathcal R_g,
\label{eq:logical-flow}\\
&\sum_{a\in\mathcal A_g(s)}y_{sa}
=\sum_{\bar s\in\mathcal S_g}
\sum_{a\in\mathcal A_g(\bar s)}
y_{\bar s a}P_g(s\mid\bar s,a)
&&\forall s\in\mathcal S_g,
\label{eq:cache-flow}\\
&\sum_{s\in\mathcal S_g}\sum_{a\in\mathcal A_g(s)}
y_{sa}\tau_{sa}\le1,
\label{eq:gpu-time}\\
&\lambda
=\sum_{s\in\mathcal S_g}\sum_{a\in\mathcal A_g(s)}
y_{sa}\bar h_{sa},
\label{eq:completion-consistency}\\
&y_{sa}\ge0,\qquad \lambda\ge0.
\end{align}

Equation~\eqref{eq:logical-flow} balances the \emph{mean rate} of every host-queue type; it does not certify that every requested action finds that work ready on every sample path.  Equation~\eqref{eq:cache-flow} balances expected entries to and exits from each observable cache state.  Unlike a deterministic arc balance, its right side averages over the outcomes revealed after the action.  Equation~\eqref{eq:gpu-time} allocates at most one GPU-second of batch work per wall-clock second.  Equation~\eqref{eq:completion-consistency} is often implied by the other balances, but retaining it catches model-generation errors.  Let $\lambda_g^\star$ denote the optimum of this expected-flow relaxation.

Let $w_{sa,\ell j}$ be the number of stage-$j$ evaluations on length type $\ell$ in action $(s,a)$.  For strict task-first or pipeline execution, every feasible $(\lambda,y)$ satisfies
\begin{equation}
\sum_{s,a}y_{sa}w_{sa,\ell j}
=\lambda p_\ell\pi_j.
\label{eq:strict-stage-rates}
\end{equation}
At the optimum set $\lambda=\lambda_g^\star$.  Full speculation instead evaluates every stage at rate $\lambda p_\ell$.  Partial speculation is handled by the action outcome law and \cref{eq:speculative-survival}.

When atomic task-first or atomic full speculation has no per-row KV state across batches, $\mathcal S_g$ collapses to one recurrent state and \cref{eq:cache-flow} is automatic.  The model then reduces to the simpler LP over batch frequencies.  Pinned task prefixes in that recurrent state are built by explicit finite-workload fill batches; their one-time construction vanishes only in the asymptotic rate.  The strict pipeline generally does not admit this reduction.

\subsection{A concrete two-stage cache transition}

For intuition, consider one length type, two filters, and a pipeline that retains every passing prefix until $F_2$ runs.  Let state $m$ mean that $m$ resident document prefixes have passed $F_1$ and await $F_2$.  A mixed action $(u,v)$ consumes $u$ new rows from the initial host queue, runs document prefill plus $F_1$ for them, and runs $F_2$ for $v\le m$ cached survivors.  If row outcomes are independent and
\begin{equation}
Z\sim\operatorname{Binomial}(u,s_1),
\end{equation}
then the next state is
\begin{equation}
m'=m-v+Z,
\label{eq:two-stage-cache-transition}
\end{equation}
subject to the batch and state fitting HBM.  Thus
\begin{equation}
P(m'\mid m,(u,v))=\Prb\{Z=m'-m+v\}.
\end{equation}
State balance implies $\E[v]=s_1\E[u]$: over the long run, $F_2$ must remove cached survivors at the same rate that $F_1$ creates them.  This is continuous batching---one physical batch can contain new $F_1$ work and ready $F_2$ work for different documents.

For this action, the number of completed documents is
\begin{equation}
h(u,v,Z)=v+(u-Z),
\label{eq:two-stage-completions}
\end{equation}
the $v$ rows evaluated by their final filter plus the $u-Z$ new rows rejected by $F_1$.  Combining $\E[v]=s_1\E[u]$ with \cref{eq:two-stage-completions} shows that long-run completions equal long-run admissions, as required by \cref{eq:completion-consistency}.

Task-first has the same logical $F_2$ rate, but the survivor is a fresh host-side request whose document tokens must be processed again under $F_2$.  Full speculation evaluates both branches for each of the $u$ documents in one fused action, so no survivor cache state exists.  If the pipeline may evict, the action's retention rule sends an unretained survivor to the uncached frontier; a later action must explicitly recompute its document prefix before using it.

\subsection{What the optimum proves}
\label{sec:lp-claim}

\begin{proposition}[Expected-flow throughput upper bound]
Fix finite sets $\mathcal L$ and $\mathcal S_g$, finite feasible action sets $\mathcal A_g(s)$, the joint outcome law, and deterministic constants $\tau_{sa}$.  Consider any admissible stochastic scheduler using only those actions for which the relevant time-average state-action rates exist and host-queue and cache inventories have sublinear boundary growth.  If its external admission counts satisfy
\begin{equation}
\frac{A_r(T)}{T}\longrightarrow\lambda b_r
\quad\text{for every initial type }r,
\label{eq:admission-mix-limit}
\end{equation}
no external arrivals enter other types, and long-run admissions equal completions, then its long-run document throughput is at most $\lambda_g^\star$.
\end{proposition}
\begin{proof}
Run the scheduler for wall time $T$, count every selected state-action pair, and divide by $T$.  The host-queue conservation identities hold exactly up to initial-minus-final inventory terms; \cref{eq:admission-mix-limit} supplies their external $\lambda b_r$ terms.  Cache-state entry and exit counts differ only by the initial and final state.  Because the scheduler is nonanticipating and outcomes follow the stated conditional law, the finite-state martingale law of large numbers makes empirical transition and production frequencies converge to the corresponding $P_g$, $\bar v$, and $\bar h$ averages along any convergent occupation subsequence.  Admission-minus-completion is another boundary inventory, and the sum of sequential batch times cannot exceed $T$.  Under the stated boundary assumptions, every limiting rate vector therefore satisfies \cref{eq:logical-flow,eq:cache-flow,eq:gpu-time,eq:completion-consistency}.  Its throughput is a feasible LP objective value and cannot exceed $\lambda_g^\star$.
\end{proof}

The converse is false without an additional argument.  A feasible vector $y$ can request work of type $r$ whenever the marginal cache state is $s$ even if, on actual sample paths, that queue is empty whenever $s$ occurs.  Equations~\eqref{eq:logical-flow} and \eqref{eq:cache-flow} do not encode this cache--queue correlation.  Thus $y$ is a fluid plan, not by itself a stationary randomized online policy.

\subsection{The exact queue-augmented occupation LP}

For comparison, specify a finite-state source/admission process $\eta$ with stationary initial-type mix $b$, bound every downstream host queue by a vector $Q$, and augment the observable state to
\begin{equation}
x=(s,q,\eta)\in\widehat{\mathcal S}_g.
\label{eq:queue-augmented-state}
\end{equation}
An action belongs to $\widehat{\mathcal A}_g(x)$ only when every non-source input is present in $q$.  For every possible outcome $\omega$, it must specify source admissions $e_{xa}(\omega)$ and satisfy the componentwise update
\begin{equation}
q'(\omega)
=q-u_{xa}+v_{xa}(\omega)+e_{xa}(\omega),
\qquad
0\le q'(\omega)\le Q.
\label{eq:bounded-queue-update}
\end{equation}
An action whose output can overflow is infeasible unless blocking or spilling is an explicit physical transition; no row may be silently dropped.  Source admissions enter only initial types $\mathcal R_g^0$.  The outcome updates the physical cache, queue inventories, and source state jointly according to $\widehat P_g(x'\mid x,a)$.  Let $z_{xa}$ be its occupation rate per wall-clock second, let $\bar e_{xa,r}=\E[e_{xa,r}(\omega)]$, and let $\bar h_{xa}$ be its expected completions.  The queue-augmented LP is
\begin{align}
\widehat\lambda_g^\star
=\max_{z\ge0}\quad
&\sum_{x,a}z_{xa}\bar h_{xa}
\label{eq:augmented-objective}\\
\text{s.t.}\quad
&\sum_{a\in\widehat{\mathcal A}_g(x)}z_{xa}
=\sum_{\bar x}\sum_{a\in\widehat{\mathcal A}_g(\bar x)}
z_{\bar x a}\widehat P_g(x\mid\bar x,a)
&&\forall x\in\widehat{\mathcal S}_g,
\label{eq:augmented-state-flow}\\
&\sum_{x,a}z_{xa}\bar e_{xa,r}
=b_r\sum_{x,a}z_{xa}\bar h_{xa}
&&\forall r\in\mathcal R_g^0,
\label{eq:augmented-source-mix}\\
&\sum_{x,a}z_{xa}\tau_{xa}\le1.
\label{eq:augmented-time}
\end{align}

\begin{proposition}[Exactness of the queue-augmented LP]
Fix the finite source, queue, cache, and action state spaces and a stated initial/source state $x_0$.  Assume positive action times and either that the augmented model is communicating/unichain from $x_0$ or that the LP is restricted to a recurrent class reached from $x_0$ with probability one and finite expected hitting time under a stated transient policy.  Then $\widehat\lambda_g^\star$ is the maximum average completion rate of that finite semi-Markov control model.  Moreover,
\begin{equation}
\widehat\lambda_g^\star\le\lambda_g^\star.
\label{eq:augmented-relaxation-order}
\end{equation}
\end{proposition}
\begin{proof}
Every admissible stationary policy with source mix $b$ induces state-action occupation rates satisfying \cref{eq:augmented-state-flow,eq:augmented-source-mix,eq:augmented-time}.  Conversely, from a feasible recurrent occupation vector define $\nu_x=\sum_a z_{xa}$ and choose action $a$ in state $x$ with probability $z_{xa}/\nu_x$ whenever $\nu_x>0$.  State balance gives its invariant visit rates, the source constraint fixes its admitted type mix, the time constraint converts visits to wall-clock rate, and queue feasibility is enforced pointwise because it is part of $\widehat{\mathcal A}_g(x)$.  The almost-sure finite-hitting assumption handles the transient fill path.  This is the standard finite semi-Markov occupation-measure equivalence.

For the inequality, sum any feasible augmented occupation vector over queue and source states while retaining its cache state and physical action signature.  Stationarity of the bounded queues together with \cref{eq:augmented-source-mix} gives \cref{eq:logical-flow} at $\lambda=\sum_{x,a}z_{xa}\bar h_{xa}$; marginalizing the joint state transition gives \cref{eq:cache-flow}; completion and time give \cref{eq:completion-consistency,eq:gpu-time}.  The projected vector is feasible for the expected-flow relaxation with the same completion rate.
\end{proof}

The main experiment uses the smaller expected-flow LP as a planning relaxation and replays its rates with real queue inventories.  A successful replay establishes only the emitted finite schedule and its latency.  The exact queue-augmented LP is reserved for small instances because its state space grows with the Cartesian product of cache and queue inventories.  It establishes $\widehat\lambda_g^\star$ for its own infinite-horizon bounded-queue control problem, not the relaxed target $\lambda_g^\star$ and not the one-shot 10,000-document makespan; the finite Bellman model remains the exact one-shot reference.

The result is deliberately narrower than finite-workload optimality.  It uses expected host-queue production and a marginal invariant cache flow.  A finite execution has integral documents, finite queues, and random survivor counts.  Therefore $N/\lambda_g^\star$ is an asymptotic fluid speed-of-light target, not automatically a valid finite-$N$ bound and not an attained makespan for $N=10{,}000$.

\subsection{From the LP to a finite 10,000-document execution}

Define the asymptotic fluid latency target
\begin{equation}
T_{g}^{\mathrm{LP}}(N)=\frac{N}{\lambda_g^\star}.
\label{eq:lp-latency-target}
\end{equation}
The proved throughput direction is equivalently
\begin{equation}
\limsup_{T\to\infty}\frac{C_g(T)}{T}
\le\lambda_g^\star
\qquad\text{almost surely},
\label{eq:fluid-asymptotic}
\end{equation}
where $C_g(T)$ is the number of completions by time $T$, under the convergence and boundary assumptions of the proposition.  For the hitting time $T_{g,N}=\inf\{T:C_g(T)\ge N\}$ of any infinite completing sequence, the precise inverse statement is
\begin{equation}
\liminf_{N\to\infty}\frac{T_{g,N}}{N}
\ge\frac{1}{\lambda_g^\star}
\qquad\text{almost surely}.
\label{eq:inverse-throughput-bound}
\end{equation}
Equality is an empirical or separately proved achievability claim, not a consequence of the LP.  The actual 10,000-document candidate is obtained by replay.  Define
\begin{equation}
\nu_s^\star=\sum_{a\in\mathcal A_g(s)}y_{sa}^\star,
\qquad
\mu^\star(a\mid s)=\frac{y_{sa}^\star}{\nu_s^\star}
\quad\text{when }\nu_s^\star>0.
\label{eq:lp-action-proportions}
\end{equation}
Then:
\begin{enumerate}
  \item Initialize the exact remaining counts $N_\ell$ and empty downstream queues and cache state.
  \item Fill the pipeline using feasible actions; admit documents by length type so cumulative admissions track $p_\ell$ without ever exceeding $N_\ell$.
  \item Let $n_s$ be replay visits to state $s$ and $n_{sa}$ uses of action $(s,a)$.  Among actions whose physical and logical inputs are ready, prefer the one maximizing the state-conditioned deficit $\mu^\star(a\mid s)n_s-n_{sa}$.
  \item Apply the realized filter outcomes, the action's contingent final resident set, and the resulting queue/cache transition.  If $\nu_s^\star=0$, the target action is unready, or replay reaches a state absent from the LP support, use the fixed safe fallback and record that whole batch as repair.
  \item Keep $\mathbf s$ and the target mixture fixed throughout the paired replay; method-specific observations never update selectivity.  Stop admissions after exactly $N$ rows and drain all unfinished work.
\end{enumerate}
The action generator must provide, from every replay-reachable state, a safe \emph{method-compatible} fallback whose repeated use makes strict progress and terminates: task-first work for task-first, strict $k=1$ work for pipeline, and a smaller or chunked full-speculative action for the full-speculation baseline.  The runtime is an ordinary continuous-batching scheduler guided by LP frequencies; it makes each decision only after the preceding batch outcomes are visible.  It emits a concrete manifest.  The authoritative analytical latency is
\begin{equation}
T_{g,N}^{\mathrm{construct}}
=\sum_{t=1}^{T}\tau_\theta(B_t).
\label{eq:finite-latency-decomposition}
\end{equation}
For reporting, partition the batch IDs into disjoint sets $\mathcal I_{\mathrm{fill}},\mathcal I_{\mathrm{core}},\mathcal I_{\mathrm{repair}},\mathcal I_{\mathrm{drain}}$ using a fixed phase rule and define each component as the sum of whole-batch costs over its set.  A repair batch can still perform useful core work; its nonlinear cost is never split token by token.  These four sums equal \cref{eq:finite-latency-decomposition} by construction.

This is a feasible finite latency only after validation, not a proof of the finite Bellman optimum.  The signed difference from \cref{eq:lp-latency-target} is a \emph{deviation}, not a guaranteed nonnegative overhead or gap, because the asymptotic target is not a finite-$N$ lower bound.  Kernel inefficiency remains a separate measured quantity.

\subsection{Generating the state-action table}

The LP is linear because every nonlinear hardware maximum is evaluated before optimization: each concrete feasible batch has a constant $\tau_{sa}$.  The difficult part is that there may be many states and actions.  The first implementation should make that combinatorial object explicit rather than introduce another headline formulation:
\begin{enumerate}
  \item enumerate reachable cache states under the chosen quanta and HBM bound;
  \item generate physically legal batches from each state;
  \item compute their outcome transition probabilities and batch costs;
  \item solve the resulting sparse LP;
  \item if enumeration is too large, begin with a restricted state-action graph and repeatedly search jointly for omitted reachable states and transition actions that improve the current solution.
\end{enumerate}
Action pricing from a fixed state is a local resource-constrained packing problem, but full certification also requires state completeness.  The full discretized relaxation is certified only by exhaustive reachable-state enumeration with exact action pricing from every state, or by a joint state/action generation proof that no omitted reachable transition column can improve the objective.  If generation is heuristic, the result is only the optimum of the supplied restricted expected-flow model.  In particular,
\begin{equation}
\lambda_{g,\mathrm{restricted}}^\star
\le\lambda_{g,\mathrm{full}}^\star.
\label{eq:restricted-full-order}
\end{equation}
The restricted value remains an upper bound for stable schedulers constrained to its supplied state-action graph, but it is neither an upper bound for schedulers using omitted actions nor an achievability proof.  Only a successful finite replay supplies an achieved schedule.

If exact empirical lengths create too many types, use lower and upper representatives $d_i^-\le d_i\le d_i^+$ for each bin.  Provided the same action templates remain valid and work and memory are monotone in length,
\begin{equation}
\lambda_g^\star(\mathbf d^-)
\ge
\lambda_g^\star(\mathbf d)
\ge
\lambda_g^\star(\mathbf d^+).
\label{eq:length-rate-bracket}
\end{equation}
The lower endpoints are optimistic and the upper endpoints conservative.  A midpoint or mean length provides no such bound because attention work is nonlinear.  Every result records the token-chunk quantum, cache quantum, length types, states, actions, transition probabilities, LP residuals, and action-search termination status.

\section{Resource lower bounds}
\label{sec:resource-lower-bounds}

Fix the realized workload, outcomes, initialization convention, and physical method $g$, and let $\Sigma_g$ be its feasible finite schedules.  For $\sigma=(B_1,\ldots,B_T)\in\Sigma_g$, define
\begin{equation}
\begin{split}
U(\sigma)&=\sum_tU(B_t),
\qquad
A(\sigma)=\sum_tA(B_t),\\
V_{KV}(\sigma)&=\sum_tB_{KV}(B_t),
\qquad
m(\sigma)=\sum_t\one\{U(B_t)>0\}.
\end{split}
\label{eq:schedule-resource-ledgers}
\end{equation}
These quantities depend on the schedule: eviction can cause recomputation, speculation changes prompt work, and physical fusion changes HBM traffic.  A \emph{certified lower ledger} for method $g$ is any quadruple $(\ell_g^U,\ell_g^A,\ell_g^V,\ell_g^m)$ satisfying
\begin{equation}
U(\sigma)\ge\ell_g^U,
\qquad
A(\sigma)\ge\ell_g^A,
\qquad
V_{KV}(\sigma)\ge\ell_g^V,
\qquad
m(\sigma)\ge\ell_g^m
\quad\forall\sigma\in\Sigma_g.
\label{eq:certified-resource-ledgers}
\end{equation}
The strongest separate choices are the four minima of the corresponding ledgers over $\Sigma_g$, but the implementation may use weaker values that have simpler proofs.  The four values need not be attained by one schedule.  Monotonicity is sufficient for the bound
\begin{equation}
LB_{g,\mathrm{res}}
=
\max\left\{
\frac{2P\ell_g^U}{R_D},
\frac{\ell_g^mW_{\mathrm{run}}}{BW}
\right\}
+
\max\left\{
\frac{4Lw_Q\ell_g^A}{R_A},
\frac{\ell_g^V}{BW}
\right\}.
\label{eq:resource-lb}
\end{equation}

Dense-work and attention ledgers can often be proved by removing recomputation and counting only causally required calls under the stated method and realized outcomes.  That reasoning does \emph{not} automatically lower-bound KV traffic: recomputation can exchange extra dense work for fewer persistent KV reads and writes.  The safe default is therefore $\ell_g^V=0$.  A positive value is used only after a separate block-level lemma proves compulsory materializations or loads under the selected physical convention.  Each reported bound stores the four ledger formulas, proofs or lemma identifiers, and inputs.  It never copies totals from the candidate schedule being certified.

The nonempty-batch ledger also requires a derivation.  The universally safe value is $\ell_g^m=1$ for a nonempty workload.  If every batch has a certified upper bound $\bar U_g$ on its new-token count, then
\begin{equation}
\ell_g^m
=
\max\left\{1,
\left\lceil\frac{\ell_g^U}{\bar U_g}\right\rceil\right\}
\label{eq:batch-count-token-bound}
\end{equation}
is valid for $\bar U_g>0$.  An explicit new-token cap gives $\bar U_g=C_U$.  HBM alone gives such a bound only after a kernel-specific premise---for example,
\begin{equation}
\max_r\sum_{b\in\mathcal L_B(r)}\residentBytes(b)
\ge\gamma_gU(B)
\quad\text{for every feasible }B
\end{equation}
with $\gamma_g>0$---is proved.  One may then set $\bar U_g=\lfloor(M-W_{\mathrm{mem}}-S)/\gamma_g\rfloor$.  Without an explicit cap or such a premise, use $\ell_g^m=1$.  This rule avoids inferring a batch count from an average document length or from schedule-dependent cache occupancy.

The bound may remain loose because it pools resources globally and because its four minima can be mutually incompatible.  Looseness is acceptable; invalid accounting is not.

\begin{proposition}[Equality certificate]
If a feasible schedule $\sigma\in\Sigma_g$ satisfies $T_0(\sigma)=LB_{g,\mathrm{res}}$, then $\sigma$ is optimal within method $g$ under the analytical cost $\tau_0$.
\end{proposition}
\begin{proof}
For every schedule, $\sum_t\max\{x_t,y_t\}\ge\max\{\sum_tx_t,\sum_ty_t\}$.  Applying this inequality to the dense group and the attention group in every batch gives
\begin{equation}
T_0(\sigma)
\ge
\max\left\{\frac{2PU(\sigma)}{R_D},\frac{m(\sigma)W_{\mathrm{run}}}{BW}\right\}
+\max\left\{\frac{4Lw_QA(\sigma)}{R_A},\frac{V_{KV}(\sigma)}{BW}\right\}.
\end{equation}
Substituting the four certified lower ledgers yields $T_0(\sigma)\ge LB_{g,\mathrm{res}}$.  The exhibited schedule reaches this lower bound.
\end{proof}

A positive gap to \cref{eq:resource-lb} does not prove that the schedule is suboptimal; it only means that the available lower bound does not certify optimality.  For random online outcomes, compute the bound scenario by scenario before averaging; a bound formed from expected resource totals is a different claim and requires its own proof.

\section{Experimental program}

\subsection{Research questions}

The first numerical study should answer:
\begin{enumerate}
  \item For $n=2$, what fluid throughput upper bound and constructed finite rate does each physical method obtain as selectivity and the realized document-length distribution change?
  \item After the LP rates guide a validated execution on 10,000 documents, how do the queue-starvation, fill, repair, and drain phase shares compare with the asymptotic fluid LP target?
  \item Which state-conditioned batches does the LP relaxation select, and which batches does the finite replay actually execute when finite KV capacity causes the pipeline method to retain, evict, or recompute a document prefix?
  \item How much does chunked prefill improve the best batch mixture relative to atomic requests under a throughput objective?
  \item Which break-even conclusions change between Qwen3-4B and Qwen3-32B, and between H100 and L40S?
  \item For $n>2$, when does bounded speculative lookahead beat both strict pipelining and full speculation, and when does a hybrid mixture use more than one physical method?
\end{enumerate}

\subsection{Constructing the 10,000-document workload}

The motivating IMDb review histogram is measured in whitespace-separated words.  It must not be used as though its x-axis were Qwen tokens.  The preferred construction is:
\begin{enumerate}
  \item Pin the dataset release, archive hash, included splits, and sampling pool.  IMDb filenames are unique only within a split and sentiment directory, so the row identifier includes that full relative path.
  \item Sample 10,000 unique rows uniformly without replacement from the pinned pool using a recorded seed.
  \item Store the raw text hash and tokenize each review with the exact tokenizer revision used by the model.
  \item Record every $d_i$ and reject or truncate only under a written context-window rule.
  \item Use the identical realized $\mathbf d$ for every policy, model, device, and selectivity comparison.
\end{enumerate}
If only the histogram is available, the fallback must specify a within-bin distribution, a model for the open-ended tail, and a measured words-to-Qwen-tokens conversion.  That synthetic workload is labeled as such.

The canonical workload file contains
\begin{equation}
(\texttt{dataset\_release},\texttt{split},\texttt{relative\_path},
\texttt{text\_hash},d_i,\texttt{sample\_seed},\texttt{tokenizer\_revision}).
\label{eq:workload-schema}
\end{equation}
The filter file stores the exact prompt text, a prompt hash, $p_j$ under each tokenizer, the required answer format, and the logit position used for classification.  Results are not reproducible from ``approximately 50 tokens'' alone.

\subsection{Outcome generation}

The expected-flow LP uses the selectivity vector $\mathbf s$ directly; it does not need sampled filter outcomes.  The experiment pins the full selectivity grid before running.  Samples are required only when the LP mixture is converted into a finite online execution.  For $n=2$ and selectivity $s$, generate $R$ independent Bernoulli vectors $X^{(r)}$ using recorded seeds, independently of $\mathbf d$.  Record $R$ and the seeds, and reuse each vector across every physical method so finite-schedule differences are paired rather than confounded by different survivors.  Full speculation performs outcome-independent model work, although its final SQL outputs still use the sampled outcomes.

For $n>2$, the baseline generator samples the full latent matrix $X_{ij}$ in advance, independently across rows and stages with Bernoulli parameter $s_j$.  The value $s_n$ is specified for SQL-output generation even though it cannot create later inference work.  Entries after a row's first failure do not affect conjunction semantics, but storing them gives pipeline, partial speculation, and full speculation one coupled scenario.  If a correlated joint model is studied instead, its conditional generator and parameters replace the independence assumption and are used consistently to build $P_g$, $\bar v$, and $\bar h$.  The file records $(s_1,\ldots,s_n)$, the generator version, random seed, and every $X_{ij}$.

\subsection{Hardware and model configuration}

Each run records:
\begin{itemize}
  \item model repository and immutable revision;
  \item tokenizer repository and immutable revision, normalization settings, BOS/EOS insertion, chat template, and the exact rendered sequence for every physical method;
  \item $P,L,h,n_Q,n_{KV},d_h,w_Q,L_{ctx}$;
  \item total loaded weight bytes $W_{\mathrm{mem}}$ and per-batch compulsory transformer-weight traffic $W_{\mathrm{run}}$;
  \item KV dtype, $q_{KV}$, $\kappa_{\mathrm{meta}}$, per-page resident allocation sizes, transfer-byte rules for occupied ranges, scale granularity, and transaction overhead convention;
  \item device SKU, physical memory $M$, bandwidth input $BW$, and dense/attention rate inputs, including whether $R_A$ is a nominal ceiling or a measured microbenchmark;
  \item reserve $S$, scratch convention, ideal-versus-measured KV traffic convention, cache page size, chunk quantum $\delta$, length-bin boundaries, active scheduling window, and any sequence-count or new-token cap;
  \item LP solver and version, feasibility and optimality tolerances, state/action enumeration limits, improving-action search method and termination certificate, and every random tie-break seed;
  \item exact filter prompts and hashes, selectivity grid, outcome repetition count, and all random seeds.
\end{itemize}
All primary speed-of-light curves use FP8 weights and FP8 KV.  Other KV precisions are outside the initial experiment.

\subsection{Computation sequence}

The primary implementation path is:
\begin{enumerate}
  \item Tokenize the 10,000 documents, construct exact length types or conservative length brackets, and freeze the model and device configuration.
  \item Implement one independently tested batch-cost routine.  Given a batch manifest, it uses $w_Q=n_Qd_h$ and the selected KV convention to return dense work, attention work, physical KV traffic, peak HBM, and $\tau_{sa}$.
  \item For each physical method separately, generate an initial set of observable cache states and feasible state-conditioned actions and solve \cref{eq:fluid-objective,eq:completion-consistency} as an expected-flow relaxation.
  \item Search jointly for omitted reachable cache states and feasible transition actions that would improve the current LP.  Add them and re-solve.  Stop only after the configured search is exhausted; record whether state discovery and action pricing were exact or heuristic, together with the final LP residuals.
  \item Convert $y_{sa}^\star$ into a finite 10,000-document execution, run the state-conditioned continuous-batching tracker on coupled outcome scenarios, and emit every executed batch to the manifest.
  \item Independently build certified method-specific lower ledgers, then recompute each manifest's feasibility and latency.  Report the resource lower bound, asymptotic fluid LP target, and constructed fill/core/repair/drain time as distinct quantities.
  \item As a validation exercise rather than the large-scale solver, hand-enumerate tiny cases and compare the LP's long-run mixture with repeated finite Bellman solutions as the replicated workload grows.
\end{enumerate}

The comparison therefore entails three separate LP solves---one with task-first actions, one with strict-pipeline actions, and one with full-speculation actions.  A fourth, optional solve over a compatible union of those actions answers a different question: the performance of an engine allowed to mix physical methods.  It is not a replacement for the three baseline solves.

An LP solver proving zero numerical gap establishes optimality only for the expected-flow relaxation over the states and actions supplied to it.  Optimality over the full discretized relaxation additionally requires exhaustive reachable-state generation plus exact action pricing, or an equivalent joint certificate covering omitted states and transitions.  The relaxed rate $\lambda_g^\star$ is called achievable only if a valid queue-augmented policy has $\widehat\lambda_g^\star=\lambda_g^\star$ or a separate asymptotic construction proves equality.  A finite replay instead certifies one finite schedule.  Heuristic state/action search certifies neither full-relaxation optimality nor an executable rate.

\subsection{Schedule manifest}

The planning output records $(s,a,y_{sa}^\star)$ for every used state-action pair, each action's complete contingent mapping $R_{sa}(\omega)$, $\lambda_g^\star$, whether the state-action set is exhaustive, and the host-flow, cache-flow, completion, and GPU-time residuals.  The realized finite execution uses a block-event manifest rather than a list of logical prefixes.  It records only the outcome branch actually selected.  For each batch $t$ it emits:
\begin{itemize}
  \item \texttt{batch\_id}, policy family, revealed-history hash, \texttt{host\_before\_pre}, and \texttt{host\_at\_launch};
  \item \texttt{resident\_before}, \texttt{pre\_evictions}, and \texttt{resident\_at\_launch};
  \item for every physical block version, a stable \texttt{block\_id}, immutable semantic identity, owner, stage or shared-prefix role, token span, $\residentBytes(b)$, producer operation, birth ordinal, and free ordinal;
  \item every temporary or scratch HBM allocation with allocation ID, role, $\residentBytes(b)$, birth ordinal, and free ordinal;
  \item for every operation, an \texttt{operation\_id}, topological ordinal, document ID, filter stage, token interval, causal parent block IDs, and a supported \texttt{fusion\_group\_id};
  \item every load and store event with block ID, event ordinal, kernel/load group, transferred byte range, $\xferBytes(e)$, and whether a load source is launch-resident or same-batch-produced;
  \item ephemeral terminal leaf positions, generated logits, revealed outcomes, the chosen contingent \texttt{resident\_after} set, and the resulting host queues and cache state;
  \item $U_t,A_t,K_{L,t},K_{W,t},B_{KV,t}^{\mathrm{xfer}},M_t^{\mathrm{peak}},D_t,H_t,$ and $\tau_t$.
\end{itemize}
The validator reconstructs causal visibility from these records and checks every declared fusion/load group against the supported kernel plan; groups cannot be inferred from the causal graph alone.  Equal logical token strings do not imply one physical block.  Several consumers receive one-load credit only when they name the same versioned block and validated load group.  Under the primary write-through convention, a block produced and consumed in one fused group has one store and no load; a separate group must reload it.  The outcome is revealed after compute, the contingent $\texttt{resident\_after}(\omega)$ branch is then applied, and that set is exactly the next cache state.  There is no second, ambiguous post-outcome eviction field.

\subsection{Plots}

The initial paper should contain:
\begin{itemize}
  \item asymptotic fluid LP latency target $N/\lambda_g^\star$ and constructed finite latency versus first-stage selectivity for the three two-stage methods;
  \item the same two latency layers at document-length multipliers $1\times$, $10\times$, and $100\times$; any point with $\alpha d_i+p_j>L_{ctx}$ is marked context-infeasible rather than silently truncated;
  \item a selectivity-by-length break-even map for each latency layer;
  \item atomic versus chunked LP latency, with the chunk quantum stated;
  \item peak resident KV and recomputed document tokens;
  \item the LP mixture over state-conditioned batches and, for the constructed execution, the number of batches, new-token fill, and active sequences per batch;
  \item hardware resource bound, asymptotic fluid LP target, constructed finite schedule, and measured engine latency on the same axes;
  \item fill, core, repair, and drain whole-batch phase shares of constructed latency;
  \item H100 versus L40S and Qwen3-4B versus Qwen3-32B.
\end{itemize}
Latency plots use a logarithmic y-axis.  Captions state whether curves are resource bounds, expected-flow LP targets for an enumerated state-action set, constructed finite schedules, calibrated estimates, or measured engine results.  A fluid target is labeled ``achievable'' only when a queue-augmented policy matches its value or a separate asymptotic construction proves equality; a finite replay establishes only its plotted finite schedule.

\subsection{Break-even reporting}

For policies $g$ and $h$, define
\begin{equation}
\Delta^{\mathrm{LP}}_{g,h}(\mathbf s,\mathbf d)
=\frac{N}{\lambda_g^\star}-\frac{N}{\lambda_h^\star}
\label{eq:break-even}
\end{equation}
and define $\Delta^{\mathrm{construct}}_{g,h}$ as the paired difference between the two constructed finite latencies on the same outcome scenario.  The LP-target and finite break-even points need not coincide.  Changes of the active LP actions make the LP curve piecewise, while queue readiness, integer batches, random survivors, and fill/drain effects make the finite curve nonsmooth.  Report an interval or grid cell in which the sign changes, together with length-quantization bounds, action-search status, signed replay deviation, and Monte Carlo uncertainty where applicable.  Do not report a high-precision root unsupported by those resolutions.

\subsection{Monte Carlo and optimization uncertainty}

For a fixed realized length vector, the expected-flow LP value $N/\lambda_g^\star$ is deterministic at a stated $\mathbf s$.  For $R$ coupled outcome scenarios, estimate the expected latency of the finite LP-guided execution by
\begin{equation}
\widehat L^{\mathrm{construct}}_g(\mathbf d)
=\frac{1}{R}\sum_{r=1}^{R}T_{g,N}^{\mathrm{construct},(r)}.
\label{eq:mc-estimator}
\end{equation}
Use paired differences $T_{g,N}^{\mathrm{construct},(r)}-T_{h,N}^{\mathrm{construct},(r)}$ for method comparisons and report their sampling interval.  Random-outcome uncertainty, document-sampling uncertainty, length-quantization bounds, and incomplete action search are four different sources of uncertainty and are reported separately.  If tiny-instance Bellman optima are also computed, they are labeled as a validation experiment and are not substituted for the 10,000-document LP result.

\section{Validation and claim discipline}

Before accepting a numerical result, verify:
\begin{itemize}
  \item Every document has its actual token length; no solver input substitutes a mean.
  \item Every generated state and action obeys the chosen physical method.  In particular, task-first receives no cross-stage document-prefix reuse, strict pipeline crosses no unresolved filter gate, and speculation receives a one-load credit only when one physical shared block appears in its manifest.
  \item The LP's host-queue, cache-state, completion, and GPU-time residuals are below stated tolerances.
  \item The LP result is labeled as an expected-flow relaxation.  Achievability of $\lambda_g^\star$ requires equality with a valid queue-augmented policy or a separate asymptotic construction; a concrete replay certifies only its finite manifest and latency.
  \item Every positive-rate action starts in a state with the blocks it uses, every possible outcome leads to a valid next state, and every host-queue consumption has the required upstream supply.
  \item Attention work is summed separately for each sequence, with no padding to the longest sequence and no attention across packed documents.
  \item Attention FLOPs use $w_Q=n_Qd_h$ while KV bytes use $n_{KV}d_h$; the two grouped-query widths are not conflated.
  \item Attention work across all chunks equals the work for the same unchunked left-to-right sequence.
  \item Every required input KV block is resident or produced earlier in the same batch.
  \item \texttt{resident\_before}, pre-evictions, and \texttt{resident\_at\_launch} reconcile exactly; all remaining launch blocks count at peak.
  \item Shared blocks are counted once only when the batch manifest uses one physical block and one load group.  Nonfused consumers create distinct load events.
  \item Every new block that can be retained under any outcome is materialized and charged before that outcome; retention changes persistence, not the action's deterministic cost.
  \item Peak memory includes weights, reserve, resident blocks, temporary new blocks, and prompt branches.
  \item A non-speculative operation never crosses an unresolved outcome gate.
  \item An online decision depends only on its observed history.
  \item Evicted incomplete prefixes lose progress; evicted complete document prefixes must be recomputed before reuse.
  \item The independent validator reproduces $U,A,K_L,K_W,B_{KV}^{\mathrm{xfer}},M^{peak}$ and every $\tau_t$.
  \item ``Optimal fluid LP value'' is claimed only for a stated state-action set with a solved LP; optimality over the full discretized relaxation additionally requires an exact joint reachable-state and action-pricing certificate.
  \item ``Finite optimum'' is claimed only when a finite feasible schedule matches a valid finite lower bound.  An LP-guided finite schedule is otherwise labeled ``constructed.''
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
  \item The main LP uses expected survivor flow and omits host-queue inventories from the state.  It upper-bounds sustainable throughput but can be optimistic because marginal cache and queue balances need not be jointly realizable on sample paths.
  \item The LP is exact only as an optimization of its finite expected-flow relaxation.  Length brackets quantify length discretization; an incomplete action search must be labeled as such.  Queue augmentation is required for an exact finite-state occupation-measure control model.
  \item The finite runtime that tracks LP frequencies is a feasible continuous-batching policy only after its manifest passes validation.  It is not a proof of finite online optimality.  Queue starvation, integer rounding, and pipeline fill and drain are measured explicitly.
  \item The finite Bellman reference is exponentially large and is retained only for small validation instances.
\end{itemize}

\section{Conclusion}

An $n$-stage AI filter query is simultaneously a short-circuit evaluation problem, a ragged GPU-packing problem, and a KV inventory problem.  The two-stage case is a useful warm-up, but the central computational object is the flow of length-specific documents through host queues and observable physical cache states.  A valid batch mixture may mix document lengths, divide long prefills into chunks, retain selected prefixes across gates, and speculate over a chosen block of future filters.

The implementation should therefore begin with the batch-cost and feasibility checker, the physical-method-specific state-action generators, and the expected-flow LP.  The LP is solved separately for task-first, pipeline, and speculation; an optional union solve measures a hybrid engine.  Its rates are planning targets, then converted into an explicit 10,000-document continuous-batching execution whose fill, repair, queue-starvation, and drain costs are visible.  The resulting plots must place the resource bound, asymptotic fluid LP target, constructed finite schedule, and eventual engine measurement on separate labeled layers.  That sequence can support quantitative conclusions about when pipeline retention or fused speculation wins on H100 and L40S without claiming that mean flow balance is an executable scheduler or that an existing engine attains the analytical schedule.

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
$\mathcal L,N_\ell,p_\ell$ & Empirical document-length types, count of type $\ell$, and its workload mass. \\
$\mathcal R_g,b_r$ & Host-queue types for method $g$ and their external input mix. \\
$\mathcal S_g,\mathcal A_g(s)$ & Observable cache states and feasible next-batch actions for method $g$. \\
$u_{sa,r},\bar v_{sa,r},\bar h_{sa}$ & Host work consumed, expected host work produced, and expected documents completed by $(s,a)$. \\
$P_g(s'\mid s,a)$ & Outcome-induced next-cache-state probability. \\
$y_{sa},\lambda_g^\star$ & Expected-flow action rate and optimum throughput upper bound of the stated LP relaxation. \\
$z_i$ & Next unresolved stage for document $i$; $n+1$ means complete. \\
$r_i$ & Policy-specific resident prefix progress for document $i$. \\
$\mathcal K_t$ & Physical KV blocks in the observed state before step $t$; pre-evictions produce the launch set. \\
$Q(B)$ & New tokens evaluated in batch $B$. \\
$U(B)$ & Number of new token nodes in batch $B$. \\
$A(B)$ & Number of allowed query-key pairs in batch $B$. \\
$K_L(B),K_W(B)$ & Token spans summed over all HBM load events and over new KV positions stored to HBM. \\
$\residentBytes(b),\xferBytes(e)$ & HBM allocation size of block $b$ and bytes transferred by event $e$. \\
$P,W_{\mathrm{mem}},W_{\mathrm{run}}$ & Repeated dense parameter count, resident weight bytes, and per-batch compulsory transformer-weight traffic. \\
$n_Q,n_{KV},d_h,w_Q$ & Query heads, KV heads, head width, and total query width $w_Q=n_Qd_h$. \\
$\kappa,\kappa_{\mathrm{meta}}$ & Ideal KV bytes per cached token and metadata/scale overhead included in it. \\
$M,S$ & Physical GPU memory and non-weight/non-KV reserve. \\
$R_D,R_A,BW$ & Dense compute, attention compute, and memory-bandwidth ceilings. \\
$D(B),H(B),\tau(B)$ & Dense time, attention time, and assigned batch latency. \\
$\ell_g^U,\ell_g^A,\ell_g^V,\ell_g^m$ & Independently certified lower ledgers for new tokens, attention pairs, KV bytes, and nonempty batches. \\
$\sigma,T(\sigma)$ & Concrete schedule and its makespan. \\
$\OPT_{\mathrm{off}},\OPT_{\mathrm{on}}$ & Clairvoyant and online optima under the stated model. \\
\bottomrule
\end{longtable}

\section{Independent schedule-checker contract}

The optimizer and validator must not share the objective implementation.  Before constructing a finite schedule, an LP checker recomputes the host-queue balance, cache-state balance, completion consistency, and GPU-time use from the saved state-action table and solution vector.  It reports maximum absolute and scaled residuals and verifies that every positive-rate action has passed the physical-method-specific feasibility checker.

The finite-schedule validator consumes the immutable workload, outcome scenario, hardware-model configuration, and schedule manifest.  It performs the following deterministic pass:
\begin{enumerate}
  \item Initialize an empty KV cache, the exact initial host queues, and zero document progress.  Task prefixes and every other reusable block must be produced by explicit fill batches because $T_{\mathrm{init}}=0$.
  \item For each batch in order, verify \texttt{resident\_before} and \texttt{host\_before\_pre}, apply \texttt{pre\_evictions}, create the required uncached items, and confirm both \texttt{resident\_at\_launch} and \texttt{host\_at\_launch} before applying batch consumption.
  \item Reconstruct every operation's causal parents.  Each parent must be launch-resident and loaded, produced earlier in the same supported fusion group and streamed, or produced earlier in another group, stored, and then reloaded.  Verify information availability and the selected policy's speculation rules.
  \item Validate every declared fusion and load group against the supported kernel plan.  Verify store-before-load and last-use-before-free ordering, transferred byte ranges, and immutable versioned block identities.  Recompute $K_{L,t}$, $K_{W,t}$, and $B_{KV,t}^{\mathrm{xfer}}$ from the declared physical events.
  \item Verify that every new KV position with a causal descendant has a store under the primary write-through convention and that every unstored position is a terminal leaf.  Recompute KV and scratch allocation lifetimes and $M_t^{\mathrm{peak}}$ from $\residentBytes(b)$; no unconstrained on-chip persistence is permitted.  In the state-action table, every block named by any contingent $R_{sa}(\omega)$ must remain live through the outcome boundary for every $\omega$.
  \item Apply completions, reveal outcomes, and then apply the one contingent \texttt{resident\_after} set.  Confirm the next host queues and cache state, including exactly one uncached host item for every evicted unfinished row not consumed in the action.
  \item Recompute $D_t,H_t,\tau_t$ from the configuration rather than trusting solver fields.
  \item At termination, verify exactly $N$ distinct document-ID admissions, exactly one completion per admitted row, every logically required result, empty host queues, and no unfinished resident KV.
\end{enumerate}
The checker returns a batch-indexed error, not a Boolean alone, so a faulty formulation can be diagnosed.

\section{Recommended implementation artifacts}

The computational repository should contain:
\begin{itemize}
  \item \texttt{workloads/documents.parquet}: immutable document IDs, hashes, and token lengths;
  \item \texttt{workloads/outcomes/}: coupled outcome matrices and selectivity metadata;
  \item \texttt{configs/prompts/}: exact filter text, rendered templates, prompt hashes, answer convention, and classification position;
  \item \texttt{configs/models/}: immutable Qwen and tokenizer revisions, architecture, special-token policy, and measured weight footprints;
  \item \texttt{configs/devices/}: H100 and L40S capacities, rates, reserve, cache-page convention, KV dtype, resident-allocation measurements, and transfer-byte measurements;
  \item \texttt{configs/solvers/}: discretization quanta, state/action limits, solver versions and tolerances, and search termination criteria;
  \item \texttt{optimizer/state\_actions/}: method-specific state-action generators and immutable state-action tables;
  \item \texttt{optimizer/steady\_state\_lp/}: expected-flow LP builder, solver interface, saved primal solutions, dual values, and residual reports;
  \item \texttt{optimizer/queue\_augmented\_lp/}: optional exact bounded-queue occupation model for small-instance checks;
  \item \texttt{optimizer/action\_search/}: exact or heuristic improving-action search with an explicit termination status;
  \item \texttt{runtime/continuous\_batching/}: LP-rate tracker, queue manager, finite-workload repair, and manifest writer;
  \item \texttt{reference/finite\_bellman/}: optional exact small-instance validation code;
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